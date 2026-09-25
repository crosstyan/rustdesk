// Jetson NvBufSurface / VIC / V4L2 encoder shim for jetson.rs.
//
// The Tegra NVENC is driven through NVIDIA's libv4l2 plugin (libnvv4l2): /dev/v4l2-nvenc is only a
// placeholder node, every v4l2_* call is served in user space by the NvMM stack. Input frames are
// NvBufSurface dma-bufs queued with V4L2_MEMORY_DMABUF, so a frame converted on the VIC never
// passes through the CPU.

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#include <libv4l2.h>
#include <linux/videodev2.h>

#include "nvbufsurface.h"
#include "nvbufsurftransform.h"
#include "v4l2_nv_extensions.h"

#define JZ_MAX_OUT 8
#define JZ_NCAP 4
#define JZ_CAP_SIZE (4 << 20)

enum { JZ_H264 = 0, JZ_H265 = 1, JZ_AV1 = 2 };
enum { JZ_FMT_NV12 = 0, JZ_FMT_BGRA = 1, JZ_FMT_BGRX = 2, JZ_FMT_RGBA = 3, JZ_FMT_RGBX = 4 };

static void set_err(char *err, size_t len, const char *what) {
  if (err && len) snprintf(err, len, "%s: %s", what, strerror(errno));
}

static NvBufSurfaceColorFormat color_format(int fmt) {
  switch (fmt) {
  case JZ_FMT_BGRA: return NVBUF_COLOR_FORMAT_BGRA;
  case JZ_FMT_BGRX: return NVBUF_COLOR_FORMAT_BGRx;
  case JZ_FMT_RGBA: return NVBUF_COLOR_FORMAT_RGBA;
  case JZ_FMT_RGBX: return NVBUF_COLOR_FORMAT_RGBx;
  default: return NVBUF_COLOR_FORMAT_NV12;
  }
}

// Imports a single-plane RGB dma-buf (e.g. a KMS scanout). `block_height_log2` < 0 means pitch
// linear, otherwise NVIDIA block linear with that GOB block height. `fd` stays the caller's: the
// surface imports a duplicate, which NvBufSurfaceDestroy closes.
NvBufSurface *jz_import(int fd, uint32_t w, uint32_t h, int fmt, uint32_t pitch, uint32_t offset,
                        int block_height_log2) {
  // NvBufSurface looks buffers up by fd number, and libnvv4l2 leaves stale entries behind for fds
  // it closed; a low recycled number can resolve to one of those. Keep imports out of that range.
  int own = fcntl(fd, F_DUPFD_CLOEXEC, 512);
  if (own < 0) return NULL;
  fd = own;
  off_t size = lseek(fd, 0, SEEK_END);
  if (size <= 0) {
    close(own);
    return NULL;
  }
  NvBufSurfaceMapParams mp;
  memset(&mp, 0, sizeof mp);
  mp.num_planes = 1;
  mp.fd = fd;
  mp.totalSize = size;
  mp.memType = NVBUF_MEM_SURFACE_ARRAY;
  mp.layout = block_height_log2 < 0 ? NVBUF_LAYOUT_PITCH : NVBUF_LAYOUT_BLOCK_LINEAR;
  mp.colorFormat = color_format(fmt);
  mp.planes[0].width = w;
  mp.planes[0].height = h;
  mp.planes[0].pitch = pitch;
  mp.planes[0].offset = offset;
  mp.planes[0].psize = size;
  mp.planes[0].blockheightlog2 = block_height_log2 < 0 ? 0 : block_height_log2;
  NvBufSurface *s = NULL;
  if (NvBufSurfaceImport(&s, &mp) != 0) {
    close(own);
    return NULL;
  }
  s->numFilled = 1;
  return s;
}

// Pitch-linear surface. RGB surfaces stay mapped for jz_upload.
NvBufSurface *jz_alloc(uint32_t w, uint32_t h, int fmt) {
  NvBufSurfaceCreateParams cp;
  memset(&cp, 0, sizeof cp);
  cp.width = w;
  cp.height = h;
  cp.layout = NVBUF_LAYOUT_PITCH;
  cp.memType = NVBUF_MEM_SURFACE_ARRAY;
  cp.colorFormat = color_format(fmt);
  NvBufSurface *s = NULL;
  if (NvBufSurfaceCreate(&s, 1, &cp) != 0) return NULL;
  s->numFilled = 1;
  if (fmt != JZ_FMT_NV12 && NvBufSurfaceMap(s, 0, 0, NVBUF_MAP_WRITE) != 0) {
    NvBufSurfaceDestroy(s);
    return NULL;
  }
  return s;
}

void jz_destroy(NvBufSurface *s) {
  if (!s) return;
  if (s->surfaceList[0].mappedAddr.addr[0]) NvBufSurfaceUnMap(s, 0, 0);
  NvBufSurfaceDestroy(s);
}

int jz_fd(NvBufSurface *s) { return (int)s->surfaceList[0].bufferDesc; }

// Copies packed 32-bit rows into a jz_alloc'd RGB surface.
int jz_upload(NvBufSurface *dst, const uint8_t *src, uint32_t stride, uint32_t w, uint32_t h) {
  NvBufSurfaceParams *p = &dst->surfaceList[0];
  uint8_t *base = p->mappedAddr.addr[0];
  if (!base || w > p->width || h > p->height) return -1;
  for (uint32_t y = 0; y < h; y++) memcpy(base + (size_t)y * p->pitch, src + (size_t)y * stride, (size_t)w * 4);
  return NvBufSurfaceSyncForDevice(dst, 0, 0);
}

// VIC color conversion/copy, src and dst of the same size. Session params are per thread.
int jz_convert(NvBufSurface *src, NvBufSurface *dst) {
  static __thread int session_set;
  if (!session_set) {
    NvBufSurfTransformConfigParams cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.compute_mode = NvBufSurfTransformCompute_VIC;
    if (NvBufSurfTransformSetSessionParams(&cfg) != NvBufSurfTransformError_Success) return -1;
    session_set = 1;
  }
  NvBufSurfTransformParams tp;
  memset(&tp, 0, sizeof tp);
  return NvBufSurfTransform(src, dst, &tp) == NvBufSurfTransformError_Success ? 0 : -1;
}

typedef struct jz_enc {
  int fd;
  int codec;
  int nout;
  void *cap[JZ_NCAP];
  size_t cap_len[JZ_NCAP];
  int cap_fd[JZ_NCAP];
} jz_enc;

static int ctrl(jz_enc *e, uint32_t id, int32_t value) {
  struct v4l2_ext_control c;
  struct v4l2_ext_controls cs;
  memset(&c, 0, sizeof c);
  memset(&cs, 0, sizeof cs);
  c.id = id;
  c.value = value;
  cs.ctrl_class = V4L2_CTRL_CLASS_MPEG;
  cs.count = 1;
  cs.controls = &c;
  return v4l2_ioctl(e->fd, VIDIOC_S_EXT_CTRLS, &cs);
}

void jz_enc_close(jz_enc *e) {
  if (!e) return;
  if (e->fd >= 0) {
    int t = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
    v4l2_ioctl(e->fd, VIDIOC_STREAMOFF, &t);
    t = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    v4l2_ioctl(e->fd, VIDIOC_STREAMOFF, &t);
  }
  for (int i = 0; i < JZ_NCAP; i++) {
    if (e->cap[i]) munmap(e->cap[i], e->cap_len[i]);
    if (e->cap_fd[i] >= 0) close(e->cap_fd[i]);
  }
  if (e->fd >= 0) v4l2_close(e->fd);
  free(e);
}

// Opens an encoder taking NV12 (BT.601 limited) input surfaces of w x h through `nout` DMABUF
// slots. `bitrate` in bps.
jz_enc *jz_enc_open(int codec, uint32_t w, uint32_t h, uint32_t bitrate, uint32_t gop, int nout,
                    char *err, size_t errlen) {
  jz_enc *e = calloc(1, sizeof *e);
  if (!e) return NULL;
  e->fd = -1;
  e->codec = codec;
  for (int i = 0; i < JZ_NCAP; i++) e->cap_fd[i] = -1;
  if (nout < 1 || nout > JZ_MAX_OUT) {
    errno = EINVAL;
    set_err(err, errlen, "nout");
    goto fail;
  }
  e->nout = nout;
  // Non-blocking: in blocking mode an output-plane DQBUF waits indefinitely, while the capture
  // plane answers EAGAIN anyway. dqbuf() below applies the timeouts.
  e->fd = v4l2_open("/dev/v4l2-nvenc", O_RDWR | O_NONBLOCK);
  if (e->fd < 0) {
    set_err(err, errlen, "v4l2_open");
    goto fail;
  }
  struct v4l2_format f;
  memset(&f, 0, sizeof f);
  f.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  f.fmt.pix_mp.pixelformat =
      codec == JZ_H264 ? V4L2_PIX_FMT_H264 : codec == JZ_H265 ? V4L2_PIX_FMT_H265 : V4L2_PIX_FMT_AV1;
  f.fmt.pix_mp.width = w;
  f.fmt.pix_mp.height = h;
  f.fmt.pix_mp.num_planes = 1;
  f.fmt.pix_mp.plane_fmt[0].sizeimage = JZ_CAP_SIZE;
  if (v4l2_ioctl(e->fd, VIDIOC_S_FMT, &f) < 0) {
    set_err(err, errlen, "S_FMT capture");
    goto fail;
  }
  memset(&f, 0, sizeof f);
  f.type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
  f.fmt.pix_mp.pixelformat = V4L2_PIX_FMT_NV12M;
  f.fmt.pix_mp.width = w;
  f.fmt.pix_mp.height = h;
  f.fmt.pix_mp.num_planes = 2;
  f.fmt.pix_mp.colorspace = V4L2_COLORSPACE_SMPTE170M;
  if (v4l2_ioctl(e->fd, VIDIOC_S_FMT, &f) < 0) {
    set_err(err, errlen, "S_FMT output");
    goto fail;
  }
  // Optional tuning: a rejected control only costs quality, not correctness.
  ctrl(e, V4L2_CID_MPEG_VIDEO_BITRATE_MODE, V4L2_MPEG_VIDEO_BITRATE_MODE_CBR);
  ctrl(e, V4L2_CID_MPEG_VIDEO_BITRATE, bitrate);
  ctrl(e, V4L2_CID_MPEG_VIDEO_GOP_SIZE, gop);
  ctrl(e, V4L2_CID_MPEG_VIDEO_IDR_INTERVAL, gop);
  ctrl(e, V4L2_CID_MPEG_VIDEOENC_HW_PRESET_TYPE_PARAM, V4L2_ENC_HW_PRESET_ULTRAFAST);
  ctrl(e, V4L2_CID_MPEG_VIDEO_MAX_PERFORMANCE, 1);
  ctrl(e, V4L2_CID_MPEG_VIDEOENC_NUM_BFRAMES, 0);
  if (codec == JZ_AV1) {
    ctrl(e, V4L2_CID_MPEG_VIDEOENC_AV1_HEADERS_WITH_FRAME, 0);
  } else {
    ctrl(e, V4L2_CID_MPEG_VIDEOENC_INSERT_SPS_PPS_AT_IDR, 1);
    ctrl(e, V4L2_CID_MPEG_VIDEOENC_INSERT_VUI, 1);
  }
  if (codec == JZ_H264) {
    ctrl(e, V4L2_CID_MPEG_VIDEO_H264_PROFILE, V4L2_MPEG_VIDEO_H264_PROFILE_HIGH);
    // No frame reordering: decode order == display order.
    ctrl(e, V4L2_CID_MPEG_VIDEOENC_POC_TYPE, 2);
  }
  struct v4l2_streamparm sp;
  memset(&sp, 0, sizeof sp);
  sp.type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
  sp.parm.output.timeperframe.numerator = 1;
  sp.parm.output.timeperframe.denominator = 30;
  v4l2_ioctl(e->fd, VIDIOC_S_PARM, &sp);

  struct v4l2_requestbuffers rb;
  memset(&rb, 0, sizeof rb);
  rb.count = nout;
  rb.type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
  rb.memory = V4L2_MEMORY_DMABUF;
  if (v4l2_ioctl(e->fd, VIDIOC_REQBUFS, &rb) < 0 || (int)rb.count < nout) {
    set_err(err, errlen, "REQBUFS output");
    goto fail;
  }
  memset(&rb, 0, sizeof rb);
  rb.count = JZ_NCAP;
  rb.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  rb.memory = V4L2_MEMORY_MMAP;
  if (v4l2_ioctl(e->fd, VIDIOC_REQBUFS, &rb) < 0 || rb.count < JZ_NCAP) {
    set_err(err, errlen, "REQBUFS capture");
    goto fail;
  }
  for (int i = 0; i < JZ_NCAP; i++) {
    struct v4l2_plane pl[1];
    struct v4l2_buffer b;
    memset(&b, 0, sizeof b);
    memset(pl, 0, sizeof pl);
    b.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    b.memory = V4L2_MEMORY_MMAP;
    b.index = i;
    b.m.planes = pl;
    b.length = 1;
    if (v4l2_ioctl(e->fd, VIDIOC_QUERYBUF, &b) < 0) {
      set_err(err, errlen, "QUERYBUF");
      goto fail;
    }
    struct v4l2_exportbuffer eb;
    memset(&eb, 0, sizeof eb);
    eb.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    eb.index = i;
    if (v4l2_ioctl(e->fd, VIDIOC_EXPBUF, &eb) < 0) {
      set_err(err, errlen, "EXPBUF");
      goto fail;
    }
    // libnvv4l2 maps capture buffers through the exported fd, not through v4l2_mmap. The fd
    // stays open until close: NvBufSurface looks buffers up by fd number (libnvbuf_fdmap), so a
    // surface allocated later on a recycled number would be taken for this buffer.
    e->cap_fd[i] = eb.fd;
    void *m = mmap(NULL, pl[0].length, PROT_READ | PROT_WRITE, MAP_SHARED, eb.fd, pl[0].m.mem_offset);
    if (m == MAP_FAILED) {
      set_err(err, errlen, "mmap capture");
      goto fail;
    }
    e->cap[i] = m;
    e->cap_len[i] = pl[0].length;
  }
  int t = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
  if (v4l2_ioctl(e->fd, VIDIOC_STREAMON, &t) < 0) {
    set_err(err, errlen, "STREAMON output");
    goto fail;
  }
  t = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  if (v4l2_ioctl(e->fd, VIDIOC_STREAMON, &t) < 0) {
    set_err(err, errlen, "STREAMON capture");
    goto fail;
  }
  for (int i = 0; i < JZ_NCAP; i++) {
    struct v4l2_plane pl[1];
    struct v4l2_buffer b;
    memset(&b, 0, sizeof b);
    memset(pl, 0, sizeof pl);
    b.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    b.memory = V4L2_MEMORY_MMAP;
    b.index = i;
    b.m.planes = pl;
    b.length = 1;
    if (v4l2_ioctl(e->fd, VIDIOC_QBUF, &b) < 0) {
      set_err(err, errlen, "QBUF capture");
      goto fail;
    }
  }
  return e;
fail:
  jz_enc_close(e);
  return NULL;
}

int jz_enc_set_bitrate(jz_enc *e, uint32_t bitrate) {
  return ctrl(e, V4L2_CID_MPEG_VIDEO_BITRATE, bitrate);
}

int jz_enc_force_idr(jz_enc *e) { return ctrl(e, V4L2_CID_MPEG_VIDEOENC_FORCE_IDR_FRAME, 1); }

// Queues `s` (an NV12 jz_alloc surface of the encoder's size) into output slot `index`.
int jz_enc_queue(jz_enc *e, int index, NvBufSurface *s, int64_t pts_us) {
  NvBufSurfaceParams *p = &s->surfaceList[0];
  struct v4l2_plane pl[2];
  struct v4l2_buffer b;
  memset(&b, 0, sizeof b);
  memset(pl, 0, sizeof pl);
  b.type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
  b.memory = V4L2_MEMORY_DMABUF;
  b.index = index;
  b.m.planes = pl;
  b.length = 2;
  b.flags = V4L2_BUF_FLAG_TIMESTAMP_COPY;
  b.timestamp.tv_sec = pts_us / 1000000;
  b.timestamp.tv_usec = pts_us % 1000000;
  // Plane offsets are known to libnvv4l2 from the NvBufSurface behind the fd.
  for (int i = 0; i < 2; i++) {
    pl[i].m.fd = (int)p->bufferDesc;
    pl[i].bytesused = p->planeParams.psize[i];
  }
  return v4l2_ioctl(e->fd, VIDIOC_QBUF, &b);
}

// Polls a non-blocking DQBUF in 0.5 ms steps.
static int dqbuf(jz_enc *e, struct v4l2_buffer *b, int timeout_ms) {
  for (int waited = 0;; waited++) {
    if (v4l2_ioctl(e->fd, VIDIOC_DQBUF, b) == 0) return 0;
    if (errno != EAGAIN) return -2;
    if (waited >= timeout_ms * 2) return -1;
    usleep(500);
  }
}

// Returns the output slot the encoder is done reading, -1 on timeout, -2 on error.
int jz_enc_reclaim(jz_enc *e, int timeout_ms) {
  struct v4l2_plane pl[2];
  struct v4l2_buffer b;
  memset(&b, 0, sizeof b);
  memset(pl, 0, sizeof pl);
  b.type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
  b.memory = V4L2_MEMORY_DMABUF;
  b.m.planes = pl;
  b.length = 2;
  int r = dqbuf(e, &b, timeout_ms);
  return r < 0 ? r : (int)b.index;
}

// Returns a capture index holding one access unit, -1 on timeout, -2 on error. Hand the index
// back with jz_enc_release once the data is copied.
int jz_enc_dequeue(jz_enc *e, int timeout_ms, const uint8_t **data, uint32_t *len, int *key,
                   int64_t *pts_us) {
  struct v4l2_plane pl[1];
  struct v4l2_buffer b;
  memset(&b, 0, sizeof b);
  memset(pl, 0, sizeof pl);
  b.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  b.memory = V4L2_MEMORY_MMAP;
  b.m.planes = pl;
  b.length = 1;
  int r = dqbuf(e, &b, timeout_ms);
  if (r < 0) return r;
  if (b.index >= JZ_NCAP || pl[0].bytesused > e->cap_len[b.index]) return -2;
  const uint8_t *p = e->cap[b.index];
  uint32_t n = pl[0].bytesused;
  if (e->codec == JZ_AV1) {
    // R36.4 wraps AV1 in IVF regardless of V4L2_CID_MPEG_VIDEOENC_AV1_HEADERS_WITH_FRAME: a 32-byte
    // file header on the first buffer and a 12-byte frame header (LE size + pts) on each. Hand
    // out the bare OBUs.
    if (n >= 32 && memcmp(p, "DKIF", 4) == 0) {
      p += 32;
      n -= 32;
    }
    if (n >= 12 && (uint32_t)(p[0] | p[1] << 8 | p[2] << 16 | (uint32_t)p[3] << 24) == n - 12) {
      p += 12;
      n -= 12;
    }
  }
  *data = p;
  *len = n;
  *key = (b.flags & V4L2_BUF_FLAG_KEYFRAME) != 0;
  *pts_us = (int64_t)b.timestamp.tv_sec * 1000000 + b.timestamp.tv_usec;
  return (int)b.index;
}

int jz_enc_release(jz_enc *e, int index) {
  struct v4l2_plane pl[1];
  struct v4l2_buffer b;
  memset(&b, 0, sizeof b);
  memset(pl, 0, sizeof pl);
  b.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  b.memory = V4L2_MEMORY_MMAP;
  b.index = index;
  b.m.planes = pl;
  b.length = 1;
  return v4l2_ioctl(e->fd, VIDIOC_QBUF, &b);
}
