// Hardware encoder for NVIDIA Jetson (L4T / JetPack).
//
// The Jetson NVENC block is not reachable through FFmpeg NVENC (no `libnvidia-encode.so` on
// Tegra) nor VAAPI; it is exposed only through NVIDIA's libv4l2 plugin. jetson_nv.c drives it with
// NvBufSurface dma-bufs, so frames stay in NvMM memory:
//
//   GPU frame (capturer converted the KMS scanout on the VIC) -> NV12 surface -> NVENC
//   CPU frame (BGRA)   -> upload into an RGB surface -> VIC -> NV12 surface -> NVENC
//
// At 4K the VIC conversion and NVENC each take ~20 ms at idle clocks, so waiting for every frame's
// own bitstream would cap the rate near 15 fps. One frame stays in flight instead: each call
// returns the previous frame. The encoder is therefore not latency free, and video_service's
// repeat-encode on an idle screen pushes the last frame out.

use crate::{
    codec::{base_bitrate, EncoderApi, EncoderCfg},
    CodecFormat, EncodeInput, EncodeYuvFormat, Pixfmt,
};
use base::message_proto::{EncodedVideoFrame, EncodedVideoFrames, VideoFrame};
use hbb_common::{
    anyhow::{anyhow, bail},
    bytes::Bytes,
    log, ResultType,
};
use std::{
    ffi::{c_char, c_int, c_void, CStr},
    os::fd::{AsRawFd, BorrowedFd},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
};

const NVENC_DEVICE: &str = "/dev/v4l2-nvenc";
// The first frame allocates NvMM pools; later frames take a few ms.
const FIRST_FRAME_TIMEOUT_MS: c_int = 2000;
const FRAME_TIMEOUT_MS: c_int = 500;
const MAX_IN_FLIGHT: usize = 1;
// Output (input-frame) slots: in flight + the one being queued + the priming duplicate.
const SLOTS: usize = 4;
// Effectively "never", keyframes come from recreating the encoder (new subscriber, refresh).
const DEFAULT_GOP: u32 = 1 << 30;
// Tags an EncodeInput::Texture as an `Arc<JetsonSurface>` (see `texture`).
const TEXTURE_TAG: usize = 0x6a65_7473;

const JZ_FMT_NV12: c_int = 0;
const JZ_FMT_BGRA: c_int = 1;
const JZ_FMT_BGRX: c_int = 2;
const JZ_FMT_RGBA: c_int = 3;
const JZ_FMT_RGBX: c_int = 4;

#[repr(C)]
struct NvBufSurface {
    _private: [u8; 0],
}

#[repr(C)]
struct JzEnc {
    _private: [u8; 0],
}

extern "C" {
    fn jz_import(
        fd: c_int,
        w: u32,
        h: u32,
        fmt: c_int,
        pitch: u32,
        offset: u32,
        block_height_log2: c_int,
    ) -> *mut NvBufSurface;
    fn jz_alloc(w: u32, h: u32, fmt: c_int) -> *mut NvBufSurface;
    fn jz_destroy(s: *mut NvBufSurface);
    fn jz_upload(dst: *mut NvBufSurface, src: *const u8, stride: u32, w: u32, h: u32) -> c_int;
    fn jz_convert(src: *mut NvBufSurface, dst: *mut NvBufSurface) -> c_int;
    fn jz_enc_open(
        codec: c_int,
        w: u32,
        h: u32,
        bitrate: u32,
        gop: u32,
        nout: c_int,
        err: *mut c_char,
        errlen: usize,
    ) -> *mut JzEnc;
    fn jz_enc_close(e: *mut JzEnc);
    fn jz_enc_set_bitrate(e: *mut JzEnc, bitrate: u32) -> c_int;
    fn jz_enc_queue(e: *mut JzEnc, index: c_int, s: *mut NvBufSurface, pts_us: i64) -> c_int;
    fn jz_enc_reclaim(e: *mut JzEnc, timeout_ms: c_int) -> c_int;
    fn jz_enc_dequeue(
        e: *mut JzEnc,
        timeout_ms: c_int,
        data: *mut *const u8,
        len: *mut u32,
        key: *mut c_int,
        pts_us: *mut i64,
    ) -> c_int;
    fn jz_enc_release(e: *mut JzEnc, index: c_int) -> c_int;
}

lazy_static::lazy_static! {
    static ref AVAILABLE: Mutex<Vec<(CodecFormat, bool)>> = Default::default();
}
static DISABLED: AtomicBool = AtomicBool::new(false);

/// An NvBufSurface: either allocated here, or an imported dma-buf (the import owns a duplicate of
/// the fd, closed by NvBufSurfaceDestroy).
pub struct JetsonSurface {
    ptr: *mut NvBufSurface,
    width: usize,
    height: usize,
    nv12: bool,
}

// NvBufSurface handles are process-wide; the buffer itself is only touched by the hardware or
// through jz_upload by the owner.
unsafe impl Send for JetsonSurface {}
unsafe impl Sync for JetsonSurface {}

impl JetsonSurface {
    pub fn alloc_nv12(width: usize, height: usize) -> ResultType<Self> {
        Self::alloc(width, height, JZ_FMT_NV12)
    }

    fn alloc(width: usize, height: usize, fmt: c_int) -> ResultType<Self> {
        let ptr = unsafe { jz_alloc(width as _, height as _, fmt) };
        if ptr.is_null() {
            bail!("jetson: NvBufSurfaceCreate {width}x{height} failed");
        }
        Ok(Self {
            ptr,
            width,
            height,
            nv12: fmt == JZ_FMT_NV12,
        })
    }

    /// Imports a single-plane 32-bit RGB dma-buf (a KMS scanout). Handles linear and NVIDIA
    /// block-linear (`DRM_FORMAT_MOD_NVIDIA_BLOCK_LINEAR_2D`) layouts.
    pub fn import(
        fd: BorrowedFd,
        width: usize,
        height: usize,
        drm_format: u32,
        modifier: u64,
        pitch: u32,
        offset: u32,
    ) -> ResultType<Self> {
        const DRM_FORMAT_MOD_LINEAR: u64 = 0;
        const NVIDIA_VENDOR: u64 = 0x03;
        let fmt = match &drm_format.to_le_bytes() {
            b"AR24" => JZ_FMT_BGRA,
            b"XR24" => JZ_FMT_BGRX,
            b"AB24" => JZ_FMT_RGBA,
            b"XB24" => JZ_FMT_RGBX,
            _ => bail!("jetson: unsupported scanout format {drm_format:#x}"),
        };
        let block_height_log2 = if modifier == DRM_FORMAT_MOD_LINEAR {
            -1
        } else if modifier >> 56 == NVIDIA_VENDOR && modifier & 0x10 != 0 {
            // Compressed block-linear (bits 23..25) is not readable by the VIC.
            if (modifier >> 23) & 0x7 != 0 {
                bail!("jetson: compressed scanout modifier {modifier:#x}");
            }
            (modifier & 0xf) as c_int
        } else {
            bail!("jetson: unsupported scanout modifier {modifier:#x}");
        };
        let ptr = unsafe {
            jz_import(
                fd.as_raw_fd(),
                width as _,
                height as _,
                fmt,
                pitch,
                offset,
                block_height_log2,
            )
        };
        if ptr.is_null() {
            bail!("jetson: NvBufSurfaceImport failed ({width}x{height}, modifier {modifier:#x})");
        }
        Ok(Self {
            ptr,
            width,
            height,
            nv12: false,
        })
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    /// Converts (or copies) into `dst` on the VIC. Both surfaces must have the same size.
    pub fn convert_into(&self, dst: &JetsonSurface) -> ResultType<()> {
        if (self.width, self.height) != (dst.width, dst.height) {
            bail!("jetson: VIC size mismatch");
        }
        if unsafe { jz_convert(self.ptr, dst.ptr) } != 0 {
            bail!("jetson: NvBufSurfTransform failed");
        }
        Ok(())
    }

    /// The EncodeInput::Texture handed to JetsonEncoder. The caller keeps `surface` alive for the
    /// duration of the encode call; the encoder takes its own reference while the frame is in
    /// flight, so a pool can reuse a surface once its strong count drops back to one.
    pub fn texture(surface: &Arc<JetsonSurface>) -> (*mut c_void, usize) {
        (Arc::as_ptr(surface) as *mut c_void, TEXTURE_TAG)
    }
}

impl Drop for JetsonSurface {
    fn drop(&mut self) {
        unsafe { jz_destroy(self.ptr) };
    }
}

#[derive(Debug, Clone)]
pub struct JetsonEncoderConfig {
    pub format: CodecFormat,
    pub width: usize,
    pub height: usize,
    pub quality: f32,
    pub keyframe_interval: Option<usize>,
}

pub struct JetsonEncoder {
    enc: *mut JzEnc,
    config: JetsonEncoderConfig,
    bitrate: u32, // kbps
    // The surface each output slot holds until the encoder has read it.
    slots: [Option<Arc<JetsonSurface>>; SLOTS],
    // CPU input: the upload target and the NV12 surfaces converted from it.
    upload: Option<JetsonSurface>,
    cpu_pool: Vec<Arc<JetsonSurface>>,
    last: Option<Arc<JetsonSurface>>,
    in_flight: usize,
    first: bool,
}

// The V4L2 handle is only used by the thread that owns the encoder.
unsafe impl Send for JetsonEncoder {}

fn codec_id(format: CodecFormat) -> Option<c_int> {
    match format {
        CodecFormat::H264 => Some(0),
        CodecFormat::H265 => Some(1),
        CodecFormat::AV1 => Some(2),
        _ => None,
    }
}

impl EncoderApi for JetsonEncoder {
    fn new(cfg: EncoderCfg, _i444: bool) -> ResultType<Self>
    where
        Self: Sized,
    {
        match cfg {
            EncoderCfg::JETSON(config) => Self::create(config),
            _ => bail!("encoder type mismatch"),
        }
    }

    fn encode_to_message(&mut self, input: EncodeInput, ms: i64) -> ResultType<VideoFrame> {
        let surface = match input {
            EncodeInput::YUV(data) => self.upload_cpu_frame(data)?,
            // A null texture asks to repeat the last frame (idle screen, see video_service).
            EncodeInput::Texture((ptr, _)) if ptr.is_null() => {
                self.last.clone().ok_or(anyhow!("jetson: no frame to repeat"))?
            }
            EncodeInput::Texture((ptr, tag)) => {
                if tag != TEXTURE_TAG {
                    bail!("jetson: not a jetson texture");
                }
                // SAFETY: `JetsonSurface::texture` produced `ptr` from an Arc the caller keeps
                // alive for this call; take a reference of our own.
                let surface = unsafe {
                    Arc::increment_strong_count(ptr as *const JetsonSurface);
                    Arc::from_raw(ptr as *const JetsonSurface)
                };
                if !surface.nv12
                    || (surface.width, surface.height) != (self.config.width, self.config.height)
                {
                    bail!(
                        "jetson: texture {}x{} does not match the encoder's {}x{}",
                        surface.width,
                        surface.height,
                        self.config.width,
                        self.config.height
                    );
                }
                surface
            }
        };
        self.last = Some(surface.clone());

        // Priming with a duplicate of the first frame keeps one frame in flight from the start, so
        // every call has a finished frame to return.
        let copies = if self.first { 2 } else { 1 };
        for _ in 0..copies {
            let slot = self.free_slot()?;
            if unsafe { jz_enc_queue(self.enc, slot as _, surface.ptr, ms.max(0) * 1000) } != 0 {
                bail!("jetson: QBUF failed: {}", std::io::Error::last_os_error());
            }
            self.slots[slot] = Some(surface.clone());
            self.in_flight += 1;
        }
        drop(surface);

        let timeout = if self.first {
            FIRST_FRAME_TIMEOUT_MS
        } else {
            FRAME_TIMEOUT_MS
        };
        // Pull only down to MAX_IN_FLIGHT: draining further would leave the next call with
        // nothing finished to return.
        let mut frames = Vec::new();
        while self.in_flight > MAX_IN_FLIGHT {
            let (mut data, mut len, mut key, mut pts_us) = (std::ptr::null(), 0u32, 0, 0i64);
            let index =
                unsafe { jz_enc_dequeue(self.enc, timeout, &mut data, &mut len, &mut key, &mut pts_us) };
            if index == -1 {
                // An input produced no output; don't let the count drift.
                self.in_flight = MAX_IN_FLIGHT;
                break;
            }
            if index < 0 {
                bail!("jetson: DQBUF capture failed: {}", std::io::Error::last_os_error());
            }
            self.in_flight -= 1;
            // SAFETY: jz_enc_dequeue bounds `len` by the mapped capture buffer, which stays valid
            // until jz_enc_release.
            let bytes = unsafe { std::slice::from_raw_parts(data, len as usize) };
            if !bytes.is_empty() {
                frames.push(EncodedVideoFrame {
                    data: Bytes::copy_from_slice(bytes),
                    key: key != 0,
                    pts: pts_us / 1000,
                    ..Default::default()
                });
            }
            unsafe { jz_enc_release(self.enc, index) };
        }
        self.reclaim_slots(0);
        if frames.is_empty() {
            bail!("no valid frame");
        }
        self.first = false;
        let frames = EncodedVideoFrames {
            frames: frames.into(),
            ..Default::default()
        };
        let mut vf = VideoFrame::new();
        match self.config.format {
            CodecFormat::H264 => vf.set_h264s(frames),
            CodecFormat::H265 => vf.set_h265s(frames),
            CodecFormat::AV1 => vf.set_av1s(frames),
            f => bail!("unsupported format: {f:?}"),
        }
        Ok(vf)
    }

    fn yuvfmt(&self) -> EncodeYuvFormat {
        EncodeYuvFormat {
            pixfmt: Pixfmt::BGRA,
            w: self.config.width,
            h: self.config.height,
            stride: vec![self.config.width * 4],
            u: 0,
            v: 0,
        }
    }

    #[cfg(feature = "vram")]
    fn input_texture(&self) -> bool {
        false
    }

    fn set_quality(&mut self, ratio: f32) -> ResultType<()> {
        let bitrate = Self::calc_bitrate(&self.config, ratio);
        if bitrate > 0 {
            if unsafe { jz_enc_set_bitrate(self.enc, bitrate * 1000) } != 0 {
                bail!("jetson: set bitrate: {}", std::io::Error::last_os_error());
            }
            self.bitrate = bitrate;
        }
        self.config.quality = ratio;
        Ok(())
    }

    fn bitrate(&self) -> u32 {
        self.bitrate
    }

    fn support_changing_quality(&self) -> bool {
        true
    }

    fn latency_free(&self) -> bool {
        false
    }

    fn is_hardware(&self) -> bool {
        true
    }

    fn disable(&self) {
        Self::disable_all();
    }
}

impl JetsonEncoder {
    fn create(config: JetsonEncoderConfig) -> ResultType<Self> {
        let codec = codec_id(config.format)
            .ok_or(anyhow!("jetson: unsupported format {:?}", config.format))?;
        let bitrate = Self::calc_bitrate(&config, config.quality);
        let gop = config
            .keyframe_interval
            .map(|v| v as u32)
            .unwrap_or(DEFAULT_GOP);
        let mut err = [0 as c_char; 256];
        let enc = unsafe {
            jz_enc_open(
                codec,
                config.width as _,
                config.height as _,
                bitrate * 1000,
                gop,
                SLOTS as _,
                err.as_mut_ptr(),
                err.len(),
            )
        };
        if enc.is_null() {
            let err = unsafe { CStr::from_ptr(err.as_ptr()) }.to_string_lossy();
            bail!("jetson: open {:?} encoder: {err}", config.format);
        }
        log::info!(
            "jetson encoder: {:?} {}x{} {} kbps",
            config.format,
            config.width,
            config.height,
            bitrate
        );
        Ok(Self {
            enc,
            config,
            bitrate,
            slots: Default::default(),
            upload: None,
            cpu_pool: Vec::new(),
            last: None,
            in_flight: 0,
            first: true,
        })
    }

    fn upload_cpu_frame(&mut self, data: &[u8]) -> ResultType<Arc<JetsonSurface>> {
        let (w, h) = (self.config.width, self.config.height);
        if data.len() < w * h * 4 {
            bail!("frame too small: {} < {}", data.len(), w * h * 4);
        }
        if self.upload.is_none() {
            self.upload = Some(JetsonSurface::alloc(w, h, JZ_FMT_BGRA)?);
        }
        self.reclaim_slots(0);
        let upload = self.upload.as_ref().ok_or(anyhow!("jetson: no upload surface"))?;
        if unsafe { jz_upload(upload.ptr, data.as_ptr(), (w * 4) as _, w as _, h as _) } != 0 {
            bail!("jetson: upload failed");
        }
        let nv12 = match self.cpu_pool.iter().find(|s| Arc::strong_count(s) == 1) {
            Some(s) => s.clone(),
            None => {
                let s = Arc::new(JetsonSurface::alloc_nv12(w, h)?);
                self.cpu_pool.push(s.clone());
                s
            }
        };
        upload.convert_into(&nv12)?;
        Ok(nv12)
    }

    fn reclaim_slots(&mut self, timeout_ms: c_int) -> bool {
        let mut any = false;
        let mut timeout = timeout_ms;
        loop {
            let index = unsafe { jz_enc_reclaim(self.enc, timeout) };
            if index < 0 || index as usize >= SLOTS {
                return any;
            }
            self.slots[index as usize] = None;
            any = true;
            timeout = 0;
        }
    }

    fn free_slot(&mut self) -> ResultType<usize> {
        for wait in [0, FRAME_TIMEOUT_MS] {
            if let Some(i) = self.slots.iter().position(|s| s.is_none()) {
                return Ok(i);
            }
            self.reclaim_slots(wait);
        }
        bail!("jetson: encoder did not release an input frame");
    }

    // Same curve as the other hardware encoders (hwcodec.rs `calc_bitrate`), in kbps.
    fn calc_bitrate(config: &JetsonEncoderConfig, ratio: f32) -> u32 {
        let base = base_bitrate(config.width as _, config.height as _) as f32 * ratio;
        let threshold = 2000.0;
        let decay_rate = 0.001;
        let (low, high) = if config.format == CodecFormat::H264 {
            (2.0, 1.0)
        } else {
            (1.5, 0.5)
        };
        let factor = if base > threshold {
            1.0 + high / (1.0 + (base - threshold) * decay_rate)
        } else {
            low
        };
        (base * factor) as u32
    }

    pub fn disable_all() {
        log::error!("jetson encoder disabled");
        DISABLED.store(true, Ordering::SeqCst);
    }

    /// Probes every format on a background thread. Call once at server start; `available`
    /// reports false until the probe for that format has finished.
    pub fn start_probe() {
        let spawned = std::thread::Builder::new()
            .name("jetson-probe".to_owned())
            .spawn(|| {
                // Publish together, after every probe encoder has been dropped.
                let results: Vec<_> = [CodecFormat::H264, CodecFormat::H265, CodecFormat::AV1]
                    .iter()
                    .map(|&format| (format, Self::probe(format)))
                    .collect();
                log::info!("jetson encoder available: {results:?}");
                *AVAILABLE.lock().unwrap() = results;
            });
        if let Err(e) = spawned {
            log::error!("jetson: failed to spawn probe thread: {e}");
        }
    }

    pub fn available(format: CodecFormat) -> bool {
        if DISABLED.load(Ordering::SeqCst)
            || std::env::var("RUSTDESK_JETSON_DISABLE").map_or(false, |v| v == "1")
        {
            return false;
        }
        // The L4T R36.4 AV1 encoder intermittently emits streams that stop decoding after a few
        // frames at some widths (1366, 1376, 3440 seen), so AV1 is opt-in.
        if format == CodecFormat::AV1
            && std::env::var("RUSTDESK_JETSON_AV1").map_or(true, |v| v != "1")
        {
            return false;
        }
        AVAILABLE
            .lock()
            .unwrap()
            .iter()
            .any(|(f, v)| *f == format && *v)
    }

    fn probe(format: CodecFormat) -> bool {
        if !std::path::Path::new(NVENC_DEVICE).exists() {
            return false;
        }
        let config = JetsonEncoderConfig {
            format,
            width: 256,
            height: 256,
            quality: 1.0,
            keyframe_interval: None,
        };
        let result = (|| -> ResultType<()> {
            let mut enc = Self::create(config)?;
            let frame = vec![0x80u8; 256 * 256 * 4];
            enc.encode_to_message(EncodeInput::YUV(&frame), 0)?;
            Ok(())
        })();
        if let Err(e) = &result {
            log::warn!("jetson encoder {format:?} probe failed: {e:?}");
        }
        result.is_ok()
    }
}

impl Drop for JetsonEncoder {
    fn drop(&mut self) {
        // STREAMOFF returns every queued buffer; only then may the surfaces go.
        unsafe { jz_enc_close(self.enc) };
        self.slots = Default::default();
    }
}
