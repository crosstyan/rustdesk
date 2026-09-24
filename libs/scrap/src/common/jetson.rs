// Hardware encoder for NVIDIA Jetson (L4T / JetPack).
//
// The Jetson NVENC block is not reachable through FFmpeg NVENC (no `libnvidia-encode.so` on
// Tegra) nor VAAPI; it is exposed only through the V4L2/NvMM stack, which GStreamer wraps as
// `nvv4l2h264enc` / `nvv4l2h265enc` / `nvv4l2av1enc`. The encoder runs
//
//   appsrc (system memory) ! nvvidconv ! video/x-raw(memory:NVMM),format=NV12
//     ! nvv4l2XXXenc ! appsink
//
// `nvvidconv` does the copy into NvMM and, when the input is BGRx/RGBA, the color conversion on
// the VIC engine, so the CPU never runs the RGB->YUV conversion.
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
use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_app::{AppSink, AppSrc};
use hbb_common::{
    anyhow::{anyhow, bail, Context},
    bytes::Bytes,
    log, ResultType,
};
use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Mutex,
    },
    time::Duration,
};

const NVENC_DEVICE: &str = "/dev/v4l2-nvenc";
// The first frame opens the V4L2 device and allocates NvMM pools; later frames take a few ms.
const FIRST_FRAME_TIMEOUT: Duration = Duration::from_millis(2000);
const FRAME_TIMEOUT: Duration = Duration::from_millis(500);
const MAX_IN_FLIGHT: usize = 1;
// Effectively "never", keyframes come from recreating the encoder (new subscriber, refresh).
const DEFAULT_GOP: u32 = 1 << 30;

lazy_static::lazy_static! {
    static ref AVAILABLE: Mutex<Vec<(CodecFormat, bool)>> = Default::default();
}
static DISABLED: AtomicBool = AtomicBool::new(false);

#[derive(Debug, Clone)]
pub struct JetsonEncoderConfig {
    pub format: CodecFormat,
    pub width: usize,
    pub height: usize,
    pub quality: f32,
    pub keyframe_interval: Option<usize>,
}

pub struct JetsonEncoder {
    pipeline: gst::Pipeline,
    appsrc: AppSrc,
    appsink: AppSink,
    enc: gst::Element,
    config: JetsonEncoderConfig,
    input: Pixfmt,
    frame_size: usize,
    bitrate: u32, // kbps
    in_flight: usize,
    first: bool,
}

fn element_name(format: CodecFormat) -> Option<&'static str> {
    match format {
        CodecFormat::H264 => Some("nvv4l2h264enc"),
        CodecFormat::H265 => Some("nvv4l2h265enc"),
        CodecFormat::AV1 => Some("nvv4l2av1enc"),
        _ => None,
    }
}

fn nv12_stride(w: usize) -> usize {
    (w + 3) & !3
}

fn round_up_2(v: usize) -> usize {
    (v + 1) & !1
}

// `bgra` (default) hands the captured frame to the VIC unchanged; `nv12` converts on the CPU
// first like the other encoders do. Kept switchable for measuring.
fn input_pixfmt() -> Pixfmt {
    match std::env::var("RUSTDESK_JETSON_INPUT").as_deref() {
        Ok("nv12") => Pixfmt::NV12,
        _ => Pixfmt::BGRA,
    }
}

// The caller's frame, handed to GStreamer without a copy. Dropping it (when the pipeline releases
// the buffer) disconnects `_released`.
struct BorrowedFrame {
    ptr: *const u8,
    len: usize,
    _released: std::sync::mpsc::SyncSender<()>,
}

unsafe impl Send for BorrowedFrame {}

impl AsRef<[u8]> for BorrowedFrame {
    fn as_ref(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl EncoderApi for JetsonEncoder {
    fn new(cfg: EncoderCfg, _i444: bool) -> ResultType<Self>
    where
        Self: Sized,
    {
        match cfg {
            EncoderCfg::JETSON(config) => Self::create(config, input_pixfmt()),
            _ => bail!("encoder type mismatch"),
        }
    }

    fn encode_to_message(&mut self, input: EncodeInput, ms: i64) -> ResultType<VideoFrame> {
        let data = input.yuv()?;
        if data.len() < self.frame_size {
            bail!("frame too small: {} < {}", data.len(), self.frame_size);
        }
        self.check_bus()?;
        let (released_tx, released_rx) = std::sync::mpsc::sync_channel::<()>(0);
        let mut buffer = gst::Buffer::from_slice(BorrowedFrame {
            ptr: data.as_ptr(),
            len: self.frame_size,
            _released: released_tx,
        });
        if let Some(buffer) = buffer.get_mut() {
            buffer.set_pts(gst::ClockTime::from_mseconds(ms.max(0) as u64));
        }
        // Priming with a duplicate of the first frame keeps one frame in flight from the start, so
        // every call has a finished frame to return.
        let copies = if self.first { 2 } else { 1 };
        let mut pushed = Ok(gst::FlowSuccess::Ok);
        for _ in 0..copies {
            pushed = pushed.and(self.appsrc.push_buffer(buffer.clone()));
        }
        drop(buffer);
        // `data` is only borrowed for this call, so wait until nvvidconv has copied it into NvMM
        // and dropped the buffer.
        if released_rx.recv_timeout(FIRST_FRAME_TIMEOUT)
            != Err(std::sync::mpsc::RecvTimeoutError::Disconnected)
        {
            self.pipeline.set_state(gst::State::Null).ok();
            released_rx.recv().ok();
            bail!("jetson: input frame not released");
        }
        pushed.map_err(|e| anyhow!("push_buffer: {e:?}"))?;
        self.in_flight += copies;

        let timeout = if self.first {
            FIRST_FRAME_TIMEOUT
        } else {
            FRAME_TIMEOUT
        };
        let mut frames = Vec::new();
        loop {
            let waiting = self.in_flight > MAX_IN_FLIGHT;
            let wait = if waiting { timeout } else { Duration::ZERO };
            let Some(sample) = self
                .appsink
                .try_pull_sample(gst::ClockTime::from_nseconds(wait.as_nanos() as _))
            else {
                if waiting {
                    // An input produced no output; don't let the count drift.
                    self.in_flight = MAX_IN_FLIGHT;
                }
                break;
            };
            self.in_flight = self.in_flight.saturating_sub(1);
            if let Some(buf) = sample.get_buffer() {
                let map = buf.map_readable()?;
                frames.push(EncodedVideoFrame {
                    data: Bytes::copy_from_slice(map.as_slice()),
                    key: !buf.get_flags().contains(gst::BufferFlags::DELTA_UNIT),
                    pts: buf.get_pts().mseconds().map_or(ms, |v| v as i64),
                    ..Default::default()
                });
            }
        }
        if frames.is_empty() {
            self.check_bus()?;
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
        let (w, h) = (self.config.width, self.config.height);
        match self.input {
            // GstVideoInfo's default NV12 layout: 4-aligned stride, UV plane right after Y.
            Pixfmt::NV12 => EncodeYuvFormat {
                pixfmt: Pixfmt::NV12,
                w,
                h,
                stride: vec![nv12_stride(w), nv12_stride(w)],
                u: nv12_stride(w) * round_up_2(h),
                v: 0,
            },
            _ => EncodeYuvFormat {
                pixfmt: Pixfmt::BGRA,
                w,
                h,
                stride: vec![w * 4],
                u: 0,
                v: 0,
            },
        }
    }

    #[cfg(feature = "vram")]
    fn input_texture(&self) -> bool {
        false
    }

    fn set_quality(&mut self, ratio: f32) -> ResultType<()> {
        let bitrate = Self::calc_bitrate(&self.config, ratio);
        if bitrate > 0 {
            // nvv4l2 encoders apply a new bitrate while PLAYING.
            self.enc
                .set_property("bitrate", &(bitrate * 1000))
                .map_err(|e| anyhow!("set bitrate: {e}"))?;
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
    fn create(config: JetsonEncoderConfig, input: Pixfmt) -> ResultType<Self> {
        gst::init()?;
        let element = element_name(config.format)
            .ok_or(anyhow!("jetson: unsupported format {:?}", config.format))?;
        let (w, h) = (config.width, config.height);
        let (caps_format, frame_size) = match input {
            // libyuv's ARGBToNV12 is BT.601; without this GStreamer assumes BT.709 for HD.
            Pixfmt::NV12 => (
                "NV12,colorimetry=bt601",
                nv12_stride(w) * (round_up_2(h) + round_up_2(h) / 2),
            ),
            _ => ("BGRx", w * h * 4),
        };
        let bitrate = Self::calc_bitrate(&config, config.quality);
        let gop = config
            .keyframe_interval
            .map(|v| v as u32)
            .unwrap_or(DEFAULT_GOP);
        let codec_props = match config.format {
            // poc-type=2: no frame reordering, decode order == display order.
            CodecFormat::H264 => "profile=4 insert-sps-pps=true insert-vui=true poc-type=2",
            CodecFormat::H265 => "profile=0 insert-sps-pps=true insert-vui=true",
            CodecFormat::AV1 => "insert-seq-hdr=true",
            _ => "",
        };
        let out_caps = match config.format {
            CodecFormat::H264 => " ! video/x-h264,stream-format=byte-stream,alignment=au",
            CodecFormat::H265 => " ! video/x-h265,stream-format=byte-stream,alignment=au",
            _ => "",
        };
        let desc = format!(
            "appsrc name=src is-live=true do-timestamp=false format=time block=false \
               caps=video/x-raw,format={caps_format},width={w},height={h},framerate=30/1 \
             ! nvvidconv ! video/x-raw(memory:NVMM),format=NV12,colorimetry=bt601 \
             ! {element} name=enc control-rate=1 bitrate={br} iframeinterval={gop} \
               idrinterval={gop} preset-level=1 maxperf-enable=true {codec_props} \
             {out_caps} \
             ! appsink name=sink sync=false emit-signals=false max-buffers=8 drop=false",
            br = bitrate as u64 * 1000,
        );
        log::info!("jetson encoder pipeline: {desc}");
        let pipeline = gst::parse_launch(&desc)
            .context("jetson: parse_launch")?
            .downcast::<gst::Pipeline>()
            .map_err(|_| anyhow!("jetson: not a pipeline"))?;
        let get = |name: &str| {
            pipeline
                .get_by_name(name)
                .ok_or(anyhow!("jetson: no element {name}"))
        };
        let appsrc = get("src")?
            .dynamic_cast::<AppSrc>()
            .map_err(|_| anyhow!("jetson: src is not appsrc"))?;
        let appsink = get("sink")?
            .dynamic_cast::<AppSink>()
            .map_err(|_| anyhow!("jetson: sink is not appsink"))?;
        let enc = get("enc")?;
        let encoder = Self {
            pipeline,
            appsrc,
            appsink,
            enc,
            config,
            input,
            frame_size,
            bitrate,
            in_flight: 0,
            first: true,
        };
        encoder
            .pipeline
            .set_state(gst::State::Playing)
            .map_err(|e| anyhow!("jetson: set Playing: {e:?}"))?;
        encoder.check_bus()?;
        Ok(encoder)
    }

    fn check_bus(&self) -> ResultType<()> {
        if let Some(bus) = self.pipeline.get_bus() {
            if let Some(msg) = bus.pop_filtered(&[gst::MessageType::Error]) {
                if let gst::MessageView::Error(err) = msg.view() {
                    bail!(
                        "jetson pipeline error: {} ({:?})",
                        err.get_error(),
                        err.get_debug()
                    );
                }
            }
        }
        Ok(())
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
                for format in [CodecFormat::H264, CodecFormat::H265, CodecFormat::AV1] {
                    let v = Self::probe(format);
                    log::info!("jetson encoder {format:?} available: {v}");
                    AVAILABLE.lock().unwrap().push((format, v));
                }
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
        // nvv4l2av1enc (L4T R36.4) intermittently emits streams that stop decoding after a few
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
        if gst::init().is_err() {
            return false;
        }
        let Some(element) = element_name(format) else {
            return false;
        };
        if gst::ElementFactory::find(element).is_none()
            || gst::ElementFactory::find("nvvidconv").is_none()
        {
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
            let mut enc = Self::create(config, Pixfmt::BGRA)?;
            let frame = vec![0x80u8; enc.frame_size];
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
        self.appsrc.end_of_stream().ok();
        self.pipeline.set_state(gst::State::Null).ok();
    }
}
