// Encode synthetic 4K BGRA frames with the Jetson encoder (and libaom for reference), decode each
// frame with the client-side `Decoder`, and report CPU cost and color fidelity.
//
//   cargo run --release -p scrap --features jetson,hwcodec,wayland --example jetson_bench -- [frames]

#[cfg(feature = "jetson")]
fn main() {
    use base::message_proto::video_frame::Union;
    use scrap::libc;
    use scrap::{
        aom::AomEncoderConfig,
        codec::{Decoder, Encoder, EncoderCfg},
        jetson::JetsonEncoderConfig,
        CodecFormat, Frame, ImageFormat, ImageRgb, ImageTexture, PixelBuffer, Pixfmt,
    };
    use std::time::{Duration, Instant};

    let n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(150);
    // BENCH_SIZE=WxH, e.g. an odd height to check the NV12 layout.
    // BENCH_PROBE=N: repeat the availability probe's create/encode-one/drop cycle N times.
    if let Some(n) = std::env::var("BENCH_PROBE").ok().and_then(|v| v.parse::<usize>().ok()) {
        for f in [CodecFormat::H264, CodecFormat::H265, CodecFormat::AV1] {
            for i in 0..n {
                let t = Instant::now();
                let mut enc = Encoder::new(
                    EncoderCfg::JETSON(JetsonEncoderConfig {
                        format: f,
                        width: 256,
                        height: 256,
                        quality: 1.0,
                        keyframe_interval: None,
                    }),
                    false,
                )
                .expect("create");
                let frame = vec![0x80u8; 256 * 256 * 4];
                let r = enc.encode_to_message(scrap::EncodeInput::YUV(&frame), 0);
                let t_enc = t.elapsed();
                drop(enc);
                println!("probe {f:?} #{i}: encode {:?} {:?}, total {:?}", r.as_ref().map(|_| ()), t_enc, t.elapsed());
            }
        }
        return;
    }
    let (w, h) = std::env::var("BENCH_SIZE")
        .ok()
        .and_then(|v| {
            let (w, h) = v.split_once('x')?;
            Some((w.parse().ok()?, h.parse().ok()?))
        })
        .unwrap_or((3840usize, 2160usize));
    let fps = 30u64;

    // A desktop-like frame: flat panels, a gradient, and a moving block.
    let make = |i: usize| -> Vec<u8> {
        let mut f = vec![0u8; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let p = (y * w + x) * 4;
                let (b, g, r) = if x < w / 4 {
                    (200, 60, 30) // blue-ish sidebar (BGRA)
                } else if y < 80 {
                    (40, 40, 40)
                } else {
                    ((x * 255 / w) as u8, (y * 255 / h) as u8, 128)
                };
                f[p] = b;
                f[p + 1] = g;
                f[p + 2] = r;
                f[p + 3] = 255;
            }
        }
        let bx = (i * 97) % (w - 400).max(1);
        for y in 800.min(h)..1200.min(h) {
            for x in bx..bx + 400 {
                let p = (y * w + x) * 4;
                f[p..p + 4].copy_from_slice(&[0, 0, 255, 255]); // red block
            }
        }
        f
    };
    let frames: Vec<Vec<u8>> = (0..8).map(make).collect();

    let cpu = || {
        let mut ru: libc::rusage = unsafe { std::mem::zeroed() };
        unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut ru) };
        let t = |v: libc::timeval| v.tv_sec as f64 + v.tv_usec as f64 / 1e6;
        t(ru.ru_utime) + t(ru.ru_stime)
    };

    let mut cases: Vec<(String, EncoderCfg, CodecFormat)> = Vec::new();
    for f in [CodecFormat::H264, CodecFormat::H265, CodecFormat::AV1] {
        cases.push((
            format!("jetson-{f:?}"),
            EncoderCfg::JETSON(JetsonEncoderConfig {
                format: f,
                width: w,
                height: h,
                quality: 0.67,
                keyframe_interval: None,
            }),
            f,
        ));
    }
    if std::env::var("BENCH_AOM").is_ok() {
        cases.push((
            "libaom-AV1".into(),
            EncoderCfg::AOM(AomEncoderConfig {
                width: w as _,
                height: h as _,
                quality: 0.67,
                keyframe_interval: None,
            }),
            CodecFormat::AV1,
        ));
    }

    for (name, cfg, format) in cases {
        let mut enc = match Encoder::new(cfg, false) {
            Ok(e) => e,
            Err(e) => {
                println!("{name}: create failed: {e:?}");
                continue;
            }
        };
        let (mut yuv, mut mid) = (Vec::new(), Vec::new());
        let mut encoded = Vec::new();
        let mut enc_time = Duration::ZERO;
        let repeats: usize = std::env::var("BENCH_REPEAT").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
        let (mut repeat_ok, mut repeat_err) = (0usize, 0usize);
        let start = Instant::now();
        let cpu0 = cpu();
        for i in 0..n {
            let due = start + Duration::from_millis(i as u64 * 1000 / fps);
            if let Some(d) = due.checked_duration_since(Instant::now()) {
                std::thread::sleep(d);
            }
            let src = &frames[i % frames.len()];
            let frame = Frame::PixelBuffer(PixelBuffer::new(src, Pixfmt::BGRA, w, h));
            let t = Instant::now();
            let input = match frame.to(enc.yuvfmt(), &mut yuv, &mut mid) {
                Ok(v) => v,
                Err(e) => {
                    println!("{name}: convert failed: {e:?}");
                    break;
                }
            };
            let vf = enc.encode_to_message(input, (i as u64 * 1000 / fps) as i64);
            enc_time += t.elapsed();
            if let Ok(vf) = vf {
                encoded.push((i, vf));
            }
            // BENCH_REPEAT=N: N idle repeats (null texture) after every frame, as video_service does.
            for _ in 0..repeats {
                match enc.encode_to_message(
                    scrap::EncodeInput::Texture((std::ptr::null_mut(), 0)),
                    (i as u64 * 1000 / fps) as i64,
                ) {
                    Ok(vf) => {
                        repeat_ok += 1;
                        // Decoded like any frame, not color-checked (usize::MAX).
                        encoded.push((usize::MAX, vf));
                    }
                    Err(e) => {
                        repeat_err += 1;
                        if repeat_err <= 3 {
                            println!("  repeat after frame {i}: {e:?}");
                        }
                    }
                }
            }
        }
        let wall = start.elapsed().as_secs_f64();
        if repeats > 0 {
            println!("{name}: idle repeats ok {repeat_ok}, failed {repeat_err}");
        }
        let cpu_s = cpu() - cpu0;
        drop(enc);

        // Decode everything with the client-side decoder, outside the measured window.
        let mut dec = Decoder::new(format, None);
        let (mut bytes, mut decoded, mut max_err) = (0usize, 0, 0i32);
        for (i, vf) in &encoded {
            let Some(union) = vf.union.as_ref() else {
                continue;
            };
            bytes += match union {
                Union::H264s(v) | Union::H265s(v) | Union::Av1s(v) => {
                    v.frames.iter().map(|f| f.data.len()).sum()
                }
                _ => 0,
            };
            let mut rgb = ImageRgb::new(ImageFormat::ARGB, 1);
            let (mut tex, mut pb, mut chroma) = (ImageTexture::default(), false, None);
            if let Ok(true) =
                dec.handle_video_frame(union, &mut rgb, &mut tex, &mut pb, &mut chroma)
            {
                decoded += 1;
                if i % 30 == 29 && rgb.w == w && rgb.h == h {
                    // Flat-region pixels: sidebar, title bar, moving block.
                    let src = &frames[i % frames.len()];
                    let stride = rgb.raw.len() / h;
                    let bx = ((i % frames.len()) * 97) % (w - 400).max(1);
                    for (x, y) in [(100, h / 2), (w / 2, 40), (bx + 200, 1000.min(h - 1))] {
                        let d = &rgb.raw[y * stride + x * 4..][..3];
                        let s = &src[(y * w + x) * 4..][..3];
                        for c in 0..3 {
                            max_err = max_err.max((d[c] as i32 - s[c] as i32).abs());
                        }
                        if *i == 29 {
                            println!("  ({x},{y}) src bgr {:?} -> decoded {:?}", s, d);
                        }
                    }
                }
            }
        }
        println!(
            "{name}: encoded {}/{n}, client-decoded {decoded}, encode call {:.1} ms/frame, \
             {:.2} Mbit/s, max color err {max_err}, process cpu {:.0}% of one core",
            encoded.len(),
            enc_time.as_secs_f64() * 1000.0 / n as f64,
            bytes as f64 * 8.0 / (n as f64 / fps as f64) / 1e6,
            cpu_s / wall * 100.0,
        );
    }
}

#[cfg(not(feature = "jetson"))]
fn main() {
    eprintln!("build with --features jetson");
}
