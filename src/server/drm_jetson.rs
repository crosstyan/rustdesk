// Zero-copy DRM capture for the Jetson encoder: the scanout dma-buf is imported as an
// NvBufSurface and converted to NV12 on the VIC, skipping the EGL detile + glReadPixels readback
// and every CPU copy after it. drm_capturer's receive thread asks `convert` first and falls back to
// its GL path whenever this declines.

use crate::ipc::DmabufDesc;
use hbb_common::log;
use scrap::jetson::JetsonSurface;
use std::{
    collections::{HashMap, VecDeque},
    os::fd::{AsRawFd, BorrowedFd, OwnedFd, RawFd},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

// Scanout buffers a compositor cycles through; older fds are dropped first.
const MAX_FBS: usize = 8;
// NV12 frames: the slot's latest, the encoder's in-flight and repeat references, one converting.
const MAX_POOL: usize = 5;

static WANTED: AtomicBool = AtomicBool::new(false);
static SUSPENDED: AtomicBool = AtomicBool::new(false);

/// Set by video_service once it knows whether the encoder takes Jetson GPU frames.
pub(super) fn set_wanted(on: bool) {
    WANTED.store(on, Ordering::SeqCst);
}

pub(super) fn wanted() -> bool {
    WANTED.load(Ordering::SeqCst) && !SUSPENDED.load(Ordering::SeqCst)
}

/// A screenshot needs CPU pixels: deliver them until `resume`.
pub(super) fn suspend() {
    SUSPENDED.store(true, Ordering::SeqCst);
}

pub(super) fn resume() {
    SUSPENDED.store(false, Ordering::SeqCst);
}

#[derive(Default)]
pub(super) struct JetsonCapture {
    // The producer sends each framebuffer's fd once; keep a dup so either path can import later.
    fds: HashMap<u32, OwnedFd>,
    order: VecDeque<u32>,
    imports: HashMap<u32, JetsonSurface>,
    // Framebuffers the GL converter has been handed an fd for (its own import-once cache).
    gl_seen: HashMap<u32, ()>,
    pool: Vec<Arc<JetsonSurface>>,
    failed: bool,
}

impl JetsonCapture {
    /// Records the fd a frame carried; must see every DrmFrameDmabuf.
    pub(super) fn note(&mut self, desc: &DmabufDesc, received_fd: RawFd) {
        if !desc.has_fd || received_fd < 0 || desc.fb_id == 0 {
            return;
        }
        // SAFETY: `received_fd` is open for the duration of this call (drm_capturer closes it at
        // the end of the loop iteration).
        let dup = match unsafe { BorrowedFd::borrow_raw(received_fd) }.try_clone_to_owned() {
            Ok(fd) => fd,
            Err(err) => {
                log::warn!("drm/jetson: dup of the scanout fd failed: {err}");
                return;
            }
        };
        // A new fd for a known id is a new buffer: forget the old imports.
        self.imports.remove(&desc.fb_id);
        self.gl_seen.remove(&desc.fb_id);
        if self.fds.insert(desc.fb_id, dup).is_none() {
            self.order.push_back(desc.fb_id);
        }
        while self.order.len() > MAX_FBS {
            if let Some(old) = self.order.pop_front() {
                self.fds.remove(&old);
                self.imports.remove(&old);
                self.gl_seen.remove(&old);
            }
        }
    }

    /// The fd to hand the GL converter: its cache misses framebuffers whose fd arrived while
    /// frames were going through `convert` instead.
    pub(super) fn gl_fd(&mut self, desc: &DmabufDesc, received_fd: RawFd) -> RawFd {
        if received_fd >= 0 || self.gl_seen.contains_key(&desc.fb_id) {
            self.gl_seen.insert(desc.fb_id, ());
            return received_fd;
        }
        match self.fds.get(&desc.fb_id) {
            Some(fd) => {
                self.gl_seen.insert(desc.fb_id, ());
                fd.as_raw_fd()
            }
            None => received_fd,
        }
    }

    /// The frame as an NV12 GPU surface, or None to take the GL path.
    pub(super) fn convert(&mut self, desc: &DmabufDesc, transform: i32) -> Option<Arc<JetsonSurface>> {
        if self.failed || !wanted() {
            return None;
        }
        // Rotation and HDR tone-mapping stay on the GL path.
        if transform != 0 || desc.hdr_eotf != 0 || desc.num_planes > 1 {
            return None;
        }
        let (w, h) = (desc.width as usize, desc.height as usize);
        let result = (|| -> hbb_common::ResultType<Option<Arc<JetsonSurface>>> {
            if !self.imports.contains_key(&desc.fb_id) {
                let Some(fd) = self.fds.get(&desc.fb_id) else {
                    // No fd yet for this framebuffer (the GL converter got it); wait for one.
                    return Ok(None);
                };
                let surface = JetsonSurface::import(
                    fd.try_clone()?,
                    w,
                    h,
                    desc.format,
                    desc.modifier,
                    desc.pitches[0],
                    desc.offsets[0],
                )?;
                self.imports.insert(desc.fb_id, surface);
            }
            let Some(src) = self.imports.get(&desc.fb_id) else {
                return Ok(None);
            };
            self.pool.retain(|s| (s.width(), s.height()) == (w, h));
            let dst = match self.pool.iter().find(|s| Arc::strong_count(s) == 1) {
                Some(s) => s.clone(),
                None if self.pool.len() < MAX_POOL => {
                    let s = Arc::new(JetsonSurface::alloc_nv12(w, h)?);
                    self.pool.push(s.clone());
                    s
                }
                // Every surface is still referenced downstream: drop this frame.
                None => return Ok(None),
            };
            src.convert_into(&dst)?;
            Ok(Some(dst))
        })();
        match result {
            Ok(frame) => frame,
            Err(err) => {
                log::warn!("drm/jetson: zero-copy capture unavailable, using the GL path: {err:?}");
                self.failed = true;
                None
            }
        }
    }
}
