//! The P0 GPU text renderer.
//!
//! Structure and call sequence mirror glyphon 0.12's `hello-world` example
//! (the authoritative reference for the glyphon 0.12 / wgpu 30 / winit 0.30
//! surface API). Shaping types come from cosmic-text (`Buffer`, `FontSystem`,
//! `SwashCache`); the GPU bridge comes from glyphon (`TextAtlas`,
//! `TextRenderer`, `Viewport`, `TextArea`). Both resolve to the single
//! cosmic-text 0.19 build glyphon pins, so the types are identical.
//!
//! P0 draws three surfaces: a one-line **stats banner** (file info + live
//! keystroke->present latency percentiles) at the top, the **document window**
//! (only the scroll-visible lines) below it, and a single **block/beam cursor**
//! overlay. The document is re-shaped only on edits/scrolls/resizes; idle
//! frames reuse the existing shaping (docs/PLAN.md: allocation-light render).

use std::sync::Arc;
use std::time::{Duration, Instant};

use cosmic_text::{Attrs, Buffer, Color, Family, FontSystem, Metrics, Shaping, SwashCache};
use glyphon::{Cache, Resolution, TextArea, TextAtlas, TextBounds, TextRenderer, Viewport};
use winit::event_loop::ActiveEventLoop;
use winit::window::Window;

use wgpu::{
    CommandEncoderDescriptor, CompositeAlphaMode, DeviceDescriptor, Instance, InstanceDescriptor,
    LoadOp, MultisampleState, Operations, PresentMode, RenderPassColorAttachment,
    RenderPassDescriptor, RequestAdapterOptions, SurfaceColorSpace, SurfaceConfiguration,
    TextureFormat, TextureUsages, TextureViewDescriptor,
};

/// Edge padding from the window origin to the text origin (D5: small padding,
/// minimal chrome). Logical px — scaled by `scale_factor` for HiDPI.
const PAD: f32 = 8.0;

/// Base body-text metrics at scale 1.0 (14px monospace on a 20px line). P1
/// pulls these from the TOML config (D13); HiDPI multiplies by `scale_factor`.
const BASE_FONT: f32 = 14.0;
const BASE_LINE: f32 = 20.0;

/// The stats banner occupies this many line-heights above the document (one
/// line of text plus a little breathing room).
const STATS_GAP: f32 = 1.6;

/// Monospace advance as a fraction of the font size. The P0 cursor is placed
/// arithmetically (column * advance) rather than by reading back cosmic-text's
/// per-glyph layout; good enough for the spike, refined in P1 when the UI
/// layer tracks glyph runs for damage. See `cursor` handling in [`Renderer`].
const MONO_ADVANCE_RATIO: f32 = 0.6;

/// The glyph drawn for the cursor: a left one-eighth block reads as a beam and
/// (unlike a full block) does not hide the character under it.
const CURSOR_GLYPH: &str = "\u{258f}";

/// Capacity of the keystroke->present latency ring (samples retained for the
/// p50/p99 window). Older samples roll off; the lifetime count is kept whole.
const LAT_RING_CAP: usize = 4096;

/// A fixed-capacity ring of keystroke latencies, in microseconds. The p99 of
/// this window is the D4 GO/NO-GO metric (docs/PLAN.md P0: p99 <= 8 ms).
///
/// Honesty note: the bracket is event receipt -> `queue.present` return, i.e.
/// event->GPU-submit. On Vulkan/Metal/DX12 `present` is non-blocking, so real
/// pixel-on-screen latency adds up to a vsync period on top. The verdict is
/// scoped accordingly in [`LatencyRing::summary`].
struct LatencyRing {
    buf: Vec<u32>,
    idx: usize,
    count: u64,
    /// Sorted snapshot of `buf`, rebuilt lazily; avoids a per-frame
    /// clone+sort when the banner re-formats on idle frames.
    sorted: Vec<u32>,
    dirty: bool,
}

impl LatencyRing {
    fn new() -> Self {
        Self {
            buf: Vec::with_capacity(LAT_RING_CAP),
            idx: 0,
            count: 0,
            sorted: Vec::with_capacity(LAT_RING_CAP),
            dirty: false,
        }
    }

    fn record(&mut self, d: Duration) {
        let us = d.as_micros().min(u32::MAX as u128) as u32;
        if self.buf.len() < LAT_RING_CAP {
            self.buf.push(us);
        } else {
            self.buf[self.idx] = us;
            self.idx = (self.idx + 1) % LAT_RING_CAP;
        }
        self.count = self.count.saturating_add(1);
        self.dirty = true;
    }

    /// Lifetime keystroke count (not just what fits in the window).
    fn count(&self) -> u64 {
        self.count
    }

    /// `(p50, p99, max)` over the current window, in milliseconds. `None` until
    /// the first sample lands. Allocates and sorts — exit-summary use only; the
    /// per-frame banner goes through [`LatencyRing::percentiles_cached`].
    fn percentiles(&self) -> Option<(f32, f32, f32)> {
        if self.buf.is_empty() {
            return None;
        }
        let mut v = self.buf.clone();
        v.sort_unstable();
        Self::pick_percentiles(&v)
    }

    /// Like [`LatencyRing::percentiles`] but reuses a sorted snapshot that is
    /// invalidated only by `record`, so idle frames pay no allocation or sort.
    fn percentiles_cached(&mut self) -> Option<(f32, f32, f32)> {
        if self.buf.is_empty() {
            return None;
        }
        if self.dirty {
            self.sorted.clear();
            self.sorted.extend_from_slice(&self.buf);
            self.sorted.sort_unstable();
            self.dirty = false;
        }
        Self::pick_percentiles(&self.sorted)
    }

    fn pick_percentiles(v: &[u32]) -> Option<(f32, f32, f32)> {
        let pick = |q: f32| -> f32 {
            let i = (((v.len() - 1) as f32) * q).round() as usize;
            v[i] as f32 / 1000.0
        };
        Some((pick(0.5), pick(0.99), *v.last()? as f32 / 1000.0))
    }

    /// One-line stdout summary with the D4 verdict (printed on exit).
    fn summary(&self) -> String {
        match self.percentiles() {
            Some((p50, p99, mx)) => format!(
                "latency: p50 {:.2}ms  p99 {:.2}ms  max {:.2}ms  n={}  \u{2014} D4 approx (event\u{2192}GPU-submit, not event\u{2192}pixel) {}",
                p50,
                p99,
                mx,
                self.count,
                if p99 <= 8.0 {
                    "GO (p99 <= 8ms)"
                } else {
                    "NO-GO (p99 > 8ms)"
                }
            ),
            None => "latency: no keystroke samples recorded".to_string(),
        }
    }
}

/// Owns the wgpu surface and the glyphon text pipeline, and draws the current
/// text into the window.
pub struct Renderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    surface_config: SurfaceConfiguration,
    instance: wgpu::Instance,

    font_system: FontSystem,
    swash_cache: SwashCache,
    viewport: Viewport,
    atlas: TextAtlas,
    text_renderer: TextRenderer,

    // Three shaped surfaces: banner, document window, cursor glyph.
    stats_buffer: Buffer,
    doc_buffer: Buffer,
    cursor_buffer: Buffer,

    /// Physical-pixel HiDPI scale (window.scale_factor()); folded into metrics,
    /// padding, bounds, and the cursor position.
    scale_factor: f64,

    /// Cursor position as `(line_in_window, column_in_chars)`, or `None` when
    /// the cursor is scrolled out of the visible window.
    cursor: Option<(usize, usize)>,

    /// The last document window text, kept so a scale change can re-shape it
    /// without the caller re-supplying it.
    doc_text: String,
    /// The file-info half of the banner (the latency half is appended live).
    stats_prefix: String,
    /// The last fully-rendered banner string, to skip re-shaping unchanged
    /// banners on idle frames.
    last_stats: String,

    /// Keystroke receipt timestamps awaiting the present that includes them.
    pending: Vec<Instant>,
    latency: LatencyRing,

    // Keep the window last so it drops after the surface — the surface borrows
    // the window handle, and dropping the window first can crash on some
    // platforms (noted in the glyphon example).
    window: Arc<Window>,
}

impl Renderer {
    /// Build the renderer for `window`. Blocks on adapter/device acquisition
    /// via pollster so callers stay synchronous inside the winit event loop.
    pub fn new(window: Arc<Window>, event_loop: &ActiveEventLoop) -> Self {
        pollster::block_on(Self::new_async(window, event_loop))
    }

    async fn new_async(window: Arc<Window>, event_loop: &ActiveEventLoop) -> Self {
        let physical_size = window.inner_size();
        let scale_factor = window.scale_factor();

        let instance = Instance::new(InstanceDescriptor::new_with_display_handle(Box::new(
            event_loop.owned_display_handle(),
        )));
        let adapter = instance
            .request_adapter(&RequestAdapterOptions::default())
            .await
            .expect("request a wgpu adapter");
        let (device, queue) = adapter
            .request_device(&DeviceDescriptor::default())
            .await
            .expect("request a wgpu device");

        let surface = instance
            .create_surface(window.clone())
            .expect("create a wgpu surface");
        let swapchain_format = TextureFormat::Bgra8UnormSrgb;
        let surface_config = SurfaceConfiguration {
            usage: TextureUsages::RENDER_ATTACHMENT,
            format: swapchain_format,
            width: physical_size.width.max(1),
            height: physical_size.height.max(1),
            present_mode: PresentMode::Fifo,
            alpha_mode: CompositeAlphaMode::Opaque,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
            color_space: SurfaceColorSpace::Auto,
        };
        surface.configure(&device, &surface_config);

        let mut font_system = FontSystem::new();
        let swash_cache = SwashCache::new();
        let cache = Cache::new(&device);
        let viewport = Viewport::new(&device, &cache);
        let mut atlas = TextAtlas::new(&device, &queue, &cache, swapchain_format);
        let text_renderer =
            TextRenderer::new(&mut atlas, &device, MultisampleState::default(), None);

        let metrics = Metrics::new(
            BASE_FONT * scale_factor as f32,
            BASE_LINE * scale_factor as f32,
        );
        let stats_buffer = Buffer::new(&mut font_system, metrics);
        let doc_buffer = Buffer::new(&mut font_system, metrics);

        // The cursor is a single glyph, shaped once here and re-shaped only on a
        // scale change.
        let mut cursor_buffer = Buffer::new(&mut font_system, metrics);
        cursor_buffer.set_text(
            CURSOR_GLYPH,
            &Attrs::new().family(Family::Monospace),
            Shaping::Advanced,
            None,
        );
        let cell_w = BASE_FONT * scale_factor as f32 * MONO_ADVANCE_RATIO;
        cursor_buffer.set_size(Some(cell_w * 2.0), Some(BASE_LINE * scale_factor as f32));
        cursor_buffer.shape_until_scroll(&mut font_system, false);

        Self {
            device,
            queue,
            surface,
            surface_config,
            instance,
            font_system,
            swash_cache,
            viewport,
            atlas,
            text_renderer,
            stats_buffer,
            doc_buffer,
            cursor_buffer,
            scale_factor,
            cursor: None,
            doc_text: String::new(),
            stats_prefix: String::new(),
            last_stats: String::new(),
            pending: Vec::new(),
            latency: LatencyRing::new(),
            window,
        }
    }

    /// The window this renderer draws into (for `request_redraw`).
    pub fn window(&self) -> &Arc<Window> {
        &self.window
    }

    /// Current surface size in physical pixels.
    pub fn size(&self) -> (u32, u32) {
        (self.surface_config.width, self.surface_config.height)
    }

    // --- HiDPI-scaled geometry (all physical pixels) -----------------------

    fn font_px(&self) -> f32 {
        BASE_FONT * self.scale_factor as f32
    }

    fn line_px(&self) -> f32 {
        BASE_LINE * self.scale_factor as f32
    }

    fn pad_px(&self) -> f32 {
        PAD * self.scale_factor as f32
    }

    fn cell_w(&self) -> f32 {
        self.font_px() * MONO_ADVANCE_RATIO
    }

    /// Y of the document top: below the stats banner.
    fn doc_top(&self) -> f32 {
        self.pad_px() + self.line_px() * STATS_GAP
    }

    fn metrics(&self) -> Metrics {
        Metrics::new(self.font_px(), self.line_px())
    }

    /// Document shaping box in physical pixels (width and visible height).
    fn doc_size(&self) -> (f32, f32) {
        let w = (self.surface_config.width as f32 - self.pad_px() * 2.0).max(1.0);
        let h = (self.surface_config.height as f32 - self.doc_top() - self.pad_px()).max(1.0);
        (w, h)
    }

    /// How many whole document lines fit in the current window. The caller uses
    /// this to size the scroll window (docs/PLAN.md: shape only visible lines).
    pub fn visible_line_capacity(&self) -> usize {
        let avail = self.surface_config.height as f32 - self.doc_top() - self.pad_px();
        if avail <= 0.0 {
            0
        } else {
            (avail / self.line_px()).floor() as usize
        }
    }

    /// Replace the shaped document window (the scroll-visible lines).
    pub fn set_document(&mut self, text: &str) {
        // Unchanged window (pure caret moves, in-window navigation): skip the
        // reshape entirely — it is the dominant per-keystroke cost.
        if self.doc_text == text {
            return;
        }
        self.doc_text.clear();
        self.doc_text.push_str(text);
        self.doc_buffer.set_text(
            text,
            &Attrs::new().family(Family::Monospace),
            Shaping::Advanced,
            None,
        );
        let (w, h) = self.doc_size();
        self.doc_buffer.set_size(Some(w), Some(h));
        self.doc_buffer
            .shape_until_scroll(&mut self.font_system, false);
    }

    /// Set the file-info half of the stats banner; the latency half is appended
    /// live in [`Renderer::render`].
    pub fn set_stats_prefix(&mut self, prefix: String) {
        self.stats_prefix = prefix;
    }

    /// Set the cursor position as `(line_in_window, column_in_chars)`, or
    /// `None` to hide it (scrolled off-screen).
    pub fn set_cursor(&mut self, pos: Option<(usize, usize)>) {
        self.cursor = pos;
    }

    /// Record the receipt time of a keystroke; the next present that includes
    /// its edit will close the keystroke->present latency sample.
    pub fn mark_keystroke(&mut self, t: Instant) {
        self.pending.push(t);
    }

    /// One-line latency summary with the D4 verdict, for stdout on exit.
    pub fn latency_summary(&self) -> String {
        self.latency.summary()
    }

    /// Adopt a new HiDPI scale factor (winit `ScaleFactorChanged`). Re-creates
    /// the shaped buffers at the new metrics and re-shapes current content.
    pub fn set_scale_factor(&mut self, scale_factor: f64) {
        if (scale_factor - self.scale_factor).abs() < f64::EPSILON {
            return;
        }
        self.scale_factor = scale_factor;
        let metrics = self.metrics();
        self.stats_buffer = Buffer::new(&mut self.font_system, metrics);
        self.doc_buffer = Buffer::new(&mut self.font_system, metrics);
        self.cursor_buffer = Buffer::new(&mut self.font_system, metrics);

        self.cursor_buffer.set_text(
            CURSOR_GLYPH,
            &Attrs::new().family(Family::Monospace),
            Shaping::Advanced,
            None,
        );
        self.cursor_buffer
            .set_size(Some(self.cell_w() * 2.0), Some(self.line_px()));
        self.cursor_buffer
            .shape_until_scroll(&mut self.font_system, false);

        self.last_stats.clear();
        let text = std::mem::take(&mut self.doc_text);
        self.set_document(&text);
    }

    /// Reconfigure the surface after a resize and reflow the text to it.
    pub fn resize(&mut self, width: u32, height: u32) {
        self.surface_config.width = width.max(1);
        self.surface_config.height = height.max(1);
        self.surface.configure(&self.device, &self.surface_config);

        let (w, h) = self.doc_size();
        self.doc_buffer.set_size(Some(w), Some(h));
        self.doc_buffer
            .shape_until_scroll(&mut self.font_system, false);
        // Force the banner to re-shape at the new width on the next frame.
        self.last_stats.clear();
    }

    /// Draw one frame. Returns `false` if the frame was skipped (surface
    /// lost/outdated) and a redraw was requested; `true` once a frame is
    /// actually presented (the caller uses the first `true` for cold-start).
    pub fn render(&mut self) -> bool {
        self.viewport.update(
            &self.queue,
            Resolution {
                width: self.surface_config.width,
                height: self.surface_config.height,
            },
        );

        // Compose the banner: file-info prefix + live latency percentiles. Only
        // re-shape it when the string actually changes (idle frames skip it).
        let lat = match self.latency.percentiles_cached() {
            Some((p50, p99, _)) => format!(
                "lat p50 {:.1}ms p99 {:.1}ms n={}",
                p50,
                p99,
                self.latency.count()
            ),
            None => "lat p50 -ms p99 -ms n=0".to_string(),
        };
        let stats = if self.stats_prefix.is_empty() {
            lat
        } else {
            format!("{}    {}", self.stats_prefix, lat)
        };
        if stats != self.last_stats {
            self.stats_buffer.set_text(
                &stats,
                &Attrs::new().family(Family::Monospace),
                Shaping::Advanced,
                None,
            );
            let sw = (self.surface_config.width as f32 - self.pad_px() * 2.0).max(1.0);
            self.stats_buffer.set_size(Some(sw), Some(self.line_px()));
            self.stats_buffer
                .shape_until_scroll(&mut self.font_system, false);
            self.last_stats = stats;
        }

        // Geometry snapshot (copies) so the TextArea borrows below only touch
        // the buffer fields, keeping them disjoint from the &mut atlas/font.
        let pad = self.pad_px();
        let doc_top = self.doc_top();
        let line_px = self.line_px();
        let cell_w = self.cell_w();
        let w = self.surface_config.width as i32;
        let h = self.surface_config.height as i32;

        let mut areas: Vec<TextArea> = Vec::with_capacity(3);
        areas.push(TextArea {
            buffer: &self.stats_buffer,
            left: pad,
            top: pad,
            scale: 1.0,
            bounds: TextBounds {
                left: 0,
                top: 0,
                right: w,
                bottom: h,
            },
            default_color: Color::rgb(150, 150, 165),
            custom_glyphs: &[],
        });
        areas.push(TextArea {
            buffer: &self.doc_buffer,
            left: pad,
            top: doc_top,
            scale: 1.0,
            bounds: TextBounds {
                left: 0,
                top: doc_top as i32,
                right: w,
                bottom: h,
            },
            default_color: Color::rgb(220, 220, 220),
            custom_glyphs: &[],
        });
        if let Some((line, col)) = self.cursor {
            let x = pad + col as f32 * cell_w;
            let y = doc_top + line as f32 * line_px;
            if y < h as f32 {
                areas.push(TextArea {
                    buffer: &self.cursor_buffer,
                    left: x,
                    top: y,
                    scale: 1.0,
                    bounds: TextBounds {
                        left: 0,
                        top: doc_top as i32,
                        right: w,
                        bottom: h,
                    },
                    // Crail rust accent (Claude Code palette).
                    default_color: Color::rgb(230, 180, 120),
                    custom_glyphs: &[],
                });
            }
        }

        if let Err(err) = self.text_renderer.prepare(
            &self.device,
            &self.queue,
            &mut self.font_system,
            &mut self.atlas,
            &self.viewport,
            areas,
            &mut self.swash_cache,
        ) {
            // Atlas out of space etc. — drop this frame, try again. Discard
            // pending keystroke timestamps too: letting them pile up across
            // repeated failures would drain as huge stale samples on recovery
            // and permanently corrupt the p99.
            eprintln!("umber-ui: text prepare failed: {err:?}");
            self.pending.clear();
            self.window.request_redraw();
            return false;
        }

        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame) => frame,
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => {
                self.window.request_redraw();
                return false;
            }
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Suboptimal(_) => {
                self.surface.configure(&self.device, &self.surface_config);
                self.window.request_redraw();
                return false;
            }
            wgpu::CurrentSurfaceTexture::Lost => {
                self.surface = self
                    .instance
                    .create_surface(self.window.clone())
                    .expect("recreate a wgpu surface");
                self.surface.configure(&self.device, &self.surface_config);
                self.window.request_redraw();
                return false;
            }
            wgpu::CurrentSurfaceTexture::Validation => {
                panic!("umber-ui: surface validation error");
            }
        };

        let view = frame.texture.create_view(&TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("umber-ui text pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations {
                        // Umber-dark background (D5 minimalist chrome).
                        load: LoadOp::Clear(wgpu::Color {
                            r: 0.06,
                            g: 0.06,
                            b: 0.07,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            self.text_renderer
                .render(&self.atlas, &self.viewport, &mut pass)
                .expect("render text");
        }

        self.queue.submit(Some(encoder.finish()));
        self.queue.present(frame);
        self.atlas.trim();

        // The frame carrying these keystrokes is now presented: close each
        // keystroke->present latency sample (D4 GO/NO-GO metric).
        if !self.pending.is_empty() {
            let now = Instant::now();
            for t in self.pending.drain(..) {
                self.latency.record(now.duration_since(t));
            }
        }
        true
    }
}
