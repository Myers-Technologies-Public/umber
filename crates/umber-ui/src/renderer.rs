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

use cosmic_text::{Attrs, Buffer, Color, Family, FontSystem, Metrics, Shaping, SwashCache, Wrap};
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

/// Logical-px gap between the gutter's last digit and the document text. The
/// gutter's reserved width is `digits * cell_w + GUTTER_GAP` (scaled).
const GUTTER_GAP: f32 = 12.0;

/// Dim gutter line-number color vs. the 220-grey body text (task spec).
const GUTTER_COLOR: Color = Color::rgb(105, 105, 120);

/// Gutter/document separator rule (straight-alpha RGBA). A subtle grey, dimmer
/// than the gutter digits, sitting in the gap between the line numbers and the
/// text. Thickness is `SEPARATOR_W` logical px scaled for HiDPI.
const SEPARATOR_COLOR: [f32; 4] = [0.32, 0.32, 0.38, 0.55];
const SEPARATOR_W: f32 = 1.0;

/// Hovered-line segment painted over the separator rule at the pointer's line:
/// warm gold (~rgb(212,175,55)) so the rule always shows which line the pointer
/// is on. Straight-alpha RGBA; stands out from the grey rule and the rust caret.
const SEPARATOR_HOVER_COLOR: [f32; 4] = [0.831, 0.686, 0.216, 0.9];

/// Hovered-word recolor: warm gold (rgb(212,175,55)) drawn over the original
/// glyphs at the word's grid cells. Reads on the dark bg, stands out from the
/// 220-grey body, and differs from the rust caret (230,180,120).
const HOVER_WORD_COLOR: Color = Color::rgb(212, 175, 55);

/// Ghostty-style overlay scrollbar visuals, logical px (scaled for HiDPI).
const SCROLLBAR_W: f32 = 10.0; // track/thumb width
const SCROLLBAR_EDGE: f32 = 16.0; // right-edge hover-activation zone
const SCROLLBAR_MARGIN: f32 = 2.0; // gap from the window's right edge
const SCROLLBAR_MIN_THUMB: f32 = 24.0; // floor so the thumb stays grabbable

/// Overlay quad colors (straight-alpha RGBA). Muted grey palette already used
/// by the banner \u{2014} deliberately NOT the rust cursor accent.
const TRACK_COLOR: [f32; 4] = [0.55, 0.55, 0.60, 0.10];
const THUMB_COLOR: [f32; 4] = [0.55, 0.55, 0.60, 0.55];

/// Selection highlight fill (straight-alpha RGBA). Muted grey-blue, translucent
/// so the glyphs drawn over it stay legible \u{2014} deliberately NOT the rust
/// cursor accent.
const SELECTION_COLOR: [f32; 4] = [0.30, 0.40, 0.58, 0.35];

/// Terminal panel (P3): height fraction of the window, border colors (the
/// border doubles as the focus cue), cursor cell fill, and grid text color.
const TERM_SPLIT_FRAC: f32 = 0.35;
const TERM_BORDER_COLOR: [f32; 4] = [0.32, 0.32, 0.38, 0.8];
const TERM_BORDER_FOCUS_COLOR: [f32; 4] = [0.902, 0.706, 0.471, 0.9];
const TERM_CURSOR_COLOR: [f32; 4] = [0.902, 0.706, 0.471, 0.45];
const TERM_TEXT_COLOR: Color = Color::rgb(210, 210, 215);

/// Left activity/tab bar (P5 QoL). Width in logical px; vertical tab glyphs
/// give a mouse backup for the palette/search/agents/terminal/settings views.
const SIDEBAR_W: f32 = 40.0;
const SIDEBAR_BG_COLOR: [f32; 4] = [0.085, 0.085, 0.105, 1.0];
/// Per-tab vertical pitch as a multiple of the line height.
const SIDEBAR_TAB_PITCH: f32 = 1.4;
/// Expanded activity-bar width (icons + text labels).
const SIDEBAR_W_EXPANDED: f32 = 168.0;
/// Tab labels shown when expanded (aligned to the glyph rows).
const SIDEBAR_LABELS: &str = "Palette\nFind\nAgents\nTerminal\nSettings";
const SIDEBAR_HOVER_COLOR: [f32; 4] = [0.55, 0.55, 0.62, 0.10];
/// 1px seam lines separating chrome regions (sidebar | content, banner /
/// strip / document).
const SEAM_COLOR: [f32; 4] = [0.28, 0.28, 0.34, 0.75];
/// Left accent bar marking the active tab (Crail rust).
const SIDEBAR_ACTIVE_COLOR: [f32; 4] = [0.902, 0.706, 0.471, 0.95];
/// Faint full-row tint behind the active tab so the whole item reads as
/// selected (not just the edge bar).
const SIDEBAR_ACTIVE_BG_COLOR: [f32; 4] = [0.902, 0.706, 0.471, 0.09];
const SIDEBAR_LABEL_COLOR: Color = Color::rgb(198, 198, 212);

/// Open-document tab strip (below the banner). Height multiple of line, bg,
/// active-tab tint, and text color.
const TABSTRIP_H_MULT: f32 = 1.3;
const TABSTRIP_BG_COLOR: [f32; 4] = [0.10, 0.10, 0.125, 1.0];
const TABSTRIP_ACTIVE_COLOR: [f32; 4] = [0.902, 0.706, 0.471, 0.14];
const TABSTRIP_TEXT_COLOR: Color = Color::rgb(205, 205, 218);

/// Modal overlay palette (command palette / settings / modules). All
/// straight-alpha RGBA. The dim quad darkens the still-visible editor behind
/// the modal; the box sits behind the palette input; the highlight marks the
/// selected row (subtle grey-blue, not the rust accent).
const OVERLAY_DIM_COLOR: [f32; 4] = [0.03, 0.03, 0.04, 0.72];
const OVERLAY_BOX_COLOR: [f32; 4] = [0.15, 0.15, 0.18, 0.92];
const OVERLAY_HL_COLOR: [f32; 4] = [0.30, 0.40, 0.58, 0.42];
/// Opaque panel behind overlay page content — without it the page text sits
/// directly on the dimmed editor and is hard to read (user-reported).
const OVERLAY_PANEL_COLOR: [f32; 4] = [0.10, 0.10, 0.125, 0.97];
/// Terminal panel background: a shade darker than the editor clear color so
/// the panel reads as a distinct surface.
const TERM_BG_COLOR: [f32; 4] = [0.042, 0.042, 0.052, 1.0];

/// Overlay text colors. Title uses the Crail rust accent (Claude Code
/// palette); input is bright; hint is dim. Row column colors are supplied
/// per-page in the [`OverlaySpec`].
const OVERLAY_TITLE_COLOR: Color = Color::rgb(230, 180, 120);
const OVERLAY_INPUT_COLOR: Color = Color::rgb(232, 232, 238);
const OVERLAY_HINT_COLOR: Color = Color::rgb(140, 140, 155);

/// Max solid quads the overlay pipeline draws per frame: one per visible
/// selected line plus the scrollbar track + thumb, the gutter separator rule,
/// and its hovered-line segment; six vertices each. Sizes the reused vertex
/// staging buffer, so it must cover the tallest realistic visible line count (a
/// 4K window is ~110 lines). The selection loop clamps to `QUAD_MAX - 4` so the
/// scrollbar (2) + separator (1) + hover segment (1) always have room.
const QUAD_MAX: usize = 256;
const QUAD_VERTS: usize = QUAD_MAX * 6;

/// Git gutter markers get their own vertex buffer (one thin quad per changed
/// visible line) so they never compete with the QUAD_MAX overlay budget. Cap
/// = a generous visible-line ceiling.
const GIT_MARK_MAX: usize = 512;
const GIT_MARK_VERTS: usize = GIT_MARK_MAX * 6;
/// Gutter marker colors (added / modified / deleted).
pub const GIT_ADDED_COLOR: [f32; 4] = [0.44, 0.73, 0.42, 0.95];
pub const GIT_MODIFIED_COLOR: [f32; 4] = [0.85, 0.65, 0.30, 0.95];
pub const GIT_DELETED_COLOR: [f32; 4] = [0.80, 0.35, 0.35, 0.95];
const QUAD_FLOATS_PER_VERT: usize = 6; // vec2 position + vec4 color

/// Minimal solid-quad shader: clip-space position + per-vertex color, alpha
/// blended over the text pass. glyphon cannot draw rectangles, so the overlay
/// scrollbar rides this tiny pipeline.
const QUAD_SHADER: &str = r#"
struct VsIn {
    @location(0) pos: vec2<f32>,
    @location(1) color: vec4<f32>,
};
struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec4<f32>,
};
@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    out.clip = vec4<f32>(in.pos, 0.0, 1.0);
    out.color = in.color;
    return out;
}
@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    return in.color;
}
"#;

/// A modal overlay to draw on top of (and dimming) the editor frame: the
/// command palette, settings page, or modules page. Built by the bin on a
/// state change and handed to [`Renderer::set_overlay`], which shapes it once;
/// [`Renderer::render`] then draws it every frame with no reshaping until the
/// next `set_overlay`.
///
/// Rows are a two-column layout (monospace): `left` in `left_color`, `right`
/// in `right_color`, with the right column starting at `split_frac` of the
/// content width. This covers all three pages without per-glyph rich text
/// (palette: title/keybinding; settings: label/value; modules: name/state).
pub struct OverlaySpec {
    /// Optional title line (settings/modules). Mutually exclusive with `input`
    /// in practice, but both are supported.
    pub title: Option<String>,
    /// Optional input line (palette query); rendered with a trailing caret.
    pub input: Option<String>,
    /// The list rows as `(left, right)` column strings.
    pub rows: Vec<(String, String)>,
    /// RGB of the left column and right column.
    pub left_color: [u8; 3],
    pub right_color: [u8; 3],
    /// Right column x as a fraction of the content width.
    pub split_frac: f32,
    /// Row index (into `rows`) to highlight, or `None` for no highlight.
    pub selected: Option<usize>,
    /// Optional bottom status hint.
    pub hint: Option<String>,
}

/// Scroll position the caller hands the renderer so it can size/place the
/// overlay scrollbar thumb. Absent = scrollbar hidden this frame.
#[derive(Clone, Copy)]
pub struct ScrollbarInfo {
    pub first_line: usize,
    pub total_lines: usize,
}

/// One run of selection highlight on a single visible line, in monospace column
/// units (converted to pixels in [`Renderer::render`] with the same `cell_w`
/// arithmetic as the caret so highlight and glyphs line up). `line` is
/// window-relative (0 = first visible line). `end_col == None` means "to the
/// right text edge" \u{2014} used for the first/interior lines of a multi-line
/// selection so the trailing newline reads as selected.
#[derive(Clone, Copy)]
pub struct SelSpan {
    pub line: usize,
    pub start_col: usize,
    pub end_col: Option<usize>,
}

/// Physical-pixel scrollbar rectangles, shared by the renderer's draw path and
/// the bin's pointer hit-testing so click and paint agree.
#[derive(Clone, Copy)]
pub struct ScrollbarGeom {
    pub track_x: f32,
    pub track_w: f32,
    pub track_top: f32,
    pub track_h: f32,
    pub thumb_top: f32,
    pub thumb_h: f32,
}

/// Append one axis-aligned rectangle (two triangles, six vertices) to `out` as
/// raw `f32` bytes in the `[pos.x, pos.y, r, g, b, a]` layout the quad pipeline
/// expects. Pixel coords are converted to clip space here. Returns the vertex
/// count added. `out` is a reused buffer \u{2014} no per-frame heap allocation.
fn push_quad(
    out: &mut Vec<u8>,
    fw: f32,
    fh: f32,
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    c: [f32; 4],
) -> u32 {
    let x0 = x / fw * 2.0 - 1.0;
    let x1 = (x + w) / fw * 2.0 - 1.0;
    let y0 = 1.0 - y / fh * 2.0;
    let y1 = 1.0 - (y + h) / fh * 2.0;
    let mut vert = |px: f32, py: f32| {
        out.extend_from_slice(&px.to_ne_bytes());
        out.extend_from_slice(&py.to_ne_bytes());
        for ch in c {
            out.extend_from_slice(&ch.to_ne_bytes());
        }
    };
    vert(x0, y0);
    vert(x1, y0);
    vert(x0, y1);
    vert(x1, y0);
    vert(x1, y1);
    vert(x0, y1);
    6
}

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
    /// Second text renderer sharing `atlas`, for the modal overlay so it can be
    /// composited in its own pass ON TOP of the dim quad (glyphon draws all of
    /// a renderer's areas in one pass, so the dim layer needs a separate one).
    overlay_text_renderer: TextRenderer,

    // Three shaped surfaces: banner, document window, cursor glyph.
    stats_buffer: Buffer,
    doc_buffer: Buffer,
    cursor_buffer: Buffer,

    // Modal overlay surfaces (command palette / settings / modules). Shaped
    // only in `set_overlay` (a state-change path), reused every frame.
    overlay_left: Buffer,
    overlay_right: Buffer,
    overlay_input: Buffer,
    overlay_title: Buffer,
    overlay_hint: Buffer,
    overlay_active: bool,
    overlay_has_input: bool,
    overlay_has_title: bool,
    overlay_has_hint: bool,
    overlay_row_count: usize,
    overlay_selected: Option<usize>,
    overlay_left_color: Color,
    overlay_right_color: Color,
    overlay_split_frac: f32,

    // Line-number gutter, shaped like the document with its own change guard.
    gutter_buffer: Buffer,
    gutter_text: String,
    gutter_digits: usize,
    /// Real shaped advance width of the gutter digits (physical px), measured
    /// after shaping so the reserved column reflects the font's true advance,
    /// not just `digits * cell_w`. See [`Renderer::gutter_text_w`].
    gutter_measured_w: f32,

    // Solid-quad overlay pipeline (scrollbar). `quad_bytes` is reused each
    // frame so render() performs no per-frame heap allocation.
    quad_pipeline: wgpu::RenderPipeline,
    quad_vbuf: wgpu::Buffer,
    quad_bytes: Vec<u8>,
    /// Git gutter markers (own buffer; see [`GIT_MARK_MAX`]). Each entry is
    /// `(line_in_window, rgba)`, set by the app from git line-status.
    git_vbuf: wgpu::Buffer,
    git_bytes: Vec<u8>,
    gutter_marks: Vec<(usize, [f32; 4])>,
    /// Left tab-bar background (own buffer, one quad, drawn behind the tab
    /// glyphs before the text pass).
    sidebar_vbuf: wgpu::Buffer,
    sidebar_bytes: Vec<u8>,
    /// `Some((first_line, total_lines))` when the scrollbar is visible.
    scrollbar: Option<(usize, usize)>,
    /// Selection highlight spans for the current view (window-relative lines).
    /// Reused across frames; rebuilt by the bin only when the selection changes.
    selection: Vec<SelSpan>,

    /// Hovered-word overlay: a small dedicated buffer (like `cursor_buffer`)
    /// holding just the hovered word, re-shaped only when the word text changes
    /// so a mouse move never reshapes the document. `hover_word` is its
    /// `(line_in_window, start_col)` grid position, `None` when no word is
    /// hovered. `hover_line` is the window-relative line whose separator segment
    /// is highlighted (set for both word and empty-space hover).
    hover_word_buffer: Buffer,
    hover_word_text: String,
    hover_word: Option<(usize, usize)>,
    hover_line: Option<usize>,

    /// Physical-pixel HiDPI scale (window.scale_factor()); folded into metrics,
    /// padding, bounds, and the cursor position.
    scale_factor: f64,
    /// Body-text metrics at scale 1.0 (logical px), from the TOML config (D13)
    /// and live-updated by [`Renderer::set_metrics`]. Multiplied by
    /// `scale_factor` for HiDPI. Default to the `BASE_FONT`/`BASE_LINE` consts.
    base_font: f32,
    base_line: f32,
    /// Live feature toggles (D10): the gutter column and the latency banner
    /// segment can be turned off, reclaiming their space.
    gutter_enabled: bool,
    latency_hud: bool,

    /// Cursor position as `(line_in_window, column_in_chars)`, or `None` when
    /// the cursor is scrolled out of the visible window.
    cursor: Option<(usize, usize)>,

    /// Terminal panel state (P3): open/focus flags, last grid snapshot, and
    /// the cell cursor. The buffer reshapes only when the snapshot changes.
    term_open: bool,
    term_focused: bool,
    /// Fullscreen terminal (fills below the banner).
    term_maximized: bool,
    /// User drag-resize height fraction override (else `TERM_SPLIT_FRAC`).
    term_split_frac_override: Option<f32>,
    term_text: String,
    term_cursor: Option<(usize, usize)>,
    term_buffer: Buffer,
    sidebar_enabled: bool,
    /// Expanded (icons + labels) vs collapsed (icons only).
    sidebar_expanded: bool,
    /// Tab under the pointer (hover highlight), and the active view's tab.
    sidebar_hover: Option<usize>,
    sidebar_active: Option<usize>,
    /// Text labels column, shown when expanded.
    sidebar_labels_buffer: Buffer,
    /// Cached joined tab-label text + row count for the left file-tab list.
    sidebar_tabs_text: String,
    sidebar_tab_count: usize,
    /// Open-document tab strip: shaped label row + per-tab char-column ranges
    /// for hit-testing + the active tab index.
    tabstrip_buffer: Buffer,
    tabstrip_text: String,
    tab_layout: Vec<(usize, usize)>,
    tab_active: usize,

    /// The last document window text, kept so a scale change can re-shape it
    /// without the caller re-supplying it.
    doc_text: String,
    /// The file-info half of the banner (the latency half is appended live).
    stats_prefix: String,
    /// Banner rebuild flag + last latency sample count. The banner string is
    /// composed only when an input changed (prefix, HUD toggle, geometry, or
    /// a new latency sample) — idle redraws allocate nothing.
    banner_dirty: bool,
    last_lat_n: u64,

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
        let overlay_text_renderer =
            TextRenderer::new(&mut atlas, &device, MultisampleState::default(), None);

        let metrics = Metrics::new(
            BASE_FONT * scale_factor as f32,
            BASE_LINE * scale_factor as f32,
        );
        // Wrapping is off on every multi-line surface: the click->line mapping
        // and cursor math assume one buffer line per visual line, so a wrapped
        // long line would corrupt caret targeting (and the gutter numbers would
        // wrap when the column is narrow). Long lines clip at the right edge.
        // TODO(P1): horizontal scroll for long lines.
        let mut stats_buffer = Buffer::new(&mut font_system, metrics);
        let mut term_buffer = Buffer::new(&mut font_system, metrics);
        term_buffer.set_wrap(Wrap::None);
        let sidebar_metrics = Metrics::new(
            BASE_FONT * scale_factor as f32,
            BASE_LINE * scale_factor as f32 * SIDEBAR_TAB_PITCH,
        );

        let mut sidebar_labels_buffer = Buffer::new(&mut font_system, sidebar_metrics);
        sidebar_labels_buffer.set_wrap(Wrap::None);
        sidebar_labels_buffer.set_text(
            SIDEBAR_LABELS,
            &Attrs::new().family(Family::Monospace),
            Shaping::Advanced,
            None,
        );
        sidebar_labels_buffer.shape_until_scroll(&mut font_system, false);
        let mut tabstrip_buffer = Buffer::new(&mut font_system, metrics);
        tabstrip_buffer.set_wrap(Wrap::None);
        stats_buffer.set_wrap(Wrap::None);
        let mut doc_buffer = Buffer::new(&mut font_system, metrics);
        doc_buffer.set_wrap(Wrap::None);
        let mut gutter_buffer = Buffer::new(&mut font_system, metrics);
        gutter_buffer.set_wrap(Wrap::None);

        // Modal overlay surfaces (shaped on demand by `set_overlay`).
        let mut overlay_left = Buffer::new(&mut font_system, metrics);
        overlay_left.set_wrap(Wrap::None);
        let mut overlay_right = Buffer::new(&mut font_system, metrics);
        overlay_right.set_wrap(Wrap::None);
        let mut overlay_input = Buffer::new(&mut font_system, metrics);
        overlay_input.set_wrap(Wrap::None);
        let mut overlay_title = Buffer::new(&mut font_system, metrics);
        overlay_title.set_wrap(Wrap::None);
        let mut overlay_hint = Buffer::new(&mut font_system, metrics);
        overlay_hint.set_wrap(Wrap::None);

        // The hovered-word overlay buffer: like the cursor glyph, shaped only
        // when the hovered word changes (never per mouse move).
        let mut hover_word_buffer = Buffer::new(&mut font_system, metrics);
        hover_word_buffer.set_wrap(Wrap::None);

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

        // Minimal solid-quad pipeline for the overlay scrollbar (glyphon draws
        // only glyphs). Color is a vertex attribute, so no bind groups/layout.
        let quad_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("umber-ui quad shader"),
            source: wgpu::ShaderSource::Wgsl(QUAD_SHADER.into()),
        });
        let quad_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("umber-ui quad pipeline"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &quad_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: (QUAD_FLOATS_PER_VERT * 4) as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x2,
                            offset: 0,
                            shader_location: 0,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x4,
                            offset: 8,
                            shader_location: 1,
                        },
                    ],
                })],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &quad_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: swapchain_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        let git_vbuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("umber git gutter marks"),
            size: (GIT_MARK_VERTS * QUAD_FLOATS_PER_VERT * 4) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sidebar_vbuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("umber sidebar bg"),
            size: (96 * QUAD_FLOATS_PER_VERT * 4) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let quad_vbuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("umber-ui quad vertices"),
            size: (QUAD_VERTS * QUAD_FLOATS_PER_VERT * 4) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

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
            overlay_text_renderer,
            stats_buffer,
            doc_buffer,
            cursor_buffer,
            gutter_buffer,
            gutter_text: String::new(),
            gutter_digits: 0,
            gutter_measured_w: 0.0,
            overlay_active: false,
            overlay_left,
            overlay_right,
            overlay_input,
            overlay_title,
            overlay_hint,
            overlay_has_input: false,
            overlay_has_title: false,
            overlay_has_hint: false,
            overlay_row_count: 0,
            overlay_selected: None,
            overlay_left_color: Color::rgb(220, 220, 220),
            overlay_right_color: Color::rgb(150, 150, 150),
            overlay_split_frac: 0.5,
            quad_pipeline,
            quad_vbuf,
            quad_bytes: Vec::with_capacity(QUAD_VERTS * QUAD_FLOATS_PER_VERT * 4),
            git_vbuf,
            git_bytes: Vec::with_capacity(GIT_MARK_VERTS * QUAD_FLOATS_PER_VERT * 4),
            gutter_marks: Vec::new(),
            sidebar_vbuf,
            sidebar_bytes: Vec::with_capacity(96 * QUAD_FLOATS_PER_VERT * 4),
            scrollbar: None,
            selection: Vec::new(),
            hover_word_buffer,
            hover_word_text: String::new(),
            hover_word: None,
            hover_line: None,
            scale_factor,
            base_font: BASE_FONT,
            base_line: BASE_LINE,
            gutter_enabled: true,
            latency_hud: true,
            cursor: None,
            term_open: false,
            term_focused: false,
            term_maximized: false,
            term_split_frac_override: None,
            term_text: String::new(),
            term_cursor: None,
            term_buffer,
            sidebar_enabled: true,
            sidebar_expanded: true,
            sidebar_hover: None,
            sidebar_active: None,
            sidebar_labels_buffer,
            sidebar_tabs_text: String::new(),
            sidebar_tab_count: 0,
            tabstrip_buffer,
            tabstrip_text: String::new(),
            tab_layout: Vec::new(),
            tab_active: 0,
            doc_text: String::new(),
            stats_prefix: String::new(),
            banner_dirty: true,
            last_lat_n: 0,
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
        self.base_font * self.scale_factor as f32
    }

    pub fn line_px(&self) -> f32 {
        self.base_line * self.scale_factor as f32
    }

    fn pad_px(&self) -> f32 {
        PAD * self.scale_factor as f32
    }

    pub fn cell_w(&self) -> f32 {
        self.font_px() * MONO_ADVANCE_RATIO
    }

    /// Width in px reserved for the gutter's digits (no trailing gap). Taken as
    /// the larger of the arithmetic `digits * cell_w` estimate and the real
    /// shaped advance measured in [`Renderer::set_gutter`], so a font whose
    /// monospace advance disagrees with `MONO_ADVANCE_RATIO` never clips.
    fn gutter_text_w(&self) -> f32 {
        let arithmetic = self.gutter_digits as f32 * self.cell_w();
        arithmetic.max(self.gutter_measured_w)
    }

    /// Total px reserved for the gutter column (digits + trailing gap), or 0
    /// when no line count has been supplied yet.
    fn gutter_width(&self) -> f32 {
        if !self.gutter_enabled || self.gutter_digits == 0 {
            0.0
        } else {
            self.gutter_text_w() + GUTTER_GAP * self.scale_factor as f32
        }
    }

    /// X of the document text origin: window pad + gutter column. The bin maps
    /// clicks against this; the renderer places glyphs and the cursor from it.
    /// Screen rect `(x, y, w, h)` of the settings gear glyph at the start of
    /// the top banner (D5: keyboard-first, mouse-hover backup). The bin
    /// hit-tests clicks against this to open settings.
    pub fn gear_bounds(&self) -> (f32, f32, f32, f32) {
        (
            self.left_edge(),
            self.pad_px(),
            self.cell_w() * 2.5,
            self.line_px(),
        )
    }

    /// Width of the left activity bar in physical px (0 when disabled).
    pub fn sidebar_w(&self) -> f32 {
        if !self.sidebar_enabled {
            return 0.0;
        }
        let base = if self.sidebar_expanded {
            SIDEBAR_W_EXPANDED
        } else {
            SIDEBAR_W
        };
        base * self.scale_factor as f32
    }

    pub fn sidebar_expanded(&self) -> bool {
        self.sidebar_expanded
    }

    /// Expand/collapse the activity bar; reflows content to the new left edge.
    pub fn set_sidebar_expanded(&mut self, expanded: bool) {
        if self.sidebar_expanded == expanded {
            return;
        }
        self.sidebar_expanded = expanded;
        self.reflow_terminal_geometry();
        self.banner_dirty = true;
    }

    /// Set the hovered tab (highlight), returns true if it changed.
    pub fn set_sidebar_hover(&mut self, hover: Option<usize>) -> bool {
        if self.sidebar_hover == hover {
            return false;
        }
        self.sidebar_hover = hover;
        true
    }

    /// Set the active tab (accent bar), matching the current view.
    pub fn set_sidebar_active(&mut self, active: Option<usize>) {
        self.sidebar_active = active;
    }

    /// Set the left file-tab list: one label per open tab + the active index.
    /// Reshapes only when the label text changes.
    pub fn set_sidebar_tabs(&mut self, labels: &[String], active: usize) {
        self.sidebar_tab_count = labels.len();
        self.sidebar_active = if labels.is_empty() {
            None
        } else {
            Some(active.min(labels.len() - 1))
        };
        let text = labels.join("\n");
        if self.sidebar_tabs_text != text {
            self.sidebar_tabs_text.clear();
            self.sidebar_tabs_text.push_str(&text);
            self.sidebar_labels_buffer.set_text(
                &text,
                &Attrs::new().family(Family::Monospace),
                Shaping::Advanced,
                None,
            );
            let w = self.sidebar_w().max(1.0);
            let hgt = self.surface_config.height as f32;
            self.sidebar_labels_buffer.set_size(Some(w), Some(hgt));
            self.sidebar_labels_buffer
                .shape_until_scroll(&mut self.font_system, false);
        }
    }

    /// Left content edge: past the activity bar, plus window padding. Banner,
    /// gutter, and terminal all originate here so the sidebar shift is applied
    /// in exactly one place.
    pub fn left_edge(&self) -> f32 {
        self.sidebar_w() + self.pad_px()
    }

    /// Y where the sidebar file-tab list begins: below the banner + activity
    /// strip line, aligned with the document top (keeps the window corner
    /// clean instead of crowding a tab against it).
    fn sidebar_top(&self) -> f32 {
        self.doc_top() + self.line_px() * 0.2
    }

    /// Sidebar tab index at physical `(x, y)`, or `None`. Tabs stack from the
    /// top at `SIDEBAR_TAB_PITCH` line-heights, matching the shaped column.
    pub fn sidebar_tab_at(&self, x: f32, y: f32) -> Option<usize> {
        if !self.sidebar_enabled || x < 0.0 || x > self.sidebar_w() {
            return None;
        }
        let top = self.sidebar_top();
        if y < top {
            return None;
        }
        let pitch = self.line_px() * SIDEBAR_TAB_PITCH;
        let row = ((y - top) / pitch).floor() as usize;
        if row < self.sidebar_tab_count {
            Some(row)
        } else {
            None
        }
    }

    pub fn text_left(&self) -> f32 {
        self.left_edge() + self.gutter_width()
    }

    /// Y of the document top: below the stats banner.
    pub fn doc_top(&self) -> f32 {
        self.pad_px() + self.line_px() * STATS_GAP + self.tabstrip_h()
    }

    /// Y of the tab strip (just below the banner).
    pub fn tabstrip_top(&self) -> f32 {
        self.pad_px() + self.line_px() * STATS_GAP
    }

    /// Tab strip height (0 when no tabs are set).
    pub fn tabstrip_h(&self) -> f32 {
        if self.tab_layout.is_empty() {
            0.0
        } else {
            self.line_px() * TABSTRIP_H_MULT
        }
    }

    /// Set the open-document tabs (labels + active index). Reshapes the strip
    /// only when the label text changes.
    pub fn set_tabs(&mut self, labels: &[String], active: usize) {
        self.tab_active = active;
        let sep = "    ";
        let mut text = String::new();
        self.tab_layout.clear();
        let mut col = 0usize;
        for (i, label) in labels.iter().enumerate() {
            if i > 0 {
                text.push_str(sep);
                col += sep.chars().count();
            }
            let start = col;
            text.push_str(label);
            col += label.chars().count();
            self.tab_layout.push((start, col));
        }
        if self.tabstrip_text != text {
            self.tabstrip_text.clear();
            self.tabstrip_text.push_str(&text);
            self.tabstrip_buffer.set_text(
                &text,
                &Attrs::new().family(Family::Monospace),
                Shaping::Advanced,
                None,
            );
            let w = (self.surface_config.width as f32).max(1.0);
            self.tabstrip_buffer.set_size(Some(w), Some(self.line_px()));
            self.tabstrip_buffer
                .shape_until_scroll(&mut self.font_system, false);
        }
    }

    /// Tab index at physical `(x, y)` in the strip, or `None`.
    pub fn tabstrip_at(&self, x: f32, y: f32) -> Option<usize> {
        if self.tab_layout.is_empty() {
            return None;
        }
        let top = self.tabstrip_top();
        if y < top || y >= top + self.tabstrip_h() {
            return None;
        }
        let origin = self.left_edge() + self.pad_px() * 0.5;
        if x < origin {
            return None;
        }
        let col = ((x - origin) / self.cell_w()) as usize;
        self.tab_layout
            .iter()
            .position(|(s, e)| col >= *s && col < *e)
    }

    fn metrics(&self) -> Metrics {
        Metrics::new(self.font_px(), self.line_px())
    }

    /// Document shaping box in physical pixels (width and visible height).
    fn doc_size(&self) -> (f32, f32) {
        let w = (self.surface_config.width as f32 - self.text_left() - self.pad_px()).max(1.0);
        let h = (self.doc_bottom() - self.doc_top() - self.pad_px()).max(1.0);
        (w, h)
    }

    /// Y of the document region's bottom edge: the terminal panel's top when
    /// the panel is open, else the window bottom. Every consumer of the doc
    /// region (shaping box, line capacity, scrollbar track, hover/click
    /// mapping in the bin) derives from this so the panel shrink is uniform.
    pub fn doc_bottom(&self) -> f32 {
        if self.term_open {
            self.term_top()
        } else {
            self.surface_config.height as f32
        }
    }

    /// Terminal panel height in physical px (whole line-heights + padding),
    /// or 0 when closed.
    pub fn term_split_h(&self) -> f32 {
        if !self.term_open {
            return 0.0;
        }
        // Maximized: the panel fills everything below the banner (fullscreen
        // terminal, banner kept for the gear + status).
        if self.term_maximized {
            return (self.surface_config.height as f32 - self.doc_top()).max(self.line_px());
        }
        let h = self.surface_config.height as f32;
        let frac = self.term_split_frac_override.unwrap_or(TERM_SPLIT_FRAC);
        let lines = ((h * frac) / self.line_px()).floor().max(2.0);
        lines * self.line_px() + self.pad_px() * 2.0
    }

    /// Y of the terminal panel top (== window bottom when closed).
    pub fn term_top(&self) -> f32 {
        self.surface_config.height as f32 - self.term_split_h()
    }

    /// Terminal grid size `(cols, lines)` for PTY sizing.
    pub fn term_grid_size(&self) -> (usize, usize) {
        // Width available to the grid = window minus the sidebar + padding,
        // or long lines wrap/clip at the right edge.
        let cols = ((self.surface_config.width as f32 - self.left_edge() - self.pad_px())
            / self.cell_w())
        .floor()
        .max(1.0) as usize;
        let lines = ((self.term_split_h() - self.pad_px() * 2.0) / self.line_px())
            .floor()
            .max(1.0) as usize;
        (cols, lines)
    }

    /// Cell size `(width, height)` in physical px for the PTY `WindowSize`.
    pub fn cell_px(&self) -> (u16, u16) {
        (
            self.cell_w().round().max(1.0) as u16,
            self.line_px().round().max(1.0) as u16,
        )
    }

    /// How many whole document lines fit in the current window. The caller uses
    /// this to size the scroll window (docs/PLAN.md: shape only visible lines).
    pub fn visible_line_capacity(&self) -> usize {
        let avail = self.doc_bottom() - self.doc_top() - self.pad_px();
        if avail <= 0.0 {
            0
        } else {
            (avail / self.line_px()).floor() as usize
        }
    }

    /// Width in px of the right-edge zone whose hover reveals the scrollbar.
    pub fn scrollbar_edge_zone(&self) -> f32 {
        SCROLLBAR_EDGE * self.scale_factor as f32
    }

    /// Physical-px scrollbar rectangles for `(first_line, total_lines)`, or
    /// `None` when the document fits (no scrollbar). Shared by draw + hit-test.
    pub fn scrollbar_geom(&self, first_line: usize, total_lines: usize) -> Option<ScrollbarGeom> {
        let visible = self.visible_line_capacity();
        if visible == 0 || total_lines <= visible {
            return None;
        }
        let s = self.scale_factor as f32;
        let track_w = SCROLLBAR_W * s;
        let track_x = self.surface_config.width as f32 - track_w - SCROLLBAR_MARGIN * s;
        let track_top = self.doc_top();
        let track_h = (self.doc_bottom() - track_top).max(1.0);
        let min_thumb = (SCROLLBAR_MIN_THUMB * s).min(track_h);
        let thumb_h = (track_h * visible as f32 / total_lines as f32).max(min_thumb);
        let scroll_range = (total_lines - visible) as f32;
        let frac = if scroll_range > 0.0 {
            (first_line as f32 / scroll_range).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let thumb_top = track_top + frac * (track_h - thumb_h);
        Some(ScrollbarGeom {
            track_x,
            track_w,
            track_top,
            track_h,
            thumb_top,
            thumb_h,
        })
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

    /// Replace the shaped line-number gutter for the visible window. `numbers`
    /// is one right-aligned number per shaped line; `digits` is the digit count
    /// of the whole file's last line, which fixes the column width so it never
    /// jitters while scrolling. Mirrors [`Renderer::set_document`]'s
    /// only-on-change guard: the string changes exactly when the first visible
    /// line or the line count changes.
    pub fn set_gutter(&mut self, numbers: &str, digits: usize) {
        let digits_changed = digits != self.gutter_digits;
        if !digits_changed && self.gutter_text == numbers {
            return;
        }
        self.gutter_digits = digits;
        self.gutter_text.clear();
        self.gutter_text.push_str(numbers);

        let gw = self.gutter_text_w().max(1.0);
        let gh = self.doc_size().1;
        self.gutter_buffer.set_text(
            numbers,
            &Attrs::new().family(Family::Monospace),
            Shaping::Advanced,
            None,
        );
        self.gutter_buffer.set_size(Some(gw), Some(gh));
        self.gutter_buffer
            .shape_until_scroll(&mut self.font_system, false);

        // Measure the real shaped width so `gutter_text_w` (hence the document
        // origin) tracks the font's actual advance, not just the arithmetic
        // estimate. Wrapping is off, so `line_w` is the natural unclipped width.
        let mut measured = 0.0_f32;
        for run in self.gutter_buffer.layout_runs() {
            measured = measured.max(run.line_w);
        }
        self.gutter_measured_w = measured;

        // A digit-count change moves the document origin: adopt the new box
        // size but leave the (single) reshape to the `set_document` call that
        // follows in the same `apply_view` — reshaping here too would layout
        // the OLD text once and the new text again. Clearing the text cache
        // guarantees `set_document` cannot early-return with a stale-width
        // layout even if the window text happens to be unchanged.
        if digits_changed {
            let (dw, dh) = self.doc_size();
            self.doc_buffer.set_size(Some(dw), Some(dh));
            self.doc_text.clear();
        }
    }

    /// Set the file-info half of the stats banner; the latency half is appended
    /// live in [`Renderer::render`].
    pub fn set_stats_prefix(&mut self, prefix: String) {
        if prefix != self.stats_prefix {
            self.stats_prefix = prefix;
            self.banner_dirty = true;
        }
    }

    /// Set the cursor position as `(line_in_window, column_in_chars)`, or
    /// `None` to hide it (scrolled off-screen).
    pub fn set_cursor(&mut self, pos: Option<(usize, usize)>) {
        self.cursor = pos;
    }

    /// Supply the scrollbar's scroll position for this frame, or `None` to hide
    /// it. Visibility timing (show-on-scroll/hover/drag, ~800 ms linger) lives
    /// in the bin's event loop; this just carries the paint state.
    pub fn set_scrollbar(&mut self, info: Option<ScrollbarInfo>) {
        self.scrollbar = info.map(|i| (i.first_line, i.total_lines));
    }

    /// Replace the selection highlight spans (window-relative lines). The bin
    /// rebuilds these only when the selection changes; they land in a reused
    /// Vec so [`Renderer::render`] performs no per-frame allocation.
    pub fn set_selection(&mut self, spans: &[SelSpan]) {
        self.selection.clear();
        self.selection.extend_from_slice(spans);
    }

    /// Set (or clear) the hovered word: `(line_in_window, start_col, text)`.
    /// The dedicated word buffer is re-shaped ONLY when `text` differs from the
    /// last hovered word, so moving between two same-text words (or repeated
    /// calls with an unchanged word) never reshapes. `None` hides the recolor.
    /// The bin calls this only when the hover target actually changes.
    pub fn set_hover_word(&mut self, word: Option<(usize, usize, &str)>) {
        match word {
            Some((line, start_col, text)) => {
                self.hover_word = Some((line, start_col));
                if self.hover_word_text != text {
                    self.hover_word_text.clear();
                    self.hover_word_text.push_str(text);
                    self.hover_word_buffer.set_text(
                        text,
                        &Attrs::new().family(Family::Monospace),
                        Shaping::Advanced,
                        None,
                    );
                    // One extra cell of width headroom so the last glyph never
                    // clips; height is one line.
                    let w = self.cell_w() * (text.chars().count() as f32 + 1.0);
                    self.hover_word_buffer
                        .set_size(Some(w.max(1.0)), Some(self.line_px()));
                    self.hover_word_buffer
                        .shape_until_scroll(&mut self.font_system, false);
                }
            }
            // Intentionally leaves `hover_word_text` + the shaped buffer
            // intact: an invisible surface costs nothing, and re-hovering the
            // same word (the common flicker case) skips the reshape entirely.
            // `rebuild_shaped_buffers` clears the cache on metrics changes.
            None => self.hover_word = None,
        }
    }

    /// Set (or clear) the window-relative line whose separator segment is
    /// highlighted. Cheap: just stored, drawn as a quad in [`Renderer::render`].
    pub fn set_hover_line(&mut self, line: Option<usize>) {
        self.hover_line = line;
    }

    /// Set git gutter markers as `(line_in_window, rgba)` pairs (P5). Cheap:
    /// stored, drawn as quads from the dedicated git buffer in `render`.
    pub fn set_gutter_marks(&mut self, marks: Vec<(usize, [f32; 4])>) {
        self.gutter_marks = marks;
    }

    /// Show/hide the terminal panel and set its keyboard focus. An open/close
    /// changes the document geometry, so the doc + gutter reflow here; the
    /// caller must re-apply its view (line window) afterwards.
    pub fn set_terminal(&mut self, open: bool, focused: bool) {
        if self.term_open == open && self.term_focused == focused {
            return;
        }
        let geometry_changed = self.term_open != open;
        self.term_open = open;
        self.term_focused = focused;
        if geometry_changed {
            let (w, h) = self.doc_size();
            self.doc_buffer.set_size(Some(w), Some(h));
            self.doc_buffer
                .shape_until_scroll(&mut self.font_system, false);
            let gw = self.gutter_text_w().max(1.0);
            let gh = self.doc_size().1;
            self.gutter_buffer.set_size(Some(gw), Some(gh));
            self.gutter_buffer
                .shape_until_scroll(&mut self.font_system, false);
        }
    }

    /// Replace the terminal grid snapshot + cursor cell. Reshapes only when
    /// the snapshot text actually changed (the coalesced-wakeup path).
    pub fn set_terminal_text(&mut self, text: &str, cursor: Option<(usize, usize)>) {
        self.term_cursor = cursor;
        if self.term_text == text {
            return;
        }
        self.term_text.clear();
        self.term_text.push_str(text);
        self.term_buffer.set_text(
            text,
            &Attrs::new().family(Family::Monospace),
            Shaping::Advanced,
            None,
        );
        let w = (self.surface_config.width as f32 - self.left_edge() - self.pad_px()).max(1.0);
        let h = (self.term_split_h() - self.pad_px()).max(1.0);
        self.term_buffer.set_size(Some(w), Some(h));
        self.term_buffer
            .shape_until_scroll(&mut self.font_system, false);
    }

    pub fn terminal_open(&self) -> bool {
        self.term_open
    }

    pub fn terminal_focused(&self) -> bool {
        self.term_focused
    }

    pub fn terminal_maximized(&self) -> bool {
        self.term_maximized
    }

    /// Toggle/set fullscreen terminal, reflowing document + gutter + grid.
    pub fn set_terminal_maximized(&mut self, maximized: bool) {
        if self.term_maximized == maximized {
            return;
        }
        self.term_maximized = maximized;
        self.reflow_terminal_geometry();
    }

    /// Set the drag-resize split fraction (clamped 0.1..0.85); reflows.
    pub fn set_terminal_split_frac(&mut self, frac: f32) {
        if self.term_maximized {
            return;
        }
        self.term_split_frac_override = Some(frac.clamp(0.1, 0.85));
        self.reflow_terminal_geometry();
    }

    /// Reflow document, gutter, and terminal buffers to the current panel
    /// geometry (shared by maximize + drag-resize).
    fn reflow_terminal_geometry(&mut self) {
        let (w, h) = self.doc_size();
        self.doc_buffer.set_size(Some(w), Some(h));
        self.doc_buffer
            .shape_until_scroll(&mut self.font_system, false);
        let gw = self.gutter_text_w().max(1.0);
        let gh = self.doc_size().1;
        self.gutter_buffer.set_size(Some(gw), Some(gh));
        self.gutter_buffer
            .shape_until_scroll(&mut self.font_system, false);
        if self.term_open {
            let tw = (self.surface_config.width as f32 - self.pad_px() * 2.0).max(1.0);
            let th = (self.term_split_h() - self.pad_px()).max(1.0);
            self.term_buffer.set_size(Some(tw), Some(th));
            self.term_buffer
                .shape_until_scroll(&mut self.font_system, false);
        }
    }

    /// Window-relative list-row index at physical-y `y` on an overlay page, or
    /// `None` above the first row / below the last visible row. Lets the app
    /// map a mouse click to a settings/modules row.
    pub fn overlay_row_at(&self, y: f32) -> Option<usize> {
        let rows_top = self.overlay_rows_top();
        if y < rows_top {
            return None;
        }
        let row = ((y - rows_top) / self.line_px()).floor() as usize;
        if row < self.overlay_row_capacity() {
            Some(row)
        } else {
            None
        }
    }

    // --- modal overlay (command palette / settings / modules) --------------

    /// Left x of the overlay content box (physical px).
    fn overlay_content_left(&self) -> f32 {
        self.pad_px() * 3.0
    }

    /// Width of the overlay content box (physical px).
    fn overlay_content_width(&self) -> f32 {
        (self.surface_config.width as f32 - self.overlay_content_left() * 2.0).max(1.0)
    }

    /// Y of the overlay header line (title or input).
    fn overlay_top(&self) -> f32 {
        self.pad_px() + self.line_px()
    }

    /// Y of the first overlay list row (below the header, if any).
    fn overlay_rows_top(&self) -> f32 {
        if self.overlay_has_input || self.overlay_has_title {
            self.overlay_top() + self.line_px() * 1.6
        } else {
            self.overlay_top()
        }
    }

    /// How many list rows fit below the overlay header and above the hint line.
    /// The bin uses this to window a long command list around the selection.
    pub fn overlay_row_capacity(&self) -> usize {
        let top = self.pad_px() + self.line_px() + self.line_px() * 1.6;
        let avail = self.surface_config.height as f32 - top - self.pad_px() - self.line_px();
        if avail <= 0.0 {
            1
        } else {
            (avail / self.line_px()).floor().max(1.0) as usize
        }
    }

    /// Install (or clear with `None`) the modal overlay. Shapes the supplied
    /// text once here (the change path); [`Renderer::render`] then draws it
    /// every frame with no reshaping until the next `set_overlay`.
    pub fn set_overlay(&mut self, spec: Option<OverlaySpec>) {
        let spec = match spec {
            Some(s) => s,
            None => {
                self.overlay_active = false;
                return;
            }
        };
        self.overlay_active = true;
        self.overlay_row_count = spec.rows.len();
        self.overlay_selected = spec.selected;
        self.overlay_left_color =
            Color::rgb(spec.left_color[0], spec.left_color[1], spec.left_color[2]);
        self.overlay_right_color = Color::rgb(
            spec.right_color[0],
            spec.right_color[1],
            spec.right_color[2],
        );
        self.overlay_split_frac = spec.split_frac;

        let attrs = Attrs::new().family(Family::Monospace);
        let content_w = self.overlay_content_width();
        let tall = self.surface_config.height as f32;
        let line_px = self.line_px();

        // Two-column list shaped as two multi-line buffers (monospace, so
        // per-column uniform color needs no rich text).
        let mut left = String::new();
        let mut right = String::new();
        for (i, (l, r)) in spec.rows.iter().enumerate() {
            if i > 0 {
                left.push('\n');
                right.push('\n');
            }
            left.push_str(l);
            right.push_str(r);
        }
        self.overlay_left
            .set_text(&left, &attrs, Shaping::Advanced, None);
        self.overlay_left.set_size(Some(content_w), Some(tall));
        self.overlay_left
            .shape_until_scroll(&mut self.font_system, false);
        let right_w = (content_w * (1.0 - self.overlay_split_frac)).max(1.0);
        self.overlay_right
            .set_text(&right, &attrs, Shaping::Advanced, None);
        self.overlay_right.set_size(Some(right_w), Some(tall));
        self.overlay_right
            .shape_until_scroll(&mut self.font_system, false);

        self.overlay_has_input = spec.input.is_some();
        if let Some(input) = &spec.input {
            // A trailing one-eighth block reads as the input caret.
            let line = format!("{input}\u{258f}");
            self.overlay_input
                .set_text(&line, &attrs, Shaping::Advanced, None);
            self.overlay_input.set_size(Some(content_w), Some(line_px));
            self.overlay_input
                .shape_until_scroll(&mut self.font_system, false);
        }

        self.overlay_has_title = spec.title.is_some();
        if let Some(title) = &spec.title {
            self.overlay_title
                .set_text(title, &attrs, Shaping::Advanced, None);
            self.overlay_title.set_size(Some(content_w), Some(line_px));
            self.overlay_title
                .shape_until_scroll(&mut self.font_system, false);
        }

        self.overlay_has_hint = spec.hint.is_some();
        if let Some(hint) = &spec.hint {
            self.overlay_hint
                .set_text(hint, &attrs, Shaping::Advanced, None);
            self.overlay_hint.set_size(Some(content_w), Some(line_px));
            self.overlay_hint
                .shape_until_scroll(&mut self.font_system, false);
        }
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

    /// Re-create every shaped buffer at the current metrics and re-shape the
    /// current content. Shared by [`Renderer::set_scale_factor`] (HiDPI change)
    /// and [`Renderer::set_metrics`] (live config font/line change).
    fn rebuild_shaped_buffers(&mut self) {
        let metrics = self.metrics();
        self.stats_buffer = Buffer::new(&mut self.font_system, metrics);
        self.term_buffer = Buffer::new(&mut self.font_system, metrics);
        self.term_buffer.set_wrap(Wrap::None);
        let term_text = std::mem::take(&mut self.term_text);
        let term_cursor = self.term_cursor;
        self.set_terminal_text(&term_text, term_cursor);
        self.stats_buffer.set_wrap(Wrap::None);
        self.doc_buffer = Buffer::new(&mut self.font_system, metrics);
        self.doc_buffer.set_wrap(Wrap::None);
        self.gutter_buffer = Buffer::new(&mut self.font_system, metrics);
        self.gutter_buffer.set_wrap(Wrap::None);
        self.cursor_buffer = Buffer::new(&mut self.font_system, metrics);
        self.hover_word_buffer = Buffer::new(&mut self.font_system, metrics);
        self.hover_word_buffer.set_wrap(Wrap::None);
        // Force the hovered word to re-shape at the new metrics on its next set.
        self.hover_word_text.clear();
        // Overlay surfaces carry the metrics too; recreate them empty and let
        // the next `set_overlay` repopulate (the bin refreshes the overlay
        // right after a live metrics change).
        self.overlay_left = Buffer::new(&mut self.font_system, metrics);
        self.overlay_left.set_wrap(Wrap::None);
        self.overlay_right = Buffer::new(&mut self.font_system, metrics);
        self.overlay_right.set_wrap(Wrap::None);
        self.overlay_input = Buffer::new(&mut self.font_system, metrics);
        self.overlay_input.set_wrap(Wrap::None);
        self.overlay_title = Buffer::new(&mut self.font_system, metrics);
        self.overlay_title.set_wrap(Wrap::None);
        self.overlay_hint = Buffer::new(&mut self.font_system, metrics);
        self.overlay_hint.set_wrap(Wrap::None);

        // Force the gutter to re-shape at the new metrics on the next view push.
        self.gutter_text.clear();
        // Advance scales with the font, so the measured gutter width is stale.
        self.gutter_measured_w = 0.0;

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

        self.banner_dirty = true;
        let text = std::mem::take(&mut self.doc_text);
        self.set_document(&text);
    }

    /// Adopt a new HiDPI scale factor (winit `ScaleFactorChanged`). Re-creates
    /// the shaped buffers at the new metrics and re-shapes current content.
    pub fn set_scale_factor(&mut self, scale_factor: f64) {
        if (scale_factor - self.scale_factor).abs() < f64::EPSILON {
            return;
        }
        self.scale_factor = scale_factor;
        self.rebuild_shaped_buffers();
    }

    /// Live-apply the config body metrics (D13): font size + line height in
    /// logical px. Rebuilds shaped buffers exactly like a scale change so edits
    /// from the settings page take effect immediately.
    pub fn set_metrics(&mut self, font_size: f32, line_height: f32) {
        if (self.base_font - font_size).abs() < f32::EPSILON
            && (self.base_line - line_height).abs() < f32::EPSILON
        {
            return;
        }
        self.base_font = font_size;
        self.base_line = line_height;
        self.rebuild_shaped_buffers();
    }

    /// Enable/disable the line-number gutter (config `gutter`). Disabling
    /// reclaims the gutter width for the document and forces a reshape at the
    /// new text origin.
    pub fn set_gutter_enabled(&mut self, on: bool) {
        if self.gutter_enabled == on {
            return;
        }
        self.gutter_enabled = on;
        let (dw, dh) = self.doc_size();
        self.doc_buffer.set_size(Some(dw), Some(dh));
        // Force `set_document` to reshape even if the window text is unchanged.
        self.doc_text.clear();
    }

    /// Show/hide the banner latency segment (config `latency_hud`).
    pub fn set_latency_hud(&mut self, on: bool) {
        if self.latency_hud == on {
            return;
        }
        self.latency_hud = on;
        // Force the banner to re-compose on the next frame.
        self.banner_dirty = true;
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
        // Reflow the gutter to the new viewport height (its width is unchanged).
        let gw = self.gutter_text_w().max(1.0);
        let gh = self.doc_size().1;
        self.gutter_buffer.set_size(Some(gw), Some(gh));
        self.gutter_buffer
            .shape_until_scroll(&mut self.font_system, false);
        // Reflow the terminal snapshot to the new panel box.
        if self.term_open {
            let tw = (self.surface_config.width as f32 - self.pad_px() * 2.0).max(1.0);
            let th = (self.term_split_h() - self.pad_px()).max(1.0);
            self.term_buffer.set_size(Some(tw), Some(th));
            self.term_buffer
                .shape_until_scroll(&mut self.font_system, false);
        }
        // Force the banner to re-shape at the new width on the next frame.
        self.banner_dirty = true;
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

        // Compose the banner only when an input changed (prefix text, HUD
        // toggle, geometry, or a new latency sample) — the scalar checks come
        // first so unchanged frames allocate nothing.
        let lat_n = self.latency.count();
        if self.banner_dirty || (self.latency_hud && lat_n != self.last_lat_n) {
            let stats = if !self.latency_hud {
                // Latency HUD off (config `latency_hud`): prefix only.
                self.stats_prefix.clone()
            } else {
                let lat = match self.latency.percentiles_cached() {
                    Some((p50, p99, _)) => {
                        format!("lat p50 {:.1}ms p99 {:.1}ms n={}", p50, p99, lat_n)
                    }
                    None => "lat p50 -ms p99 -ms n=0".to_string(),
                };
                if self.stats_prefix.is_empty() {
                    lat
                } else {
                    format!("{}    {}", self.stats_prefix, lat)
                }
            };
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
            self.last_lat_n = lat_n;
            self.banner_dirty = false;
        }

        // Geometry snapshot (copies) so the TextArea borrows below only touch
        // the buffer fields, keeping them disjoint from the &mut atlas/font.
        let pad = self.pad_px();
        let left = self.left_edge();
        let doc_top = self.doc_top();
        let line_px = self.line_px();
        let cell_w = self.cell_w();
        let text_left = self.text_left();
        let term_open = self.term_open;
        let term_top = self.term_top();
        let doc_bottom = self.doc_bottom();
        let w = self.surface_config.width as i32;
        let h = self.surface_config.height as i32;

        let mut areas: Vec<TextArea> = Vec::with_capacity(8);
        if self.sidebar_enabled && self.sidebar_tab_count > 0 {
            // Left file-tab list: one open editor tab per row (dynamic labels).
            areas.push(TextArea {
                buffer: &self.sidebar_labels_buffer,
                left: pad * 1.1,
                top: self.sidebar_top(),
                scale: 1.0,
                bounds: TextBounds {
                    left: 0,
                    top: 0,
                    right: self.sidebar_w() as i32,
                    bottom: h,
                },
                default_color: SIDEBAR_LABEL_COLOR,
                custom_glyphs: &[],
            });
        }
        areas.push(TextArea {
            buffer: &self.stats_buffer,
            left,
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
        if !self.tab_layout.is_empty() {
            let ts_top = pad + line_px * STATS_GAP;
            areas.push(TextArea {
                buffer: &self.tabstrip_buffer,
                left: left + pad * 0.5,
                top: ts_top + line_px * 0.25,
                scale: 1.0,
                bounds: TextBounds {
                    left: left as i32,
                    top: ts_top as i32,
                    right: w,
                    bottom: h,
                },
                default_color: TABSTRIP_TEXT_COLOR,
                custom_glyphs: &[],
            });
        }
        if self.gutter_enabled && self.gutter_digits > 0 {
            areas.push(TextArea {
                buffer: &self.gutter_buffer,
                left,
                top: doc_top,
                scale: 1.0,
                bounds: TextBounds {
                    left: 0,
                    top: doc_top as i32,
                    right: text_left as i32,
                    bottom: h,
                },
                default_color: GUTTER_COLOR,
                custom_glyphs: &[],
            });
        }
        areas.push(TextArea {
            buffer: &self.doc_buffer,
            left: text_left,
            top: doc_top,
            scale: 1.0,
            bounds: TextBounds {
                left: text_left as i32,
                top: doc_top as i32,
                right: w,
                bottom: h,
            },
            default_color: Color::rgb(220, 220, 220),
            custom_glyphs: &[],
        });
        // Terminal panel grid (P3), clipped to the panel region below the
        // document. Drawn like the doc: under the modal dim when one is up.
        if term_open {
            areas.push(TextArea {
                buffer: &self.term_buffer,
                left,
                top: term_top + pad,
                scale: 1.0,
                bounds: TextBounds {
                    left: 0,
                    top: term_top as i32,
                    right: w,
                    bottom: h,
                },
                default_color: TERM_TEXT_COLOR,
                custom_glyphs: &[],
            });
        }
        // Hovered word recolored gold, drawn over the original glyphs at the
        // word's exact grid cell (monospace -> covers them precisely). Shaped
        // only when the word changes (set_hover_word); here it is just placed.
        if let Some((line, start_col)) = self.hover_word {
            if !self.overlay_active {
                let x = text_left + start_col as f32 * cell_w;
                let y = doc_top + line as f32 * line_px;
                if y < h as f32 {
                    areas.push(TextArea {
                        buffer: &self.hover_word_buffer,
                        left: x,
                        top: y,
                        scale: 1.0,
                        bounds: TextBounds {
                            left: text_left as i32,
                            top: doc_top as i32,
                            right: w,
                            bottom: h,
                        },
                        default_color: HOVER_WORD_COLOR,
                        custom_glyphs: &[],
                    });
                }
            }
        }
        if let Some((line, col)) = self.cursor {
            let x = text_left + col as f32 * cell_w;
            let y = doc_top + line as f32 * line_px;
            if y < h as f32 {
                areas.push(TextArea {
                    buffer: &self.cursor_buffer,
                    left: x,
                    top: y,
                    scale: 1.0,
                    bounds: TextBounds {
                        left: text_left as i32,
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

        // Modal overlay text prepared on its own renderer so its glyphs draw in
        // a pass after the dim quad (glyphon renders a renderer's areas in one
        // pass). Areas borrow the overlay buffers; geometry is snapshotted from
        // the immutable accessors first.
        if self.overlay_active {
            let ov_left = self.overlay_content_left();
            let ov_w = self.overlay_content_width();
            let ov_top = self.overlay_top();
            let ov_rows_top = self.overlay_rows_top();
            let right_x = ov_left + ov_w * self.overlay_split_frac;
            let hint_y = self.surface_config.height as f32 - self.pad_px() - line_px;
            let full = TextBounds {
                left: 0,
                top: 0,
                right: w,
                bottom: h,
            };
            let mut ov_areas: Vec<TextArea> = Vec::with_capacity(5);
            if self.overlay_has_title {
                ov_areas.push(TextArea {
                    buffer: &self.overlay_title,
                    left: ov_left,
                    top: ov_top,
                    scale: 1.0,
                    bounds: full,
                    default_color: OVERLAY_TITLE_COLOR,
                    custom_glyphs: &[],
                });
            }
            if self.overlay_has_input {
                ov_areas.push(TextArea {
                    buffer: &self.overlay_input,
                    left: ov_left,
                    top: ov_top,
                    scale: 1.0,
                    bounds: full,
                    default_color: OVERLAY_INPUT_COLOR,
                    custom_glyphs: &[],
                });
            }
            ov_areas.push(TextArea {
                buffer: &self.overlay_left,
                left: ov_left,
                top: ov_rows_top,
                scale: 1.0,
                bounds: full,
                default_color: self.overlay_left_color,
                custom_glyphs: &[],
            });
            ov_areas.push(TextArea {
                buffer: &self.overlay_right,
                left: right_x,
                top: ov_rows_top,
                scale: 1.0,
                bounds: full,
                default_color: self.overlay_right_color,
                custom_glyphs: &[],
            });
            if self.overlay_has_hint {
                ov_areas.push(TextArea {
                    buffer: &self.overlay_hint,
                    left: ov_left,
                    top: hint_y,
                    scale: 1.0,
                    bounds: full,
                    default_color: OVERLAY_HINT_COLOR,
                    custom_glyphs: &[],
                });
            }
            if let Err(err) = self.overlay_text_renderer.prepare(
                &self.device,
                &self.queue,
                &mut self.font_system,
                &mut self.atlas,
                &self.viewport,
                ov_areas,
                &mut self.swash_cache,
            ) {
                eprintln!("umber-ui: overlay prepare failed: {err:?}");
                self.window.request_redraw();
                return false;
            }
        }

        // Build overlay quads into the reused staging buffer (no per-frame heap
        // allocation): selection highlights first (vertices drawn BEHIND the
        // text), then the scrollbar track + thumb (drawn OVER the text). Both
        // ride the same vertex buffer and pipeline, split by vertex range.
        let fw = self.surface_config.width as f32;
        let fh = self.surface_config.height as f32;
        self.quad_bytes.clear();

        // Selection: one quad per visible highlighted line, using the same
        // `col * cell_w` arithmetic as the caret. Clamped to `QUAD_MAX - 7` so
        // the scrollbar (2), separator rule + hover segment (2), and terminal
        // background + border + cursor (3) always fit the vertex buffer.
        // the scrollbar's two quads plus the gutter separator rule and its
        // hovered-line segment always fit the vertex buffer.
        let sel_right_edge = fw - pad;
        let mut sel_verts: u32 = 0;
        if !self.overlay_active {
            for span in self.selection.iter().take(QUAD_MAX - 7) {
                let y = doc_top + span.line as f32 * line_px;
                if y >= fh {
                    continue;
                }
                let x = text_left + span.start_col as f32 * cell_w;
                let right = match span.end_col {
                    Some(c) => text_left + c as f32 * cell_w,
                    None => sel_right_edge,
                };
                let width = (right - x).max(0.0);
                if width > 0.0 {
                    sel_verts += push_quad(
                        &mut self.quad_bytes,
                        fw,
                        fh,
                        x,
                        y,
                        width,
                        line_px,
                        SELECTION_COLOR,
                    );
                }
            }
        }

        // Scrollbar track + thumb, appended after the selection vertices (also
        // suppressed while a modal overlay is up).
        let geom = if self.overlay_active {
            None
        } else {
            self.scrollbar
                .and_then(|(first, total)| self.scrollbar_geom(first, total))
        };
        let mut bar_verts: u32 = 0;
        if let Some(g) = geom {
            bar_verts += push_quad(
                &mut self.quad_bytes,
                fw,
                fh,
                g.track_x,
                g.track_top,
                g.track_w,
                g.track_h,
                TRACK_COLOR,
            );
            bar_verts += push_quad(
                &mut self.quad_bytes,
                fw,
                fh,
                g.track_x,
                g.thumb_top,
                g.track_w,
                g.thumb_h,
                THUMB_COLOR,
            );
        }

        // Gutter separator rule + hovered-line segment (editor only -- the modal
        // dim would cover them). Thin vertical quad centered in the gutter gap,
        // full document height; the hover segment repaints one line of it in
        // gold so the rule always shows the pointer's line. Both count toward
        // QUAD_MAX (selection is capped at QUAD_MAX-4 to reserve these).
        let mut sep_verts: u32 = 0;
        if !self.overlay_active && self.gutter_enabled && self.gutter_digits > 0 {
            let s = self.scale_factor as f32;
            let sep_w = (SEPARATOR_W * s).max(1.0);
            let sep_x =
                self.left_edge() + self.gutter_text_w() + GUTTER_GAP * s * 0.5 - sep_w * 0.5;
            let sep_h = (doc_bottom - doc_top).max(0.0);
            if sep_h > 0.0 {
                sep_verts += push_quad(
                    &mut self.quad_bytes,
                    fw,
                    fh,
                    sep_x,
                    doc_top,
                    sep_w,
                    sep_h,
                    SEPARATOR_COLOR,
                );
            }
            if let Some(line) = self.hover_line {
                let y = doc_top + line as f32 * line_px;
                if y < fh {
                    sep_verts += push_quad(
                        &mut self.quad_bytes,
                        fw,
                        fh,
                        sep_x,
                        y,
                        sep_w,
                        line_px,
                        SEPARATOR_HOVER_COLOR,
                    );
                }
            }
        }

        // Git gutter markers: a thin colored bar at the far-left edge for each
        // changed visible line (own buffer, own draw — never touches the
        // QUAD_MAX overlay budget). Editor view only.
        self.git_bytes.clear();
        let mut git_verts: u32 = 0;
        if !self.overlay_active && self.gutter_enabled {
            let s = self.scale_factor as f32;
            let mark_w = (3.0 * s).max(1.0);
            let mark_x = self.sidebar_w() + self.pad_px() * 0.4;
            for (line, color) in self.gutter_marks.iter().take(GIT_MARK_MAX) {
                let y = doc_top + *line as f32 * line_px;
                if y >= doc_bottom {
                    continue;
                }
                git_verts += push_quad(
                    &mut self.git_bytes,
                    fw,
                    fh,
                    mark_x,
                    y,
                    mark_w,
                    line_px,
                    *color,
                );
            }
        }

        // Terminal panel border + cursor cell, appended after the separator
        // range. The border doubles as the focus cue: rust accent while the
        // terminal owns the keyboard, muted grey otherwise.
        let mut term_verts: u32 = 0;
        if term_open {
            let s = self.scale_factor as f32;
            let border_h = (1.0 * s).max(1.0);
            // Border + cursor only — the panel BACKGROUND is drawn in the
            // pre-text chrome range (drawing it here, after the text pass,
            // painted over the grid glyphs: the missing-terminal-text bug).
            term_verts += push_quad(
                &mut self.quad_bytes,
                fw,
                fh,
                0.0,
                term_top,
                fw,
                border_h,
                if self.term_focused {
                    TERM_BORDER_FOCUS_COLOR
                } else {
                    TERM_BORDER_COLOR
                },
            );
            if !self.overlay_active {
                if let Some((row, col)) = self.term_cursor {
                    // Grid origin = the content left edge (past the sidebar).
                    let x = left + col as f32 * cell_w;
                    let y = term_top + pad + row as f32 * line_px;
                    if y + line_px <= fh {
                        term_verts += push_quad(
                            &mut self.quad_bytes,
                            fw,
                            fh,
                            x,
                            y,
                            cell_w,
                            line_px,
                            TERM_CURSOR_COLOR,
                        );
                    }
                }
            }
        }

        // Modal overlay quads (dim + input box + selected-row highlight),
        // appended after the scrollbar range; drawn over the editor and under
        // the overlay text.
        let mut ov_verts: u32 = 0;
        if self.overlay_active {
            let ov_left = self.overlay_content_left();
            let ov_w = self.overlay_content_width();
            let ov_top = self.overlay_top();
            let ov_rows_top = self.overlay_rows_top();
            ov_verts += push_quad(
                &mut self.quad_bytes,
                fw,
                fh,
                0.0,
                0.0,
                fw,
                fh,
                OVERLAY_DIM_COLOR,
            );
            // Opaque content panel behind the whole page (title/input, rows,
            // hint) so overlay text never fights the editor text behind it.
            let panel_x = (ov_left - pad * 1.5).max(0.0);
            let panel_y = (ov_top - line_px * 0.5).max(0.0);
            let panel_w = (ov_w + pad * 3.0).min(fw - panel_x);
            let panel_h = (fh - panel_y - pad).max(0.0);
            ov_verts += push_quad(
                &mut self.quad_bytes,
                fw,
                fh,
                panel_x,
                panel_y,
                panel_w,
                panel_h,
                OVERLAY_PANEL_COLOR,
            );
            if self.overlay_has_input {
                ov_verts += push_quad(
                    &mut self.quad_bytes,
                    fw,
                    fh,
                    ov_left - pad,
                    ov_top - line_px * 0.15,
                    ov_w + pad * 2.0,
                    line_px * 1.3,
                    OVERLAY_BOX_COLOR,
                );
            }
            if let Some(sel) = self.overlay_selected {
                if sel < self.overlay_row_count {
                    let hy = ov_rows_top + sel as f32 * line_px;
                    if hy < fh {
                        ov_verts += push_quad(
                            &mut self.quad_bytes,
                            fw,
                            fh,
                            ov_left - pad * 0.5,
                            hy,
                            ov_w + pad,
                            line_px,
                            OVERLAY_HL_COLOR,
                        );
                    }
                }
            }
        }

        if sel_verts + bar_verts + sep_verts + term_verts + ov_verts > 0 {
            self.queue
                .write_buffer(&self.quad_vbuf, 0, &self.quad_bytes);
        }
        if git_verts > 0 {
            self.queue.write_buffer(&self.git_vbuf, 0, &self.git_bytes);
        }
        // Left tab-bar background quad (own buffer, drawn behind the glyphs).
        let mut sidebar_verts: u32 = 0;
        self.sidebar_bytes.clear();
        if self.sidebar_enabled {
            let sw = self.sidebar_w();
            let sb_top = self.sidebar_top();
            let pitch = self.line_px() * SIDEBAR_TAB_PITCH;
            let s = self.scale_factor as f32;
            let hover = self.sidebar_hover;
            let active = self.sidebar_active;
            sidebar_verts += push_quad(
                &mut self.sidebar_bytes,
                fw,
                fh,
                0.0,
                0.0,
                sw,
                fh,
                SIDEBAR_BG_COLOR,
            );
            if let Some(hrow) = hover {
                sidebar_verts += push_quad(
                    &mut self.sidebar_bytes,
                    fw,
                    fh,
                    0.0,
                    sb_top + hrow as f32 * pitch,
                    sw,
                    pitch,
                    SIDEBAR_HOVER_COLOR,
                );
            }
            if let Some(arow) = active {
                let ay = sb_top + arow as f32 * pitch;
                sidebar_verts += push_quad(
                    &mut self.sidebar_bytes,
                    fw,
                    fh,
                    0.0,
                    ay,
                    sw,
                    pitch,
                    SIDEBAR_ACTIVE_BG_COLOR,
                );
                sidebar_verts += push_quad(
                    &mut self.sidebar_bytes,
                    fw,
                    fh,
                    0.0,
                    ay,
                    (3.0 * s).max(2.0),
                    pitch,
                    SIDEBAR_ACTIVE_COLOR,
                );
            }
        }
        // Open-document tab strip background + active-tab tint (shares the
        // sidebar vertex buffer; drawn before the text pass).
        if !self.tab_layout.is_empty() {
            let ts_top = self.tabstrip_top();
            let ts_h = self.tabstrip_h();
            let le = self.left_edge();
            let origin = le + self.pad_px() * 0.5;
            let cw = self.cell_w();
            let (astart, aend) = self
                .tab_layout
                .get(self.tab_active)
                .copied()
                .unwrap_or((0, 0));
            sidebar_verts += push_quad(
                &mut self.sidebar_bytes,
                fw,
                fh,
                le,
                ts_top,
                (fw - le).max(0.0),
                ts_h,
                TABSTRIP_BG_COLOR,
            );
            if aend > astart {
                let ax = origin + astart as f32 * cw;
                let aw = (aend - astart) as f32 * cw;
                sidebar_verts += push_quad(
                    &mut self.sidebar_bytes,
                    fw,
                    fh,
                    ax - cw * 0.5,
                    ts_top,
                    aw + cw,
                    ts_h,
                    TABSTRIP_ACTIVE_COLOR,
                );
            }
        }
        // Seam lines: sidebar|content vertical, and horizontal under the
        // activity strip — subtle 1px region separators.
        {
            let s = (1.0 * self.scale_factor as f32).max(1.0);
            let sw_edge = self.sidebar_w();
            if sw_edge > 0.0 {
                sidebar_verts += push_quad(
                    &mut self.sidebar_bytes,
                    fw,
                    fh,
                    sw_edge - s,
                    0.0,
                    s,
                    fh,
                    SEAM_COLOR,
                );
            }
            let seam_y = self.doc_top() - s;
            sidebar_verts += push_quad(
                &mut self.sidebar_bytes,
                fw,
                fh,
                sw_edge,
                seam_y,
                (fw - sw_edge).max(0.0),
                s,
                SEAM_COLOR,
            );
        }
        // Terminal panel background — BEHIND the text pass so the grid
        // glyphs render on top of it.
        if self.term_open {
            let t_top = self.term_top();
            sidebar_verts += push_quad(
                &mut self.sidebar_bytes,
                fw,
                fh,
                0.0,
                t_top,
                fw,
                (fh - t_top).max(0.0),
                TERM_BG_COLOR,
            );
        }
        if sidebar_verts > 0 {
            self.queue
                .write_buffer(&self.sidebar_vbuf, 0, &self.sidebar_bytes);
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

            // Selection highlights composited BEHIND the text (drawn first so
            // the glyphs render over them).
            if sel_verts > 0 {
                pass.set_pipeline(&self.quad_pipeline);
                pass.set_vertex_buffer(0, self.quad_vbuf.slice(..));
                pass.draw(0..sel_verts, 0..1);
            }

            // Sidebar background behind everything on the left strip; the tab
            // glyphs (in the text pass just below) draw over it.
            if sidebar_verts > 0 {
                pass.set_pipeline(&self.quad_pipeline);
                pass.set_vertex_buffer(0, self.sidebar_vbuf.slice(..));
                pass.draw(0..sidebar_verts, 0..1);
            }

            self.text_renderer
                .render(&self.atlas, &self.viewport, &mut pass)
                .expect("render text");

            // Overlay scrollbar (track + thumb) composited OVER the text, from
            // the vertex range just past the selection quads.
            if bar_verts > 0 {
                pass.set_pipeline(&self.quad_pipeline);
                pass.set_vertex_buffer(0, self.quad_vbuf.slice(..));
                pass.draw(sel_verts..sel_verts + bar_verts, 0..1);
            }

            // Gutter separator rule + hovered-line segment, composited OVER the
            // text from the range just past the scrollbar quads.
            if sep_verts > 0 {
                let start = sel_verts + bar_verts;
                pass.set_pipeline(&self.quad_pipeline);
                pass.set_vertex_buffer(0, self.quad_vbuf.slice(..));
                pass.draw(start..start + sep_verts, 0..1);
            }

            // Terminal border + cursor, from the range past the separator.
            if term_verts > 0 {
                let start = sel_verts + bar_verts + sep_verts;
                pass.set_pipeline(&self.quad_pipeline);
                pass.set_vertex_buffer(0, self.quad_vbuf.slice(..));
                pass.draw(start..start + term_verts, 0..1);
            }

            // Git gutter markers from their dedicated buffer.
            if git_verts > 0 {
                pass.set_pipeline(&self.quad_pipeline);
                pass.set_vertex_buffer(0, self.git_vbuf.slice(..));
                pass.draw(0..git_verts, 0..1);
            }

            // Modal overlay: dim + box + highlight quads, then the overlay text
            // in its own renderer so it sits above the dim.
            if ov_verts > 0 {
                let start = sel_verts + bar_verts + sep_verts + term_verts;
                pass.set_pipeline(&self.quad_pipeline);
                pass.set_vertex_buffer(0, self.quad_vbuf.slice(..));
                pass.draw(start..start + ov_verts, 0..1);
            }
            if self.overlay_active {
                self.overlay_text_renderer
                    .render(&self.atlas, &self.viewport, &mut pass)
                    .expect("render overlay text");
            }
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
