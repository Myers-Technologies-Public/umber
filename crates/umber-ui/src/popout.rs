//! Popped-out terminal windows: a **separate OS window** with its own wgpu
//! context, a glyphon text pass for the terminal grid, and a small
//! self-contained rounded-rect quad pipeline for the custom chrome (a draggable
//! title island + a min/max/close control island), mirroring the main window.
//!
//! The `umber` bin owns the moved PTY session, feeds snapshots, and routes
//! pointer/keyboard events (drag, button clicks) via the methods here.

use std::sync::Arc;
use std::time::Instant;

use cosmic_text::{
    Attrs, Buffer, Color, Family, FontSystem, Metrics, Shaping, Style, SwashCache, Weight, Wrap,
};

use crate::PaneDividerSpec;
use glyphon::{Cache, Resolution, TextArea, TextAtlas, TextBounds, TextRenderer, Viewport};
use winit::event_loop::ActiveEventLoop;
use winit::window::{ResizeDirection, Window, WindowId};

use wgpu::{
    CommandEncoderDescriptor, CompositeAlphaMode, DeviceDescriptor, Instance, InstanceDescriptor,
    LoadOp, MultisampleState, Operations, PresentMode, RenderPassColorAttachment,
    RenderPassDescriptor, RequestAdapterOptions, SurfaceColorSpace, SurfaceConfiguration,
    TextureFormat, TextureUsages, TextureViewDescriptor,
};

const BASE_FONT: f32 = 14.0;
const BASE_LINE: f32 = 20.0;
const MONO_ADVANCE_RATIO: f32 = 0.6;
const PAD: f32 = 8.0;
const SHELL_RADIUS: f32 = 12.0;
/// Editor palette, mirrored from the main renderer so a pop-out reads as the
/// same app: warm-dark island fills + a warm border + the rust focus ring,
/// floating over a lighter warm-gray shell.
const PANEL_BORDER_COLOR: [f32; 4] = [0.095, 0.070, 0.050, 1.0];
const EDITOR_PANEL_COLOR: [f32; 4] = [0.014, 0.012, 0.010, 1.0];
const TERM_BG_COLOR: [f32; 4] = [0.010, 0.009, 0.008, 1.0];
const PANE_FOCUS_BORDER_COLOR: [f32; 4] = [0.30, 0.135, 0.070, 1.0];
const SELECTION_COLOR: [f32; 4] = [0.86, 0.47, 0.30, 0.6];
const TERM_CURSOR_COLOR: [f32; 4] = [0.757, 0.373, 0.235, 0.45];

/// 11 floats/vertex: clip pos (2), rgba (4), local px (2), half px (2), radius (1).
const FLOATS_PER_VERT: usize = 11;

const QUAD_SHADER: &str = r#"
struct VsIn {
  @location(0) pos: vec2<f32>,
  @location(1) color: vec4<f32>,
  @location(2) local: vec2<f32>,
  @location(3) half: vec2<f32>,
  @location(4) radius: f32,
};
struct VsOut {
  @builtin(position) clip: vec4<f32>,
  @location(0) color: vec4<f32>,
  @location(1) local: vec2<f32>,
  @location(2) half: vec2<f32>,
  @location(3) radius: f32,
};
@vertex
fn vs_main(in: VsIn) -> VsOut {
  var out: VsOut;
  out.clip = vec4<f32>(in.pos, 0.0, 1.0);
  out.color = in.color;
  out.local = in.local;
  out.half = in.half;
  out.radius = in.radius;
  return out;
}
@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
  let q = abs(in.local) - (in.half - vec2<f32>(in.radius, in.radius));
  let d = length(max(q, vec2<f32>(0.0, 0.0))) + min(max(q.x, q.y), 0.0) - in.radius;
  let aa = clamp(0.5 - d, 0.0, 1.0);
  return vec4<f32>(in.color.rgb, in.color.a * aa);
}
"#;

/// Append one rounded rect (6 verts) to `out` (LE f32 bytes). Returns vert count.
fn push_rquad(
    out: &mut Vec<u8>,
    sw: f32,
    sh: f32,
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    color: [f32; 4],
    radius: f32,
) -> u32 {
    let cx = x + w * 0.5;
    let cy = y + h * 0.5;
    let corners = [
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (0.0, 0.0),
        (1.0, 1.0),
        (0.0, 1.0),
    ];
    for (ux, uy) in corners {
        let px = x + ux * w;
        let py = y + uy * h;
        let clip_x = px / sw * 2.0 - 1.0;
        let clip_y = 1.0 - py / sh * 2.0;
        let v = [
            clip_x,
            clip_y,
            color[0],
            color[1],
            color[2],
            color[3],
            px - cx,
            py - cy,
            w * 0.5,
            h * 0.5,
            radius,
        ];
        for f in v {
            out.extend_from_slice(&f.to_le_bytes());
        }
    }
    6
}

/// Chrome geometry in physical px.
struct Chrome {
    drag: (f32, f32, f32, f32),
    control: (f32, f32, f32, f32),
    buttons: [(f32, f32, f32, f32); 3],
}

/// One tiled terminal inside a popup (mirrors renderer.rs' TermPaneView —
/// kept here so the popup can host real splits reusing the data shape the
/// in-app pane system already uses).
pub struct PopoutTile {
    pub id: u64,
    pub rect: [f32; 4],
    buffer: Buffer,
    cursor: Option<(usize, usize)>,
    pub focused: bool,
    display_offset: usize,
    sel: Option<((usize, usize), (usize, usize))>,
    focus_anim: f32,
}

pub struct PopoutWindow {
    window: Arc<Window>,
    instance: Instance,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: SurfaceConfiguration,
    // Text pass.
    font_system: FontSystem,
    swash_cache: SwashCache,
    atlas: TextAtlas,
    viewport: Viewport,
    text_renderer: TextRenderer,
    buffer: Buffer,
    /// Grid cell coords of the text cursor, like the main renderer's TermPane.
    cursor: Option<(usize, usize)>,
    /// Multi-tile terminal panes, mirroring the in-app renderer. When non-empty
    /// the popup draws / hit-tests these instead of the single legacy buffer;
    /// when empty the legacy single-buffer path runs.
    tiles: Vec<PopoutTile>,
    pane_divs: Vec<PaneDividerSpec>,
    /// Id of the focused tile (only meaningful when `tiles` is non-empty).
    focused_tile: Option<u64>,
    // Quad pass (chrome).
    quad_pipeline: wgpu::RenderPipeline,
    quad_vbuf: wgpu::Buffer,
    // Metrics + state.
    scale: f32,
    cell_w: f32,
    line_px: f32,
    pad: f32,
    hover: Option<usize>,
    /// Drag-selection over the grid: `(anchor (row,col), head (row,col))`.
    sel: Option<((usize, usize), (usize, usize))>,
    selecting: bool,
    /// Scroll-back input overlay: a pinned one-line strip at the island bottom
    /// showing the being-typed prompt row while the user is scrolled up. None
    // = bottom-visible, no strip. Shaped lazily on set_overlay_text.
    overlay: Option<Buffer>,
    /// Per-button hover animation progress (0=idle, 1=hovered).
    win_btn_anim: [f32; 3],
    anim_prev: Instant,
    // Pointer context menu (mostly a port of the main renderer's menu cards).
    context_buffer: Buffer,
    context_active: bool,
    context_x: f32,
    context_y: f32,
    context_width: f32,
    context_rows: usize,
    context_hover: Option<usize>,
    context_separators: Vec<usize>,
}

impl PopoutWindow {
    pub fn new(window: Arc<Window>, event_loop: &ActiveEventLoop) -> Self {
        pollster::block_on(Self::new_async(window, event_loop))
    }

    async fn new_async(window: Arc<Window>, event_loop: &ActiveEventLoop) -> Self {
        let size = window.inner_size();
        let scale = window.scale_factor() as f32;

        let instance = Instance::new(InstanceDescriptor::new_with_display_handle(Box::new(
            event_loop.owned_display_handle(),
        )));
        let adapter = instance
            .request_adapter(&RequestAdapterOptions::default())
            .await
            .expect("popout: request a wgpu adapter");
        let (device, queue) = adapter
            .request_device(&DeviceDescriptor::default())
            .await
            .expect("popout: request a wgpu device");
        let surface = instance
            .create_surface(window.clone())
            .expect("popout: create a wgpu surface");
        let format = TextureFormat::Bgra8UnormSrgb;
        // Low-latency present mode so resize tracks the pointer (not vsync).
        let present_mode = {
            let modes = surface.get_capabilities(&adapter).present_modes;
            if modes.contains(&PresentMode::Mailbox) {
                PresentMode::Mailbox
            } else if modes.contains(&PresentMode::Immediate) {
                PresentMode::Immediate
            } else {
                PresentMode::Fifo
            }
        };
        let config = SurfaceConfiguration {
            usage: TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode,
            alpha_mode: CompositeAlphaMode::Opaque,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
            color_space: SurfaceColorSpace::Auto,
        };
        surface.configure(&device, &config);

        let mut font_system = FontSystem::new();
        let swash_cache = SwashCache::new();
        let cache = Cache::new(&device);
        let viewport = Viewport::new(&device, &cache);
        let mut atlas = TextAtlas::new(&device, &queue, &cache, format);
        let text_renderer =
            TextRenderer::new(&mut atlas, &device, MultisampleState::default(), None);

        let font_px = BASE_FONT * scale;
        let line_px = BASE_LINE * scale;
        let cell_w = font_px * MONO_ADVANCE_RATIO;
        let metrics = Metrics::new(font_px, line_px);
        let mut buffer = Buffer::new(&mut font_system, metrics);
        buffer.set_wrap(Wrap::None);

        let quad_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("popout quad shader"),
            source: wgpu::ShaderSource::Wgsl(QUAD_SHADER.into()),
        });
        let quad_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("popout quad pipeline"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &quad_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: (FLOATS_PER_VERT * 4) as wgpu::BufferAddress,
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
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x2,
                            offset: 24,
                            shader_location: 2,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x2,
                            offset: 32,
                            shader_location: 3,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32,
                            offset: 40,
                            shader_location: 4,
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
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        let quad_vbuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("popout quad vertices"),
            size: (128 * FLOATS_PER_VERT * 4) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Construct buffers that need `font_system` *before* it is moved into
        // the struct (it is borrowed again post-move otherwise).
        let mut context_buffer = Buffer::new(&mut font_system, Metrics::new(BASE_FONT, BASE_LINE));
        context_buffer.set_wrap(Wrap::None);

        let mut this = Self {
            window,
            instance,
            surface,
            device,
            queue,
            config,
            font_system,
            swash_cache,
            atlas,
            viewport,
            text_renderer,
            buffer,
            cursor: None,
            tiles: Vec::new(),
            pane_divs: Vec::new(),
            focused_tile: None,
            quad_pipeline,
            quad_vbuf,
            scale,
            cell_w,
            line_px,
            pad: PAD * scale,
            hover: None,
            sel: None,
            selecting: false,
            overlay: None,
            context_buffer,
            context_active: false,
            context_x: 0.0,
            context_y: 0.0,
            context_width: 0.0,
            context_rows: 0,
            context_hover: None,
            context_separators: Vec::new(),
            win_btn_anim: [0.0; 3],
            anim_prev: Instant::now(),
        };
        this.reshape_buffer();
        this
    }

    fn chrome(&self) -> Chrome {
        let s = self.scale;
        let gap = self.pad;
        let bpad = 7.0 * s;
        let bspace = 7.0 * s;
        let d = self.line_px * 0.5;
        let cw = 3.0 * d + 2.0 * bspace + 2.0 * bpad;
        let bar_h = d + 2.0 * bpad;
        let width = self.config.width as f32;
        // Buttons island: cut into the terminal island's top-right corner.
        let cx = width - gap - self.pad - cw;
        let cy = gap + self.pad;
        let control = (cx, cy, cw, bar_h);
        let by = cy + (bar_h - d) * 0.5;
        let b0 = cx + bpad;
        let b1 = b0 + d + bspace;
        let b2 = b1 + d + bspace;
        let buttons = [(b0, by, d, d), (b1, by, d, d), (b2, by, d, d)];
        // Drag handle: the top strip of the island, left of the buttons —
        // deliberately tall so it's easy to grab.
        let drag = (gap, gap, (cx - gap).max(1.0), bar_h + self.pad * 3.0);
        Chrome {
            drag,
            control,
            buttons,
        }
    }

    pub fn id(&self) -> WindowId {
        self.window.id()
    }

    pub fn request_redraw(&self) {
        self.window.request_redraw();
    }

    pub fn drag(&self) {
        let _ = self.window.drag_window();
    }

    pub fn set_minimized(&self) {
        self.window.set_minimized(true);
    }

    pub fn toggle_maximized(&self) {
        self.window.set_maximized(!self.window.is_maximized());
    }

    /// Which control button `(x, y)` hits (0=min, 1=max, 2=close).
    pub fn window_button_at(&self, x: f32, y: f32) -> Option<usize> {
        for (i, (bx, by, bw, bh)) in self.chrome().buttons.iter().enumerate() {
            if x >= *bx && x <= bx + bw && y >= *by && y <= by + bh {
                return Some(i);
            }
        }
        None
    }

    /// True when `(x, y)` is on the draggable title island.
    pub fn in_titlebar(&self, x: f32, y: f32) -> bool {
        let (dx, dy, dw, dh) = self.chrome().drag;
        x >= dx && x <= dx + dw && y >= dy && y <= dy + dh
    }

    /// Update the hovered button from a pointer position; true if it changed.
    pub fn set_hover(&mut self, x: f32, y: f32) -> bool {
        let h = self.window_button_at(x, y);
        if self.hover != h {
            self.hover = h;
            true
        } else {
            false
        }
    }

    pub fn cell_px(&self) -> (u16, u16) {
        (
            self.cell_w.round().max(1.0) as u16,
            self.line_px.round().max(1.0) as u16,
        )
    }

    /// The terminal's floating island rect (physical px): a rounded card inset
    /// one shell gap from the window edges, below the title bar — the same
    /// language as an editor pane tile.
    fn term_island(&self) -> (f32, f32, f32, f32) {
        let gap = self.pad;
        let w = (self.config.width as f32 - gap * 2.0).max(1.0);
        let h = (self.config.height as f32 - gap * 2.0).max(1.0);
        (gap, gap, w, h)
    }

    /// Map a pointer to a terminal `(row, col)` inside the island's text area.
    fn cell_at(&self, x: f32, y: f32) -> Option<(usize, usize)> {
        let (ix, iy, iw, ih) = self.term_island();
        if x < ix || x > ix + iw || y < iy || y > iy + ih {
            return None;
        }
        let (cols, rows) = self.grid();
        let col = (((x - (ix + self.pad)) / self.cell_w).floor().max(0.0) as usize)
            .min(cols.saturating_sub(1));
        let row = (((y - (iy + self.pad)) / self.line_px).floor().max(0.0) as usize)
            .min(rows.saturating_sub(1));
        Some((row, col))
    }

    /// Begin a drag-selection at `(x, y)` (a fresh press collapses any prior).
    pub fn begin_selection(&mut self, x: f32, y: f32) {
        self.sel = self.cell_at(x, y).map(|c| (c, c));
        self.selecting = self.sel.is_some();
        self.window.request_redraw();
    }

    /// Extend the active selection to `(x, y)`; true when it changed.
    pub fn extend_selection(&mut self, x: f32, y: f32) -> bool {
        if !self.selecting {
            return false;
        }
        if let (Some(cell), Some((anchor, head))) = (self.cell_at(x, y), self.sel) {
            if head != cell {
                self.sel = Some((anchor, cell));
                self.window.request_redraw();
                return true;
            }
        }
        false
    }

    /// End the drag; a plain click (no span) leaves no highlight.
    pub fn end_selection(&mut self) {
        self.selecting = false;
        if let Some((a, b)) = self.sel {
            if a == b {
                self.sel = None;
                self.window.request_redraw();
            }
        }
    }

    /// The active selection `(anchor, head)`, only when it spans >1 cell.
    pub fn selection(&self) -> Option<((usize, usize), (usize, usize))> {
        self.sel.filter(|(a, b)| a != b)
    }

    /// Drop any selection highlight.
    pub fn clear_selection(&mut self) {
        if self.sel.take().is_some() {
            self.window.request_redraw();
        }
    }

    /// The resize direction for a corner grab at `(x, y)`, or `None`.
    pub fn resize_dir_at(&self, x: f32, y: f32) -> Option<ResizeDirection> {
        let m = 14.0 * self.scale;
        let w = self.config.width as f32;
        let h = self.config.height as f32;
        let (l, r, t, b) = (x <= m, x >= w - m, y <= m, y >= h - m);
        Some(match (t, b, l, r) {
            (true, _, true, _) => ResizeDirection::NorthWest,
            (true, _, _, true) => ResizeDirection::NorthEast,
            (_, true, true, _) => ResizeDirection::SouthWest,
            (_, true, _, true) => ResizeDirection::SouthEast,
            (true, _, _, _) => ResizeDirection::North,
            (_, true, _, _) => ResizeDirection::South,
            (_, _, true, _) => ResizeDirection::West,
            (_, _, _, true) => ResizeDirection::East,
            _ => return None,
        })
    }

    /// Start an interactive resize from a corner (borderless window).
    pub fn start_resize(&self, dir: ResizeDirection) {
        let _ = self.window.drag_resize_window(dir);
    }

    /// Grid (cols, rows) that fit inside the terminal island (text inset one
    /// pad from the card edge).
    pub fn grid(&self) -> (usize, usize) {
        let (_, _, iw, ih) = self.term_island();
        let w = (iw - self.pad * 2.0).max(1.0);
        let h = (ih - self.pad * 2.0).max(1.0);
        let cols = (w / self.cell_w).floor().max(1.0) as usize;
        let rows = (h / self.line_px).floor().max(1.0) as usize;
        (cols, rows)
    }

    fn reshape_buffer(&mut self) {
        let (_, _, iw, ih) = self.term_island();
        let w = (iw - self.pad * 2.0).max(1.0);
        let h = (ih - self.pad * 2.0).max(1.0);
        self.buffer.set_size(Some(w), Some(h));
        self.buffer.shape_until_scroll(&mut self.font_system, false);
    }

    pub fn set_content(&mut self, text: &str) {
        self.buffer.set_text(
            text,
            &Attrs::new().family(Family::Monospace),
            Shaping::Advanced,
            None,
        );
        self.cursor = None;
        self.reshape_buffer();
    }

    /// Styled terminal content: rich text spans (ANSI fg + bold/italic) and the
    /// grid cursor, mirroring `Renderer::set_term_pane_content` so a pop-out
    /// reads identical to the in-app tile — colours, suggested-code styling,
    /// and a blinking-set cursor block — rather than plain monochrome text.
    /// Open the pointer context menu at `(x, y)` with the given row labels.
    pub fn set_context_menu(&mut self, x: f32, y: f32, labels: &[&str]) {
        let pad = 10.0 * self.scale;
        let row_h = self.line_px;
        let max_chars = labels.iter().map(|s| s.chars().count()).max().unwrap_or(1);
        let width = ((max_chars as f32 + 1.0) * self.cell_w + pad * 2.0).max(150.0 * self.scale);
        let height = labels.len() as f32 * row_h + pad;
        self.context_x = x.clamp(4.0 * self.scale, (self.config.width as f32 - width - 4.0 * self.scale).max(4.0 * self.scale));
        self.context_y = y.clamp(4.0 * self.scale, (self.config.height as f32 - height - 4.0 * self.scale).max(4.0 * self.scale));
        self.context_width = width;
        self.context_rows = labels.len();
        self.context_hover = None;
        self.context_separators.clear();
        self.context_active = !labels.is_empty();
        let text = labels.join("\n");
        self.context_buffer.set_text(&text, &Attrs::new().family(Family::Monospace), Shaping::Advanced, None);
        self.context_buffer.set_size(Some(width - pad * 2.0), Some(height));
        self.context_buffer.shape_until_scroll(&mut self.font_system, false);
        self.window.request_redraw();
    }

    pub fn clear_context_menu(&mut self) {
        if self.context_active {
            self.context_active = false;
            self.context_hover = None;
            self.window.request_redraw();
        }
    }

    /// Set the group-divider rows for the open menu (after a `set_context_menu`).
    pub fn set_context_separators(&mut self, seps: &[usize]) {
        self.context_separators.clear();
        self.context_separators.extend_from_slice(seps);
        self.window.request_redraw();
    }

    pub fn context_menu_active(&self) -> bool {
        self.context_active
    }

    pub fn context_menu_row_at(&self, x: f32, y: f32) -> Option<usize> {
        if !self.context_active {
            return None;
        }
        let pad_y = 5.0 * self.scale;
        let row_h = self.line_px;
        let height = self.context_rows as f32 * row_h + pad_y * 2.0;
        if x < self.context_x
            || x > self.context_x + self.context_width
            || y < self.context_y
            || y > self.context_y + height
        {
            return None;
        }
        let row = ((y - self.context_y - pad_y) / row_h).floor().max(0.0) as usize;
        (row < self.context_rows).then_some(row)
    }

    pub fn set_context_menu_hover(&mut self, hover: Option<usize>) -> bool {
        if self.context_hover == hover {
            return false;
        }
        self.context_hover = hover;
        true
    }

    /// Pinned input overlay: `Some(text)` shows it at the island bottom, None
    /// clears it. Useful while the user is scrolled away from the prompt.
    pub fn set_overlay_text(&mut self, text: Option<String>) {
        self.overlay = None;
        if let Some(t) = text {
            let metrics = Metrics::new(BASE_FONT, BASE_LINE);
            let mut buf = Buffer::new(&mut self.font_system, metrics);
            buf.set_wrap(Wrap::None);
            buf.set_text(&t, &Attrs::new().family(Family::Monospace), Shaping::Advanced, None);
            buf.set_size(Some(900.0), Some(self.line_px));
            buf.shape_until_scroll(&mut self.font_system, false);
            self.overlay = Some(buf);
        }
        self.window.request_redraw();
    }

    pub fn set_styled_content(
        &mut self,
        text: &str,
        cursor: Option<(usize, usize)>,
        spans: &[crate::renderer::TerminalTextSpan],
    ) {
        self.cursor = cursor;
        let default_attrs = Attrs::new().family(Family::Monospace);
        if spans.is_empty() {
            self.buffer.set_text(text, &default_attrs, Shaping::Advanced, None);
        } else {
            let mut rich: Vec<(&str, Attrs)> = Vec::with_capacity(spans.len() * 2 + 1);
            let mut pos = 0;
            for span in spans {
                if span.start > pos {
                    if let Some(segment) = text.get(pos..span.start) {
                        rich.push((segment, default_attrs.clone()));
                    }
                }
                if let Some(segment) = text.get(span.start..span.end) {
                    let mut attrs = default_attrs
                        .clone()
                        .color(Color::rgb(span.rgb[0], span.rgb[1], span.rgb[2]));
                    if span.bold {
                        attrs = attrs.weight(Weight::BOLD);
                    }
                    if span.italic {
                        attrs = attrs.style(Style::Italic);
                    }
                    rich.push((segment, attrs));
                }
                pos = span.end;
            }
            if let Some(segment) = text.get(pos..) {
                rich.push((segment, default_attrs.clone()));
            }
            self.buffer
                .set_rich_text(rich, &default_attrs, Shaping::Advanced, None);
        }
        self.reshape_buffer();
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.config.width = width.max(1);
        self.config.height = height.max(1);
        self.surface.configure(&self.device, &self.config);
        self.reshape_buffer();
        self.window.request_redraw();
    }

    pub fn render(&mut self) {
        self.viewport.update(
            &self.queue,
            Resolution {
                width: self.config.width,
                height: self.config.height,
            },
        );

        // Chrome quads: title island + control island + buttons.
        let sw = self.config.width as f32;
        let sh = self.config.height as f32;
        let ch = self.chrome();
        let s = self.scale;
        let border = s.max(1.0);
        let radius = SHELL_RADIUS * s;
        let panel_border = PANEL_BORDER_COLOR;
        let panel_fill = EDITOR_PANEL_COLOR;
        let btn_colors = [
            [0.84, 0.70, 0.35, 1.0],
            [0.49, 0.70, 0.49, 1.0],
            [0.85, 0.43, 0.28, 1.0],
        ];
        let btn_hover = [
            [0.96, 0.82, 0.47, 1.0],
            [0.62, 0.82, 0.62, 1.0],
            [0.94, 0.55, 0.39, 1.0],
        ];
        let mut bytes: Vec<u8> = Vec::new();
        let mut verts: u32 = 0;
        let panel = |bytes: &mut Vec<u8>, r: (f32, f32, f32, f32)| -> u32 {
            let mut n = push_rquad(bytes, sw, sh, r.0, r.1, r.2, r.3, panel_border, radius);
            n += push_rquad(
                bytes,
                sw,
                sh,
                r.0 + border,
                r.1 + border,
                (r.2 - border * 2.0).max(1.0),
                (r.3 - border * 2.0).max(1.0),
                panel_fill,
                (radius - border).max(1.0),
            );
            n
        };
        // Terminal island fills the window (rust focus ring + dark fill), like
        // an editor pane tile — the pop-out is essentially one tile.
        let (ix, iy, iw, ih) = self.term_island();
        verts += push_rquad(&mut bytes, sw, sh, ix, iy, iw, ih, PANE_FOCUS_BORDER_COLOR, radius);
        verts += push_rquad(
            &mut bytes,
            sw,
            sh,
            ix + border,
            iy + border,
            (iw - border * 2.0).max(1.0),
            (ih - border * 2.0).max(1.0),
            TERM_BG_COLOR,
            (radius - border).max(1.0),
        );
        // Drag-selection highlight (behind text, over the island fill).
        if let Some((a, b)) = self.selection() {
            let (start, end) = if (a.0, a.1) <= (b.0, b.1) { (a, b) } else { (b, a) };
            let (cols, _) = self.grid();
            let tx = ix + self.pad;
            let ty = iy + self.pad;
            for row in start.0..=end.0 {
                let c0 = if row == start.0 { start.1 } else { 0 };
                let c1 = if row == end.0 { end.1 } else { cols.saturating_sub(1) };
                let x0 = tx + c0 as f32 * self.cell_w;
                let y0 = ty + row as f32 * self.line_px;
                let wq = (((c1 + 1).saturating_sub(c0)) as f32 * self.cell_w)
                    .min((ix + iw - self.pad - x0).max(0.0));
                if wq > 0.0 && y0 + self.line_px <= iy + ih - self.pad {
                    verts +=
                        push_rquad(&mut bytes, sw, sh, x0, y0, wq, self.line_px, SELECTION_COLOR, 0.0);
                }
            }
        }
        // Text cursor block, same rust accent the main pane uses.
        if let Some((row, col)) = self.cursor {
            let tx = ix + self.pad;
            let ty = iy + self.pad;
            let cx0 = tx + col as f32 * self.cell_w;
            let cy0 = ty + row as f32 * self.line_px;
            if cx0 + self.cell_w <= ix + iw - self.pad
                && cy0 + self.line_px <= iy + ih - self.pad
            {
                verts += push_rquad(
                    &mut bytes,
                    sw,
                    sh,
                    cx0,
                    cy0,
                    self.cell_w,
                    self.line_px,
                    TERM_CURSOR_COLOR,
                    2.0 * s,
                );
            }
        }
        // Everything above draws UNDER the terminal text; the control island
        // and buttons below draw OVER it, so the grid is truly cut out beneath
        // them (a bordered island, no glyphs bleeding through).
        // Pointer context-menu card + scroll-back overlay pill, pushed BEFORE
        // the under/post split so they draw BEHIND their labels (label goes
        // via the text pass that runs after under-verts and before post-quad).
        let menu_area_desc: Option<(f32, f32, f32, usize)> = if self.context_active {
            let pad_y = 5.0 * s;
            let row_h = self.line_px;
            let menu_h = self.context_rows as f32 * row_h + pad_y * 2.0;
            verts += push_rquad(&mut bytes, sw, sh,
                self.context_x, self.context_y,
                self.context_width, menu_h, PANEL_BORDER_COLOR, 8.0 * s);
            verts += push_rquad(&mut bytes, sw, sh,
                self.context_x + s, self.context_y + s,
                (self.context_width - 2.0 * s).max(1.0),
                (menu_h - 2.0 * s).max(1.0), EDITOR_PANEL_COLOR, 7.0 * s);
            if let Some(row) = self.context_hover {
                verts += push_rquad(&mut bytes, sw, sh,
                    self.context_x + 4.0 * s,
                    self.context_y + pad_y + row as f32 * row_h,
                    (self.context_width - 8.0 * s).max(1.0),
                    row_h, SELECTION_COLOR, 5.0 * s);
            }
            let seps = self.context_separators.clone();
            for sep in seps {
                let ly = self.context_y + pad_y + (sep + 1) as f32 * row_h;
                verts += push_rquad(&mut bytes, sw, sh,
                    self.context_x + 6.0 * s, ly,
                    (self.context_width - 12.0 * s).max(1.0),
                    (1.0 * s).max(1.0), PANEL_BORDER_COLOR, 0.0);
            }
            Some((self.context_x + 10.0 * s, self.context_y + 5.0 * s, self.context_width, self.context_rows))
        } else {
            None
        };
        let mut overlay_top: Option<f32> = None;
        if let Some(buf) = &self.overlay {
            let n = buf.layout_runs().count().max(1) as f32;
            let strip_h = (n + 0.6) * self.line_px;
            let py = iy + ih - strip_h - self.pad;
            verts += push_rquad(&mut bytes, sw, sh,
                ix + self.pad * 0.9, py,
                (iw - self.pad * 1.8).max(1.0), strip_h,
                PANEL_BORDER_COLOR, 6.0 * s);
            verts += push_rquad(&mut bytes, sw, sh,
                ix + self.pad * 0.9 + s, py + s,
                (iw - self.pad * 1.8 - 2.0 * s).max(1.0),
                (strip_h - 2.0 * s).max(1.0),
                EDITOR_PANEL_COLOR, 5.0 * s);
            overlay_top = Some(py);
        }
        let under_verts = verts;
        // Buttons island: a small rounded panel cut into the top-right corner.
        verts += panel(&mut bytes, ch.control);
        {
            let now = Instant::now();
            let dt = now.duration_since(self.anim_prev).as_secs_f32().min(0.033);
            self.anim_prev = now;
            let step = dt / 0.16;
            for i in 0..3 {
                let target = if self.hover == Some(i) { 1.0 } else { 0.0 };
                let t = &mut self.win_btn_anim[i];
                if (*t - target).abs() <= step {
                    *t = target;
                } else {
                    *t += step * (target - *t).signum();
                }
            }
        }
        for (i, (bx, by, bw, bh)) in ch.buttons.iter().enumerate() {
            let t = self.win_btn_anim[i];
            let base = btn_colors[i];
            let hov = btn_hover[i];
            let c = [
                base[0] + (hov[0] - base[0]) * t,
                base[1] + (hov[1] - base[1]) * t,
                base[2] + (hov[2] - base[2]) * t,
                1.0,
            ];
            if t > 0.002 {
                let halo = *bw * 0.7 * t;
                let mut hc = c;
                hc[3] = 0.35 * t;
                verts += push_rquad(
                    &mut bytes, sw, sh, *bx - halo * 0.5, *by - halo * 0.5,
                    *bw + halo, *bh + halo, hc, (*bw + halo) * 0.5,
                );
            }
            let grow = *bw * 0.3 * t;
            verts += push_rquad(
                &mut bytes, sw, sh, *bx - grow * 0.5, *by - grow * 0.5,
                *bw + grow, *bh + grow, c, (*bw + grow) * 0.5,
            );
        }
        if !bytes.is_empty() {
            self.queue.write_buffer(&self.quad_vbuf, 0, &bytes);
        }

        // Text areas: terminal grid + button glyphs.
        // Shrink the terminal glyph clip away from the scroll-back overlay
        // strip so the strip's dark fill reads as solid (terminal text no
        // longer paints through it). Only when an overlay is up.
        let term_bounds_bottom: i32 =
            if let Some(&overlay_y) = overlay_top.as_ref() {
                (overlay_y - self.pad).max(iy) as i32
            } else {
                (iy + ih) as i32
            };
        let term_area = TextArea {
            buffer: &self.buffer,
            left: ix + self.pad,
            top: iy + self.pad,
            scale: 1.0,
            bounds: TextBounds {
                left: ix as i32,
                top: iy as i32,
                right: (ix + iw) as i32,
                bottom: term_bounds_bottom,
            },
            default_color: Color::rgb(220, 214, 201),
            custom_glyphs: &[],
        };
        // NOTE: the overlay pill and the menu card were already pushed into
        // `bytes` above (they had to be written before the quad vbuf upload).
        // The label TextArea for the menu must be appended here, where the
        // areas vec is assembled for glyphon.
        let mut areas = Vec::with_capacity(3);
        if let Some(buf) = &self.overlay {
            areas.push(TextArea {
                buffer: buf,
                left: ix + self.pad * 1.5,
                top: overlay_top.unwrap_or(iy + ih),
                scale: 1.0,
                bounds: TextBounds {
                    left: ix as i32,
                    top: (overlay_top.unwrap_or(iy + ih)) as i32,
                    right: (ix + iw) as i32,
                    bottom: (iy + ih) as i32,
                },
                default_color: Color::rgb(232, 232, 238),
                custom_glyphs: &[],
            });
        }
        // NOTE: popout's overlay pill was previously pushed here in `bytes`.
        // It is now drawn above as part of the menu+pill batch so we can do a
        // single write_buffer before the text pass (avoiding the push_rquad
        // borrow conflict against `self.overlay`).
        if let Some((mx, my, mw, _)) = menu_area_desc {
            areas.push(TextArea {
                buffer: &self.context_buffer,
                left: mx,
                top: my,
                scale: 1.0,
                bounds: TextBounds {
                    left: (mx - 10.0 * s) as i32,
                    top: my as i32,
                    right: (mx + mw) as i32,
                    bottom: self.config.height as i32,
                },
                default_color: Color::rgb(226, 219, 205),
                custom_glyphs: &[],
            });
        }
        areas.push(term_area);
        if self
            .text_renderer
            .prepare(
                &self.device,
                &self.queue,
                &mut self.font_system,
                &mut self.atlas,
                &self.viewport,
                areas,
                &mut self.swash_cache,
            )
            .is_err()
        {
            self.window.request_redraw();
            return;
        }

        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame) => frame,
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => {
                self.window.request_redraw();
                return;
            }
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Suboptimal(_) => {
                self.surface.configure(&self.device, &self.config);
                self.window.request_redraw();
                return;
            }
            wgpu::CurrentSurfaceTexture::Lost => {
                self.surface = self
                    .instance
                    .create_surface(self.window.clone())
                    .expect("popout: recreate a wgpu surface");
                self.surface.configure(&self.device, &self.config);
                self.window.request_redraw();
                return;
            }
            wgpu::CurrentSurfaceTexture::Validation => {
                panic!("umber-ui popout: surface validation error");
            }
        };
        let view = frame.texture.create_view(&TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("umber popout pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(wgpu::Color {
                            // Lighter warm-gray shell so the dark islands float,
                            // matching the editor's layered look.
                            r: 0.095,
                            g: 0.088,
                            b: 0.078,
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
            if under_verts > 0 {
                pass.set_pipeline(&self.quad_pipeline);
                pass.set_vertex_buffer(0, self.quad_vbuf.slice(..));
                pass.draw(0..under_verts, 0..1);
            }
            let _ = self
                .text_renderer
                .render(&self.atlas, &self.viewport, &mut pass);
            // Control island + buttons draw OVER the text: a clean cut-out.
            if verts > under_verts {
                pass.set_pipeline(&self.quad_pipeline);
                pass.set_vertex_buffer(0, self.quad_vbuf.slice(..));
                pass.draw(under_verts..verts, 0..1);
            }
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        self.queue.present(frame);
        if self.win_btn_anim.iter().any(|t| *t > 0.0 && *t < 1.0) {
            self.window.request_redraw();
        }
    }
}
