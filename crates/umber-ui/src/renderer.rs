//! The P0 GPU text renderer.
//!
//! Structure and call sequence mirror glyphon 0.12's `hello-world` example
//! (the authoritative reference for the glyphon 0.12 / wgpu 30 / winit 0.30
//! surface API). Shaping types come from cosmic-text (`Buffer`, `FontSystem`,
//! `SwashCache`); the GPU bridge comes from glyphon (`TextAtlas`,
//! `TextRenderer`, `Viewport`, `TextArea`). Both resolve to the single
//! cosmic-text 0.19 build glyphon pins, so the types are identical.

use std::sync::Arc;

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
/// minimal chrome).
const PAD: f32 = 8.0;

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
    text_buffer: Buffer,

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

        // 14px monospace body text is the P0 default; P1 pulls this from the
        // TOML config (D13).
        let text_buffer = Buffer::new(&mut font_system, Metrics::new(14.0, 20.0));

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
            text_buffer,
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

    /// Reconfigure the surface after a resize and reflow the text to it.
    pub fn resize(&mut self, width: u32, height: u32) {
        self.surface_config.width = width.max(1);
        self.surface_config.height = height.max(1);
        self.surface.configure(&self.device, &self.surface_config);
        self.reflow_buffer_size();
    }

    /// Replace the shaped text (the render spike's visible-line window).
    pub fn set_text(&mut self, text: &str) {
        self.text_buffer.set_text(
            text,
            &Attrs::new().family(Family::Monospace),
            Shaping::Advanced,
            None,
        );
        self.reflow_buffer_size();
    }

    fn reflow_buffer_size(&mut self) {
        let w = (self.surface_config.width as f32 - PAD * 2.0).max(1.0);
        let h = (self.surface_config.height as f32 - PAD * 2.0).max(1.0);
        // TODO(P0): honor `window.scale_factor()` for HiDPI in metrics + size.
        self.text_buffer.set_size(Some(w), Some(h));
        self.text_buffer
            .shape_until_scroll(&mut self.font_system, false);
    }

    /// Draw one frame of the current text buffer. Returns `false` if the frame
    /// was skipped (surface lost/outdated) and a redraw was requested.
    pub fn render(&mut self) -> bool {
        // TODO(P0): stamp a keystroke->present timer here and record the
        // present timestamp for the p99 latency exit criterion (Bidfall
        // cx-engine discipline).
        self.viewport.update(
            &self.queue,
            Resolution {
                width: self.surface_config.width,
                height: self.surface_config.height,
            },
        );

        if let Err(err) = self.text_renderer.prepare(
            &self.device,
            &self.queue,
            &mut self.font_system,
            &mut self.atlas,
            &self.viewport,
            [TextArea {
                buffer: &self.text_buffer,
                left: PAD,
                top: PAD,
                scale: 1.0,
                bounds: TextBounds {
                    left: 0,
                    top: 0,
                    right: self.surface_config.width as i32,
                    bottom: self.surface_config.height as i32,
                },
                default_color: Color::rgb(220, 220, 220),
                custom_glyphs: &[],
            }],
            &mut self.swash_cache,
        ) {
            // Atlas out of space etc. — drop this frame, try again.
            eprintln!("umber-ui: text prepare failed: {err:?}");
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
        true
    }
}
