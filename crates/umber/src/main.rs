//! umber — window, event loop, and the wiring that will host the kernel +
//! module host + workspace backend (docs/PLAN.md architecture sketch).
//!
//! P0 render spike: open a Wayland-capable winit window, hand its `Arc<Window>`
//! to umber-ui's wgpu/glyphon [`Renderer`], load the file named in argv into an
//! umber-text [`TextBuffer`] (ropey), and draw its visible lines. Typing and
//! keystroke->present latency instrumentation are stubbed with TODO(P0)
//! markers — the P0 exit criteria (docs/PLAN.md) close those out.

use std::process::ExitCode;
use std::sync::Arc;

use umber_text::TextBuffer;
use umber_ui::Renderer;

use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

/// How many lines the spike feeds to the shaper. P1 replaces this fixed window
/// with scroll-driven, damage-tracked shaping in umber-ui.
const SPIKE_VISIBLE_LINES: usize = 200;

fn main() -> ExitCode {
    // argv[1] (optional) is the file to open; absent means a scratch buffer.
    let path = std::env::args_os().nth(1);

    let buffer = match &path {
        Some(p) => match TextBuffer::from_path(p) {
            Ok(buf) => buf,
            Err(err) => {
                eprintln!("umber: cannot open {:?}: {err}", p);
                return ExitCode::FAILURE;
            }
        },
        None => TextBuffer::empty(),
    };

    let event_loop = match EventLoop::new() {
        Ok(ev) => ev,
        Err(err) => {
            eprintln!("umber: failed to create the event loop: {err}");
            return ExitCode::FAILURE;
        }
    };

    let mut app = App {
        buffer,
        renderer: None,
    };

    if let Err(err) = event_loop.run_app(&mut app) {
        eprintln!("umber: event loop error: {err}");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

struct App {
    buffer: TextBuffer,
    renderer: Option<Renderer>,
}

impl App {
    /// The initial view fed to the shaper: a file-stats banner plus the first
    /// `SPIKE_VISIBLE_LINES` lines of the buffer.
    fn initial_view(&self) -> String {
        let name = self
            .buffer
            .path()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "*scratch*".to_string());
        let banner = format!(
            "umber P0 spike — {name} — {} lines, {} bytes\n\n",
            self.buffer.len_lines(),
            self.buffer.len_bytes(),
        );
        // TODO(P0): drive `start` from scroll offset instead of always 0.
        format!(
            "{banner}{}",
            self.buffer.visible_text(0, SPIKE_VISIBLE_LINES)
        )
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.renderer.is_some() {
            return;
        }

        let attributes = Window::default_attributes()
            .with_title("umber")
            .with_inner_size(LogicalSize::new(1000.0, 700.0));
        let window = match event_loop.create_window(attributes) {
            Ok(w) => Arc::new(w),
            Err(err) => {
                eprintln!("umber: failed to create window: {err}");
                event_loop.exit();
                return;
            }
        };

        let mut renderer = Renderer::new(window, event_loop);
        renderer.set_text(&self.initial_view());
        renderer.window().request_redraw();
        self.renderer = Some(renderer);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(renderer) = self.renderer.as_mut() else {
            return;
        };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                renderer.resize(size.width, size.height);
                renderer.window().request_redraw();
            }
            WindowEvent::KeyboardInput { .. } => {
                // TODO(P0): route keystrokes into an umber-text edit + reshape,
                // and stamp keystroke->present latency (docs/PLAN.md P0 exit:
                // p99 <= 8 ms). The buffer is read-only in this slice.
            }
            WindowEvent::RedrawRequested => {
                renderer.render();
            }
            _ => {}
        }
    }
}
