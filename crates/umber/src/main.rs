//! umber — window, event loop, and the wiring that will host the kernel +
//! module host + workspace backend (docs/PLAN.md architecture sketch).
//!
//! P0 render spike: open a Wayland-capable winit window, hand its `Arc<Window>`
//! to umber-ui's wgpu/glyphon [`Renderer`], load the file named in argv into an
//! umber-text [`TextBuffer`] (ropey), and draw its scroll-visible lines. This
//! slice closes the P0 exit criteria (docs/PLAN.md): a single-cursor typing
//! path, keystroke->present latency instrumentation (D4 GO/NO-GO: p99 <= 8 ms),
//! scroll over a 100 MB file, HiDPI, and a cold-start + idle-RAM measurement
//! harness that prints everything a human needs to record the D4 verdict.

use std::process::ExitCode;
use std::sync::Arc;
use std::time::{Duration, Instant};

use umber_text::TextBuffer;
use umber_ui::Renderer;

use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, ModifiersState, NamedKey};
use winit::window::{Window, WindowId};

/// Extra lines shaped just past the visible window so a small scroll doesn't
/// reveal an unshaped gap. Only visible+margin lines are ever shaped.
const MARGIN: usize = 8;

/// Lines advanced per mouse-wheel notch (line-delta devices).
const WHEEL_LINES: f32 = 3.0;

/// Base line height in logical px, for converting pixel-delta scroll to lines.
const BASE_LINE_PX: f64 = 20.0;

fn main() -> ExitCode {
    // Cold-start clock starts at the earliest point in the process (docs/PLAN.md
    // P0 exit: cold start <= 300 ms).
    let start = Instant::now();

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
        cursor_char: 0,
        first_visible_line: 0,
        goal_col: 0,
        modifiers: ModifiersState::empty(),
        start,
        first_frame: false,
        first_frame_at: None,
        rss_printed: false,
    };

    if let Err(err) = event_loop.run_app(&mut app) {
        eprintln!("umber: event loop error: {err}");
        return ExitCode::FAILURE;
    }

    // Final D4 latency verdict to stdout (companion to the live banner).
    if let Some(renderer) = &app.renderer {
        println!("{}", renderer.latency_summary());
    }
    ExitCode::SUCCESS
}

struct App {
    buffer: TextBuffer,
    renderer: Option<Renderer>,

    /// Single cursor as an absolute char index into the buffer (multi-cursor is
    /// P1). `goal_col` preserves the visual column across vertical moves.
    cursor_char: usize,
    goal_col: usize,

    /// First document line drawn; the scroll window is `[first_visible_line ..
    /// first_visible_line + capacity + MARGIN)`.
    first_visible_line: usize,

    modifiers: ModifiersState,

    // --- measurement harness ---
    start: Instant,
    first_frame: bool,
    first_frame_at: Option<Instant>,
    rss_printed: bool,
}

impl App {
    /// Number of whole document lines that fit in the window right now.
    fn page(&self) -> usize {
        self.renderer
            .as_ref()
            .map(|r| r.visible_line_capacity().max(1))
            .unwrap_or(1)
    }

    /// Re-derive `goal_col` from the cursor's current column (called after any
    /// horizontal move or edit; vertical moves deliberately preserve it).
    fn update_goal_col(&mut self) {
        let line = self.buffer.char_to_line(self.cursor_char);
        self.goal_col = self.cursor_char - self.buffer.line_to_char(line);
    }

    /// Move the cursor up/down one line, keeping `goal_col` where possible.
    fn move_vertical(&mut self, delta: i64) {
        let line = self.buffer.char_to_line(self.cursor_char) as i64;
        let last = self.buffer.len_lines().saturating_sub(1) as i64;
        let target = (line + delta).clamp(0, last) as usize;
        let col = self.goal_col.min(self.buffer.visual_line_len_chars(target));
        self.cursor_char = self.buffer.line_to_char(target) + col;
    }

    /// Push the current buffer window, cursor, and banner prefix to the
    /// renderer. `follow_cursor` scrolls to keep the cursor visible (edits and
    /// caret moves); explicit scrolls pass `false` so the view stays put.
    fn apply_view(&mut self, follow_cursor: bool) {
        let cap = match self.renderer.as_ref() {
            Some(r) => r.visible_line_capacity().max(1),
            None => return,
        };
        let last_line = self.buffer.len_lines().saturating_sub(1);

        if follow_cursor {
            let cl = self.buffer.char_to_line(self.cursor_char);
            if cl < self.first_visible_line {
                self.first_visible_line = cl;
            } else if cl >= self.first_visible_line + cap {
                self.first_visible_line = cl + 1 - cap;
            }
        }
        if self.first_visible_line > last_line {
            self.first_visible_line = last_line;
        }

        let text = self
            .buffer
            .visible_text(self.first_visible_line, cap + MARGIN);

        let cl = self.buffer.char_to_line(self.cursor_char);
        let col = self.cursor_char - self.buffer.line_to_char(cl);
        // Cursor is only drawable inside the shaped/visible `cap` lines — the
        // MARGIN lines are in the rope slice but clipped by the shaping box, so
        // a cursor there would render invisibly.
        let cursor = if cl >= self.first_visible_line && cl < self.first_visible_line + cap {
            Some((cl - self.first_visible_line, col))
        } else {
            None
        };

        let name = self
            .buffer
            .path()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "*scratch*".to_string());
        let prefix = format!(
            "umber P0 \u{2014} {name} \u{2014} {} lines, {} bytes \u{2014} Ln {}, Col {}",
            self.buffer.len_lines(),
            self.buffer.len_bytes(),
            cl + 1,
            col + 1,
        );

        if let Some(renderer) = self.renderer.as_mut() {
            renderer.set_document(&text);
            renderer.set_cursor(cursor);
            renderer.set_stats_prefix(prefix);
        }
    }

    /// Adjust the scroll offset by `delta` lines, clamped to the buffer.
    fn scroll_by(&mut self, delta: i64) {
        let last = self.buffer.len_lines().saturating_sub(1) as i64;
        self.first_visible_line = (self.first_visible_line as i64 + delta).clamp(0, last) as usize;
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.renderer.is_some() {
            return;
        }

        // Event-driven: only wake on input/redraw (allocation-light idle).
        event_loop.set_control_flow(ControlFlow::Wait);

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

        let renderer = Renderer::new(window, event_loop);
        self.renderer = Some(renderer);
        self.apply_view(true);
        if let Some(renderer) = self.renderer.as_ref() {
            renderer.window().request_redraw();
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        if self.renderer.is_none() {
            return;
        }

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::Resized(size) => {
                if let Some(renderer) = self.renderer.as_mut() {
                    renderer.resize(size.width, size.height);
                }
                self.apply_view(false);
                if let Some(renderer) = self.renderer.as_ref() {
                    renderer.window().request_redraw();
                }
            }

            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                if let Some(renderer) = self.renderer.as_mut() {
                    renderer.set_scale_factor(scale_factor);
                }
                // A `Resized` normally follows; re-window now so the frame in
                // between is correct.
                self.apply_view(false);
                if let Some(renderer) = self.renderer.as_ref() {
                    renderer.window().request_redraw();
                }
            }

            WindowEvent::ModifiersChanged(modifiers) => {
                self.modifiers = modifiers.state();
            }

            WindowEvent::MouseWheel { delta, .. } => {
                // Scroll is a P0 exit-criterion path (100 MB fixture), so it
                // feeds the D4 latency ring exactly like keystrokes do.
                let t = Instant::now();
                let lines = match delta {
                    MouseScrollDelta::LineDelta(_, y) => (-y * WHEEL_LINES) as i64,
                    MouseScrollDelta::PixelDelta(p) => (-p.y / BASE_LINE_PX) as i64,
                };
                if lines != 0 {
                    self.scroll_by(lines);
                    self.apply_view(false);
                    if let Some(renderer) = self.renderer.as_mut() {
                        renderer.mark_keystroke(t);
                        renderer.window().request_redraw();
                    }
                }
            }

            WindowEvent::KeyboardInput { event, .. } => {
                if event.state != ElementState::Pressed {
                    return;
                }
                // Timestamp at event receipt — the head of the keystroke->present
                // latency measurement (D4).
                let t = Instant::now();
                let ctrl = self.modifiers.control_key();
                let len = self.buffer.len_chars();
                let mut changed = false;
                let mut follow = true;

                match &event.logical_key {
                    Key::Named(NamedKey::Backspace) => {
                        if self.cursor_char > 0 {
                            self.buffer
                                .remove_char_range(self.cursor_char - 1, self.cursor_char);
                            self.cursor_char -= 1;
                            self.update_goal_col();
                            changed = true;
                        }
                    }
                    Key::Named(NamedKey::Delete) => {
                        if self.cursor_char < len {
                            self.buffer
                                .remove_char_range(self.cursor_char, self.cursor_char + 1);
                            changed = true;
                        }
                    }
                    Key::Named(NamedKey::Enter) => {
                        self.buffer.insert_char(self.cursor_char, '\n');
                        self.cursor_char += 1;
                        self.update_goal_col();
                        changed = true;
                    }
                    Key::Named(NamedKey::Tab) => {
                        self.buffer.insert_char(self.cursor_char, '\t');
                        self.cursor_char += 1;
                        self.update_goal_col();
                        changed = true;
                    }
                    Key::Named(NamedKey::ArrowLeft) => {
                        self.cursor_char = self.cursor_char.saturating_sub(1);
                        self.update_goal_col();
                        changed = true;
                    }
                    Key::Named(NamedKey::ArrowRight) => {
                        self.cursor_char = (self.cursor_char + 1).min(len);
                        self.update_goal_col();
                        changed = true;
                    }
                    Key::Named(NamedKey::ArrowUp) => {
                        self.move_vertical(-1);
                        changed = true;
                    }
                    Key::Named(NamedKey::ArrowDown) => {
                        self.move_vertical(1);
                        changed = true;
                    }
                    Key::Named(NamedKey::Home) => {
                        self.cursor_char = if ctrl {
                            0
                        } else {
                            let l = self.buffer.char_to_line(self.cursor_char);
                            self.buffer.line_to_char(l)
                        };
                        self.update_goal_col();
                        changed = true;
                    }
                    Key::Named(NamedKey::End) => {
                        self.cursor_char = if ctrl {
                            len
                        } else {
                            let l = self.buffer.char_to_line(self.cursor_char);
                            self.buffer.line_to_char(l) + self.buffer.visual_line_len_chars(l)
                        };
                        self.update_goal_col();
                        changed = true;
                    }
                    Key::Named(NamedKey::PageUp) => {
                        let cap = self.page();
                        self.scroll_by(-(cap as i64));
                        changed = true;
                        follow = false;
                    }
                    Key::Named(NamedKey::PageDown) => {
                        let cap = self.page();
                        self.scroll_by(cap as i64);
                        changed = true;
                        follow = false;
                    }
                    _ => {}
                }

                // Printable input arrives as `event.text` (layout-resolved).
                // Skip when Ctrl is held so chords don't type their letter, and
                // skip control chars (Enter/Tab are handled as named keys).
                if !ctrl {
                    if let Some(text) = &event.text {
                        for ch in text.chars() {
                            if !ch.is_control() {
                                self.buffer.insert_char(self.cursor_char, ch);
                                self.cursor_char += 1;
                                changed = true;
                            }
                        }
                        if changed {
                            self.update_goal_col();
                        }
                    }
                }

                if changed {
                    self.apply_view(follow);
                    if let Some(renderer) = self.renderer.as_mut() {
                        renderer.mark_keystroke(t);
                        renderer.window().request_redraw();
                    }
                }
            }

            WindowEvent::RedrawRequested => {
                let presented = match self.renderer.as_mut() {
                    Some(renderer) => renderer.render(),
                    None => return,
                };
                if presented && !self.first_frame {
                    self.first_frame = true;
                    let now = Instant::now();
                    self.first_frame_at = Some(now);
                    println!(
                        "cold-start: {:.1} ms (main entry -> first frame presented)",
                        self.start.elapsed().as_secs_f64() * 1000.0
                    );
                    // Wake ~2s later to sample idle RAM even with no input.
                    event_loop
                        .set_control_flow(ControlFlow::WaitUntil(now + Duration::from_secs(2)));
                }
            }

            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if self.first_frame && !self.rss_printed {
            if let Some(t0) = self.first_frame_at {
                if t0.elapsed() >= Duration::from_secs(2) {
                    match read_vmrss() {
                        Some(rss) => println!("idle RAM (VmRSS): {rss}"),
                        None => println!("idle RAM (VmRSS): unavailable"),
                    }
                    self.rss_printed = true;
                    event_loop.set_control_flow(ControlFlow::Wait);
                }
            }
        }
    }
}

/// Resident set size from `/proc/self/status` (`VmRSS`), formatted as MB + kB.
/// The P0 idle-RAM exit criterion is <= 150 MB (docs/PLAN.md).
fn read_vmrss() -> Option<String> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            let kb: u64 = rest.split_whitespace().next()?.parse().ok()?;
            return Some(format!("{:.1} MB ({} kB)", kb as f64 / 1024.0, kb));
        }
    }
    None
}
