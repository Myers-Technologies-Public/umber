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

use std::fmt::Write as _;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::{Duration, Instant};

use umber_kernel::{Command, CommandRegistry, Config, FeatureRegistry};
use umber_text::TextBuffer;
use umber_ui::{OverlaySpec, Renderer, ScrollbarInfo, SelSpan};

use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent};
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

/// Number of rows on the settings page (drives selection clamping).
const SETTINGS_ROWS: usize = 6;

/// The current top-level input surface. A single keyboard dispatch point routes
/// by this state (Slice 2): the editor path is unchanged from Slice 1; the
/// three modals capture all input while open and render over a dimmed editor
/// frame.
#[derive(Clone, Copy, PartialEq, Eq)]
enum View {
    Editor,
    Palette,
    Settings,
    Modules,
}

/// The full command set (D6). Registration order is the palette's default
/// listing order and the tie-break for equal fuzzy scores.
fn build_command_registry() -> CommandRegistry {
    let mut reg = CommandRegistry::new();
    for (id, title, key) in [
        ("file.save", "File: Save", "Ctrl+S"),
        ("edit.undo", "Edit: Undo", "Ctrl+Z"),
        ("edit.redo", "Edit: Redo", "Ctrl+Shift+Z"),
        ("edit.copy", "Edit: Copy", "Ctrl+C"),
        ("edit.cut", "Edit: Cut", "Ctrl+X"),
        ("edit.paste", "Edit: Paste", "Ctrl+V"),
        ("edit.selectAll", "Edit: Select All", "Ctrl+A"),
        ("goto.fileStart", "Go: File Start", "Ctrl+Home"),
        ("goto.fileEnd", "Go: File End", "Ctrl+End"),
        (
            "view.commandPalette",
            "View: Command Palette",
            "Ctrl+Shift+P",
        ),
        ("view.settings", "Preferences: Open Settings", "Ctrl+,"),
        ("view.modules", "Modules: Manage", ""),
        (
            "view.toggle.gutter",
            "View: Toggle Gutter / Line Numbers",
            "",
        ),
        (
            "view.toggle.scrollbar",
            "View: Toggle Overlay Scrollbar",
            "",
        ),
        ("view.toggle.latencyHud", "View: Toggle Latency HUD", ""),
        ("app.quit", "Application: Quit", "Ctrl+Q"),
    ] {
        reg.register(Command {
            id,
            title,
            keybinding: key,
        });
    }
    reg
}

/// Human ON/OFF label for a boolean setting/feature.
fn onoff(v: bool) -> String {
    if v {
        "ON".to_string()
    } else {
        "OFF".to_string()
    }
}

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

    // Wayland-first clipboard (arboard + wayland-data-control). A failure here
    // must not sink the editor \u{2014} degrade to no clipboard.
    let clipboard = match arboard::Clipboard::new() {
        Ok(cb) => Some(cb),
        Err(err) => {
            eprintln!("umber: clipboard unavailable ({err}); copy/paste disabled");
            None
        }
    };

    let config = Config::load();
    let features = FeatureRegistry::from_config(&config);
    let commands = build_command_registry();
    let scrollbar_linger = Duration::from_millis(config.scrollbar_linger_ms);

    let mut app = App {
        buffer,
        renderer: None,
        view: View::Editor,
        config,
        features,
        commands,
        palette_query: String::new(),
        palette_filtered: Vec::new(),
        palette_sel: 0,
        settings_sel: 0,
        modules_sel: 0,
        modules_hint: None,
        scrollbar_linger,
        cursor_char: 0,
        goal_col: 0,
        selection_anchor: None,
        selecting: false,
        clipboard,
        sel_spans: Vec::new(),
        first_visible_line: 0,
        modifiers: ModifiersState::empty(),
        pointer: (0.0, 0.0),
        scrollbar_deadline: None,
        scrollbar_dragging: false,
        drag_anchor_y: 0.0,
        drag_anchor_first: 0,
        scrollbar_drawn: false,
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

    // --- Slice 2: kernel + modal views ---
    /// Current input surface (editor or a modal).
    view: View,
    /// Loaded config (D13); live-applied and persisted on change.
    config: Config,
    /// Feature/module registry (D10).
    features: FeatureRegistry,
    /// Command registry (D6), the palette's source.
    commands: CommandRegistry,
    /// Palette query, filtered command indices, and selected row.
    palette_query: String,
    palette_filtered: Vec<usize>,
    palette_sel: usize,
    /// Settings page selected row.
    settings_sel: usize,
    /// Modules page selected row + a transient status hint (e.g. kernel
    /// refusal per D10).
    modules_sel: usize,
    modules_hint: Option<String>,
    /// Scrollbar auto-hide linger from config (replaces the old fixed const).
    scrollbar_linger: Duration,

    /// Single cursor as an absolute char index into the buffer (multi-cursor is
    /// P1). `goal_col` preserves the visual column across vertical moves.
    cursor_char: usize,
    goal_col: usize,

    /// Selection anchor as an absolute char index; the head is `cursor_char`.
    /// `None` = no selection; a non-empty selection is `anchor != cursor_char`.
    selection_anchor: Option<usize>,
    /// True while the left button is held after a text press, so `CursorMoved`
    /// extends the selection (drag-select).
    selecting: bool,
    /// System clipboard (arboard). `None` when init failed \u{2014} copy/cut/paste
    /// then degrade to a no-op with an eprintln, never a panic.
    clipboard: Option<arboard::Clipboard>,
    /// Reused buffer for the per-view selection highlight spans, rebuilt in
    /// `apply_view` and handed to the renderer.
    sel_spans: Vec<SelSpan>,

    /// First document line drawn; the scroll window is `[first_visible_line ..
    /// first_visible_line + capacity + MARGIN)`.
    first_visible_line: usize,

    modifiers: ModifiersState,

    // --- mouse + overlay scrollbar ---
    /// Latest pointer position in physical pixels (from `CursorMoved`).
    pointer: (f64, f64),
    /// Instant the scrollbar should hide; it paints while `now < deadline` (or
    /// while dragging). `None` = hidden.
    scrollbar_deadline: Option<Instant>,
    scrollbar_dragging: bool,
    /// Drag anchors: pointer-Y and first-visible-line at grab time. Absolute
    /// mapping from the anchor avoids drift.
    drag_anchor_y: f64,
    drag_anchor_first: usize,
    /// Whether the last presented frame drew the scrollbar, so a linger-out can
    /// schedule exactly one erase redraw.
    scrollbar_drawn: bool,

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

        let dirty = if self.buffer.is_dirty() {
            "\u{2022} "
        } else {
            ""
        };
        let name = self
            .buffer
            .path()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "*scratch*".to_string());
        let prefix = format!(
            "umber P0 \u{2014} {dirty}{name} \u{2014} {} lines, {} bytes \u{2014} Ln {}, Col {}",
            self.buffer.len_lines(),
            self.buffer.len_bytes(),
            cl + 1,
            col + 1,
        );

        // Line-number gutter for the shaped window. The string changes exactly
        // when `first_visible_line` or the line count changes \u{2014} the same
        // only-on-change contract the renderer's gutter guard relies on. Width
        // is fixed by the whole file's last line number, so it never jitters.
        let total = self.buffer.len_lines();
        let digits = digit_count(total);
        let win_last = (self.first_visible_line + cap + MARGIN).min(total);
        let mut numbers = String::new();
        for ln in self.first_visible_line..win_last {
            if ln > self.first_visible_line {
                numbers.push('\n');
            }
            let _ = write!(numbers, "{:>width$}", ln + 1, width = digits);
        }

        // Selection highlight spans for the visible window (window-relative
        // lines). Interior lines are full-width (`end_col = None`); the first and
        // last selected lines are partial. Off-screen lines are skipped. Taken
        // out of `self` so the span build can borrow the buffer immutably.
        let mut spans = std::mem::take(&mut self.sel_spans);
        spans.clear();
        if let Some((sel_s, sel_e)) = self.selection_range() {
            let s_line = self.buffer.char_to_line(sel_s);
            let e_line = self.buffer.char_to_line(sel_e);
            let win_start = self.first_visible_line;
            let win_end = self.first_visible_line + cap; // exclusive
            let last_line = self.buffer.len_lines().saturating_sub(1);
            let from = s_line.max(win_start);
            let to = e_line.min(win_end.saturating_sub(1)).min(last_line);
            for line in from..=to {
                let line_start = self.buffer.line_to_char(line);
                let start_col = if line == s_line {
                    sel_s - line_start
                } else {
                    0
                };
                let end_col = if line == e_line {
                    Some(sel_e - line_start)
                } else {
                    None
                };
                spans.push(SelSpan {
                    line: line - win_start,
                    start_col,
                    end_col,
                });
            }
        }

        if let Some(renderer) = self.renderer.as_mut() {
            renderer.set_gutter(&numbers, digits);
            renderer.set_document(&text);
            renderer.set_cursor(cursor);
            renderer.set_selection(&spans);
            renderer.set_stats_prefix(prefix);
        }
        self.sel_spans = spans;
    }

    /// Adjust the scroll offset by `delta` lines, clamped to the buffer.
    fn scroll_by(&mut self, delta: i64) {
        let last = self.buffer.len_lines().saturating_sub(1) as i64;
        self.first_visible_line = (self.first_visible_line as i64 + delta).clamp(0, last) as usize;
    }

    /// Show the scrollbar and (re)start its linger countdown.
    fn poke_scrollbar(&mut self) {
        if self.config.scrollbar {
            self.scrollbar_deadline = Some(Instant::now() + self.scrollbar_linger);
        }
    }

    /// Whether the scrollbar should paint right now.
    fn scrollbar_visible(&self, now: Instant) -> bool {
        self.scrollbar_dragging || self.scrollbar_deadline.map_or(false, |d| now < d)
    }

    /// True when the pointer sits in the right-edge hover zone and the document
    /// actually overflows (hovering an un-scrollable file shows nothing).
    fn pointer_in_scrollbar_zone(&self) -> bool {
        let renderer = match self.renderer.as_ref() {
            Some(r) => r,
            None => return false,
        };
        let cap = renderer.visible_line_capacity();
        if self.buffer.len_lines() <= cap {
            return false;
        }
        let (w, _h) = renderer.size();
        let x = self.pointer.0 as f32;
        let y = self.pointer.1 as f32;
        y >= renderer.doc_top() && x >= w as f32 - renderer.scrollbar_edge_zone()
    }

    /// Handle a left-press that may land on the scrollbar. Returns `true` if the
    /// press was consumed (thumb grab or track paging), so the caller skips
    /// click-to-position.
    fn try_scrollbar_press(&mut self) -> bool {
        if !self.scrollbar_visible(Instant::now()) {
            return false;
        }
        let total = self.buffer.len_lines();
        let first = self.first_visible_line;
        let (g, width, zone, cap) = match self.renderer.as_ref() {
            Some(r) => match r.scrollbar_geom(first, total) {
                Some(g) => (
                    g,
                    r.size().0 as f32,
                    r.scrollbar_edge_zone(),
                    r.visible_line_capacity().max(1),
                ),
                None => return false,
            },
            None => return false,
        };
        let x = self.pointer.0 as f32;
        let y = self.pointer.1 as f32;
        // Grab anywhere in the edge zone, not just the thin track, for feel.
        let grab_left = (width - zone).min(g.track_x);
        if x < grab_left {
            return false;
        }
        if y >= g.thumb_top && y <= g.thumb_top + g.thumb_h {
            self.scrollbar_dragging = true;
            self.drag_anchor_y = self.pointer.1;
            self.drag_anchor_first = first;
            true
        } else if y >= g.track_top && y <= g.track_top + g.track_h {
            // Page toward the click (above the thumb -> up, below -> down).
            if y < g.thumb_top {
                self.scroll_by(-(cap as i64));
            } else {
                self.scroll_by(cap as i64);
            }
            self.apply_view(false);
            true
        } else {
            false
        }
    }

    /// Continue a thumb drag: map the pointer's Y offset since grab to a line
    /// offset from the anchored first-visible-line.
    fn drag_scrollbar(&mut self, pointer_y: f64) {
        let total = self.buffer.len_lines();
        let (track_h, thumb_h, cap) = match self.renderer.as_ref() {
            Some(r) => match r.scrollbar_geom(self.drag_anchor_first, total) {
                Some(g) => (g.track_h, g.thumb_h, r.visible_line_capacity()),
                None => return,
            },
            None => return,
        };
        let scroll_range = total.saturating_sub(cap) as f32;
        let travel = (track_h - thumb_h).max(1.0);
        let dy = (pointer_y - self.drag_anchor_y) as f32;
        let line_delta = (dy / travel * scroll_range).round() as i64;
        let target =
            (self.drag_anchor_first as i64 + line_delta).clamp(0, scroll_range as i64) as usize;
        self.first_visible_line = target;
        self.apply_view(false);
        if let Some(r) = self.renderer.as_ref() {
            r.window().request_redraw();
        }
    }

    /// Map the current pointer to a document position and move the caret there.
    /// Returns `true` if the caret moved (caller marks latency + redraws). Uses
    /// the same gutter/cell arithmetic as cursor rendering so click and caret
    /// agree.
    fn pointer_to_char(&self) -> Option<usize> {
        let (doc_top, line_px, text_left, cell_w) = match self.renderer.as_ref() {
            Some(r) => (r.doc_top(), r.line_px(), r.text_left(), r.cell_w()),
            None => return None,
        };
        let x = self.pointer.0 as f32;
        let y = self.pointer.1 as f32;
        if y < doc_top {
            return None; // banner, not the document
        }
        let rel_line = ((y - doc_top) / line_px).floor() as i64;
        let last = self.buffer.len_lines().saturating_sub(1) as i64;
        let line = (self.first_visible_line as i64 + rel_line).clamp(0, last) as usize;
        let col_f = ((x - text_left) / cell_w).round();
        let col = if col_f < 0.0 { 0 } else { col_f as usize };
        let col = col.min(self.buffer.visual_line_len_chars(line));
        Some(self.buffer.line_to_char(line) + col)
    }

    /// Ordered non-empty selection range `(start, end)` in char indices, or
    /// `None` when there is no selection (anchor absent or collapsed onto the
    /// caret).
    fn selection_range(&self) -> Option<(usize, usize)> {
        match self.selection_anchor {
            Some(a) if a != self.cursor_char => {
                Some((a.min(self.cursor_char), a.max(self.cursor_char)))
            }
            _ => None,
        }
    }

    /// Prepare for a cursor move: end the typing-coalesce run, and either open
    /// an anchor (shift held, extending the selection) or drop the selection
    /// (plain move collapses).
    fn begin_move(&mut self, shift: bool) {
        self.buffer.break_coalescing();
        if shift {
            if self.selection_anchor.is_none() {
                self.selection_anchor = Some(self.cursor_char);
            }
        } else {
            self.selection_anchor = None;
        }
    }

    /// Delete the current selection, collapsing the caret to the range start.
    /// Returns `true` if anything was removed. One undo group unless already
    /// inside a transaction.
    fn delete_selection(&mut self) -> bool {
        if let Some((s, e)) = self.selection_range() {
            self.buffer.remove_char_range(s, e);
            self.cursor_char = s;
            self.selection_anchor = None;
            self.update_goal_col();
            true
        } else {
            false
        }
    }

    /// Replace the selection (if any) with `text`, else insert at the caret. A
    /// replacement is one atomic undo group (delete + insert). Used by paste and
    /// by typing/Enter/Tab when a selection is active.
    fn replace_selection_with(&mut self, text: &str) {
        if self.selection_range().is_some() {
            self.buffer.begin_transaction();
            self.delete_selection();
            self.buffer.insert_str(self.cursor_char, text);
            self.cursor_char += text.chars().count();
            self.buffer.end_transaction();
        } else {
            self.buffer.insert_str(self.cursor_char, text);
            self.cursor_char += text.chars().count();
        }
        self.selection_anchor = None;
        self.update_goal_col();
    }

    /// Select the whole buffer (Ctrl+A): anchor at 0, head at the end.
    fn select_all(&mut self) {
        self.buffer.break_coalescing();
        self.selection_anchor = Some(0);
        self.cursor_char = self.buffer.len_chars();
        self.update_goal_col();
    }

    /// Undo one group; move the caret to the returned op site and drop any
    /// selection. Returns `true` if the buffer changed.
    fn do_undo(&mut self) -> bool {
        match self.buffer.undo() {
            Some(pos) => {
                self.cursor_char = pos;
                self.selection_anchor = None;
                self.update_goal_col();
                true
            }
            None => false,
        }
    }

    /// Redo one group; symmetric to [`App::do_undo`].
    fn do_redo(&mut self) -> bool {
        match self.buffer.redo() {
            Some(pos) => {
                self.cursor_char = pos;
                self.selection_anchor = None;
                self.update_goal_col();
                true
            }
            None => false,
        }
    }

    /// Write the buffer to disk (Ctrl+S). Scratch buffers have no path yet.
    fn do_save(&mut self) {
        match self.buffer.save() {
            Ok(true) => {}
            Ok(false) => eprintln!("umber: no path to save (scratch buffer)"),
            Err(err) => eprintln!("umber: save failed: {err}"),
        }
    }

    /// Copy the selection to the system clipboard (no-op without a selection or
    /// clipboard).
    fn clipboard_copy(&mut self) {
        let (s, e) = match self.selection_range() {
            Some(r) => r,
            None => return,
        };
        let text = self.buffer.slice_chars(s, e);
        match self.clipboard.as_mut() {
            Some(cb) => {
                if let Err(err) = cb.set_text(text) {
                    eprintln!("umber: clipboard copy failed: {err}");
                }
            }
            None => eprintln!("umber: clipboard unavailable"),
        }
    }

    /// Copy then delete the selection (Ctrl+X). Returns `true` if the buffer
    /// changed.
    fn clipboard_cut(&mut self) -> bool {
        if self.selection_range().is_none() {
            return false;
        }
        self.clipboard_copy();
        self.delete_selection()
    }

    /// Paste clipboard text over the selection (Ctrl+V). Returns `true` if the
    /// buffer changed.
    fn clipboard_paste(&mut self) -> bool {
        let text = match self.clipboard.as_mut() {
            Some(cb) => match cb.get_text() {
                Ok(t) => t,
                Err(err) => {
                    eprintln!("umber: clipboard paste failed: {err}");
                    return false;
                }
            },
            None => {
                eprintln!("umber: clipboard unavailable");
                return false;
            }
        };
        if text.is_empty() {
            return false;
        }
        self.replace_selection_with(&text);
        true
    }

    /// Set `ControlFlow` to the earliest pending wake (idle-RSS sample or the
    /// scrollbar auto-hide), or `Wait` when nothing is pending. Coexists with
    /// the existing RSS `WaitUntil` timer instead of clobbering it.
    /// `now` is supplied by the caller so expiry decisions here agree exactly
    /// with the caller's own checks (a fresh `Instant::now()` could land past a
    /// deadline the caller judged still-pending, leaving no wake scheduled and
    /// the scrollbar painted until the next external event).
    fn reschedule(&self, event_loop: &ActiveEventLoop, now: Instant) {
        let mut earliest: Option<Instant> = None;
        if self.first_frame && !self.rss_printed {
            if let Some(t0) = self.first_frame_at {
                earliest = min_deadline(earliest, t0 + Duration::from_secs(2));
            }
        }
        if !self.scrollbar_dragging {
            if let Some(d) = self.scrollbar_deadline {
                if d > now {
                    earliest = min_deadline(earliest, d);
                }
            }
        }
        match earliest {
            Some(t) => event_loop.set_control_flow(ControlFlow::WaitUntil(t)),
            None => event_loop.set_control_flow(ControlFlow::Wait),
        }
    }

    // ===================================================================
    //  Slice 2: config live-apply, modal views, command dispatch.
    // ===================================================================

    /// Push the current config into the renderer + event loop and re-render the
    /// editor view. Called at startup and after any config/feature change so
    /// font size, line height, gutter, latency HUD, and scrollbar settings take
    /// effect live (font/line rebuild renderer metrics like a scale change).
    fn apply_config(&mut self) {
        self.scrollbar_linger = Duration::from_millis(self.config.scrollbar_linger_ms);
        if !self.config.scrollbar {
            self.scrollbar_deadline = None;
            self.scrollbar_dragging = false;
        }
        if let Some(r) = self.renderer.as_mut() {
            r.set_metrics(self.config.font_size, self.config.line_height);
            r.set_gutter_enabled(self.config.gutter);
            r.set_latency_hud(self.config.latency_hud);
        }
        self.apply_view(true);
    }

    /// Rebuild the overlay spec for the current view and hand it to the renderer
    /// (or clear it in the editor), then request a redraw. All modal text is
    /// shaped here (the state-change path), never in `render`.
    fn refresh_overlay(&mut self) {
        let spec = self.build_overlay_spec();
        if let Some(r) = self.renderer.as_mut() {
            r.set_overlay(spec);
            r.window().request_redraw();
        }
    }

    /// Build the overlay spec for the current modal, or `None` for the editor.
    fn build_overlay_spec(&self) -> Option<OverlaySpec> {
        match self.view {
            View::Editor => None,
            View::Palette => {
                let cap = self
                    .renderer
                    .as_ref()
                    .map(|r| r.overlay_row_capacity())
                    .unwrap_or(1);
                let n = self.palette_filtered.len();
                let sel = if n == 0 {
                    0
                } else {
                    self.palette_sel.min(n - 1)
                };
                let start = if sel < cap { 0 } else { sel + 1 - cap };
                let end = (start + cap).min(n);
                let mut rows = Vec::with_capacity(end - start);
                for &ci in &self.palette_filtered[start..end] {
                    let c = self.commands.commands()[ci];
                    rows.push((c.title.to_string(), c.keybinding.to_string()));
                }
                Some(OverlaySpec {
                    title: None,
                    input: Some(format!("> {}", self.palette_query)),
                    rows,
                    left_color: [225, 225, 230],
                    right_color: [135, 135, 150],
                    split_frac: 0.62,
                    selected: if n == 0 { None } else { Some(sel - start) },
                    hint: Some(format!(
                        "{n} commands  \u{2014}  \u{2191}\u{2193} select \u{2022} Enter run \u{2022} Esc close"
                    )),
                })
            }
            View::Settings => {
                let c = &self.config;
                let rows = vec![
                    ("Font size (px)".to_string(), format!("{}", c.font_size)),
                    ("Line height (px)".to_string(), format!("{}", c.line_height)),
                    (
                        "Scrollbar linger (ms)".to_string(),
                        format!("{}", c.scrollbar_linger_ms),
                    ),
                    ("Line-number gutter".to_string(), onoff(c.gutter)),
                    ("Overlay scrollbar".to_string(), onoff(c.scrollbar)),
                    ("Latency HUD".to_string(), onoff(c.latency_hud)),
                ];
                Some(OverlaySpec {
                    title: Some("Preferences \u{2014} Settings".to_string()),
                    input: None,
                    rows,
                    left_color: [150, 150, 162],
                    right_color: [228, 228, 234],
                    split_frac: 0.5,
                    selected: Some(self.settings_sel),
                    hint: Some(
                        "\u{2191}\u{2193} select \u{2022} \u{2190}/\u{2192} or +/- adjust \u{2022} Enter toggle \u{2022} Esc save & close"
                            .to_string(),
                    ),
                })
            }
            View::Modules => {
                let mut rows = Vec::new();
                for f in self.features.features() {
                    let state = if f.enabled { "ON " } else { "OFF" };
                    let tag = if f.removable { "" } else { "  [kernel]" };
                    rows.push((
                        f.name.to_string(),
                        format!("{state}  \u{2022}  {}{tag}", f.description),
                    ));
                }
                let hint = self.modules_hint.clone().unwrap_or_else(|| {
                    "\u{2191}\u{2193} select \u{2022} Enter toggle \u{2022} Esc save & close"
                        .to_string()
                });
                Some(OverlaySpec {
                    title: Some("Modules \u{2014} Manage".to_string()),
                    input: None,
                    rows,
                    left_color: [225, 225, 230],
                    right_color: [150, 150, 162],
                    split_frac: 0.30,
                    selected: Some(self.modules_sel),
                    hint: Some(hint),
                })
            }
        }
    }

    /// Open the command palette (Ctrl+Shift+P, D6).
    fn open_palette(&mut self) {
        self.view = View::Palette;
        self.palette_query.clear();
        self.palette_sel = 0;
        self.palette_filtered = self.commands.filter("");
        self.refresh_overlay();
    }

    /// Open the settings page (Ctrl+, / "Preferences: Open Settings").
    fn open_settings(&mut self) {
        self.view = View::Settings;
        self.settings_sel = 0;
        self.refresh_overlay();
    }

    /// Open the modules page ("Modules: Manage").
    fn open_modules(&mut self) {
        self.view = View::Modules;
        self.modules_sel = 0;
        self.modules_hint = None;
        self.refresh_overlay();
    }

    /// Return to the editor, clearing any overlay and repainting.
    fn close_overlay(&mut self) {
        self.view = View::Editor;
        self.apply_view(false);
        if let Some(r) = self.renderer.as_mut() {
            r.set_overlay(None);
            r.window().request_redraw();
        }
    }

    /// Recompute the palette filter after the query changed.
    fn repalette(&mut self) {
        self.palette_filtered = self.commands.filter(&self.palette_query);
        self.palette_sel = 0;
        self.refresh_overlay();
    }

    /// Command palette keyboard handling (captures all input while open).
    fn palette_key(&mut self, event: KeyEvent, event_loop: &ActiveEventLoop) {
        match &event.logical_key {
            Key::Named(NamedKey::Escape) => {
                self.close_overlay();
                return;
            }
            Key::Named(NamedKey::Enter) => {
                let id = self
                    .palette_filtered
                    .get(self.palette_sel)
                    .map(|&i| self.commands.commands()[i].id);
                self.view = View::Editor;
                match id {
                    Some(id) => self.execute_command(id, event_loop),
                    None => self.close_overlay(),
                }
                return;
            }
            Key::Named(NamedKey::ArrowDown) => {
                let n = self.palette_filtered.len();
                if n > 0 {
                    self.palette_sel = (self.palette_sel + 1) % n;
                }
                self.refresh_overlay();
                return;
            }
            Key::Named(NamedKey::ArrowUp) => {
                let n = self.palette_filtered.len();
                if n > 0 {
                    self.palette_sel = (self.palette_sel + n - 1) % n;
                }
                self.refresh_overlay();
                return;
            }
            Key::Named(NamedKey::Backspace) => {
                self.palette_query.pop();
                self.repalette();
                return;
            }
            _ => {}
        }
        if let Some(text) = &event.text {
            let mut added = false;
            for ch in text.chars() {
                if !ch.is_control() {
                    self.palette_query.push(ch);
                    added = true;
                }
            }
            if added {
                self.repalette();
            }
        }
    }

    /// Settings page keyboard handling.
    fn settings_key(&mut self, event: KeyEvent, _event_loop: &ActiveEventLoop) {
        match &event.logical_key {
            Key::Named(NamedKey::Escape) => {
                let _ = self.config.save();
                self.close_overlay();
            }
            Key::Named(NamedKey::ArrowUp) => {
                self.settings_sel = self.settings_sel.saturating_sub(1);
                self.refresh_overlay();
            }
            Key::Named(NamedKey::ArrowDown) => {
                self.settings_sel = (self.settings_sel + 1).min(SETTINGS_ROWS - 1);
                self.refresh_overlay();
            }
            Key::Named(NamedKey::Enter) => {
                // Enter toggles booleans; numeric rows ignore it.
                if self.settings_sel >= 3 {
                    self.settings_adjust(1);
                }
            }
            Key::Named(NamedKey::ArrowLeft) => self.settings_adjust(-1),
            Key::Named(NamedKey::ArrowRight) => self.settings_adjust(1),
            _ => {
                if let Some(text) = &event.text {
                    match text.as_str() {
                        "+" | "=" => self.settings_adjust(1),
                        "-" | "_" => self.settings_adjust(-1),
                        _ => {}
                    }
                }
            }
        }
    }

    /// Apply a +/- step to the selected setting, then persist + live-apply.
    fn settings_adjust(&mut self, dir: i32) {
        match self.settings_sel {
            0 => {
                self.config.font_size = (self.config.font_size + dir as f32)
                    .clamp(umber_kernel::FONT_MIN, umber_kernel::FONT_MAX);
            }
            1 => {
                self.config.line_height = (self.config.line_height + dir as f32)
                    .clamp(umber_kernel::LINE_MIN, umber_kernel::LINE_MAX);
            }
            2 => {
                let v = self.config.scrollbar_linger_ms as i64 + dir as i64 * 100;
                self.config.scrollbar_linger_ms = v.clamp(
                    umber_kernel::LINGER_MIN as i64,
                    umber_kernel::LINGER_MAX as i64,
                ) as u64;
            }
            3 => self.config.gutter = !self.config.gutter,
            4 => self.config.scrollbar = !self.config.scrollbar,
            5 => self.config.latency_hud = !self.config.latency_hud,
            _ => {}
        }
        // Keep the feature registry in step with the config booleans.
        self.features = FeatureRegistry::from_config(&self.config);
        let _ = self.config.save();
        self.apply_config();
        self.refresh_overlay();
    }

    /// Modules page keyboard handling.
    fn modules_key(&mut self, event: KeyEvent, _event_loop: &ActiveEventLoop) {
        match &event.logical_key {
            Key::Named(NamedKey::Escape) => {
                let _ = self.config.save();
                self.close_overlay();
            }
            Key::Named(NamedKey::ArrowUp) => {
                self.modules_sel = self.modules_sel.saturating_sub(1);
                self.modules_hint = None;
                self.refresh_overlay();
            }
            Key::Named(NamedKey::ArrowDown) => {
                let n = self.features.features().len();
                self.modules_sel = (self.modules_sel + 1).min(n.saturating_sub(1));
                self.modules_hint = None;
                self.refresh_overlay();
            }
            Key::Named(NamedKey::Enter) => self.modules_toggle_current(),
            _ => {}
        }
    }

    /// Toggle the selected feature (D10). Kernel entries refuse with a hint.
    fn modules_toggle_current(&mut self) {
        match self.features.toggle(self.modules_sel) {
            Ok(_) => {
                self.modules_hint = None;
                self.features.apply_to_config(&mut self.config);
                let _ = self.config.save();
                self.apply_config();
            }
            Err(hint) => self.modules_hint = Some(hint.to_string()),
        }
        self.refresh_overlay();
    }

    /// Toggle a feature by id (from a palette command). Kernel entries no-op,
    /// leaving a hint for the modules page.
    fn toggle_feature(&mut self, id: &str) {
        if let Some(idx) = self.features.index_of(id) {
            match self.features.toggle(idx) {
                Ok(_) => {
                    self.features.apply_to_config(&mut self.config);
                    let _ = self.config.save();
                    self.apply_config();
                }
                Err(hint) => self.modules_hint = Some(hint.to_string()),
            }
        }
    }

    /// Run a registered command by id. Commands that open a modal switch the
    /// view and return; in-place commands run and drop back to the editor.
    fn execute_command(&mut self, id: &str, event_loop: &ActiveEventLoop) {
        // Commands that move the caret must scroll the view to it after the
        // overlay closes (matching the apply_view(true) their keyboard paths
        // use) — close_overlay alone would leave the viewport behind.
        let mut follow = false;
        match id {
            "view.commandPalette" => {
                self.open_palette();
                return;
            }
            "view.settings" => {
                self.open_settings();
                return;
            }
            "view.modules" => {
                self.open_modules();
                return;
            }
            "app.quit" => {
                event_loop.exit();
                return;
            }
            "file.save" => self.do_save(),
            "edit.undo" => {
                self.do_undo();
                follow = true;
            }
            "edit.redo" => {
                self.do_redo();
                follow = true;
            }
            "edit.copy" => self.clipboard_copy(),
            "edit.cut" => {
                self.clipboard_cut();
                follow = true;
            }
            "edit.paste" => {
                self.clipboard_paste();
                follow = true;
            }
            "edit.selectAll" => self.select_all(),
            "goto.fileStart" => {
                self.buffer.break_coalescing();
                self.selection_anchor = None;
                self.cursor_char = 0;
                self.update_goal_col();
                follow = true;
            }
            "goto.fileEnd" => {
                self.buffer.break_coalescing();
                self.selection_anchor = None;
                self.cursor_char = self.buffer.len_chars();
                self.update_goal_col();
                follow = true;
            }
            "view.toggle.gutter" => self.toggle_feature("gutter"),
            "view.toggle.scrollbar" => self.toggle_feature("scrollbar"),
            "view.toggle.latencyHud" => self.toggle_feature("latency-hud"),
            _ => {}
        }
        // In-place command finished: return to the editor and repaint.
        self.close_overlay();
        if follow {
            self.apply_view(true);
        }
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
        // Push config metrics/toggles into the fresh renderer, then draw.
        self.apply_config();
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
                // Modal overlays are shaped to the surface width at set_overlay
                // time; a resize while one is open must re-spec it or its text
                // stays laid out for the old geometry.
                if self.view != View::Editor {
                    self.refresh_overlay();
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
                if self.view != View::Editor {
                    return;
                }
                // Scroll is a P0 exit-criterion path (100 MB fixture), so it
                // feeds the D4 latency ring exactly like keystrokes do. It also
                // reveals the overlay scrollbar (Ghostty-style).
                let t = Instant::now();
                self.poke_scrollbar();
                let lines = match delta {
                    MouseScrollDelta::LineDelta(_, y) => (-y * WHEEL_LINES) as i64,
                    MouseScrollDelta::PixelDelta(p) => (-p.y / BASE_LINE_PX) as i64,
                };
                if lines != 0 {
                    self.scroll_by(lines);
                    self.apply_view(false);
                    if let Some(renderer) = self.renderer.as_mut() {
                        renderer.mark_keystroke(t);
                    }
                }
                if let Some(renderer) = self.renderer.as_ref() {
                    renderer.window().request_redraw();
                }
            }

            WindowEvent::CursorMoved { position, .. } => {
                self.pointer = (position.x, position.y);
                if self.scrollbar_dragging {
                    self.drag_scrollbar(position.y);
                } else if self.selecting {
                    // Drag-extend the selection. Throttle: only re-render when the
                    // mapped char actually changes, not on raw mouse motion.
                    if let Some(pos) = self.pointer_to_char() {
                        if pos != self.cursor_char {
                            self.cursor_char = pos;
                            self.update_goal_col();
                            self.apply_view(true);
                            if let Some(renderer) = self.renderer.as_ref() {
                                renderer.window().request_redraw();
                            }
                        }
                    }
                } else if self.pointer_in_scrollbar_zone() {
                    // Only the hidden->visible transition needs a frame; while
                    // already visible, hovering just extends the linger timer
                    // (no geometry change, so no redraw — a redraw per
                    // CursorMoved would be a full-frame storm).
                    let was_visible = self.scrollbar_visible(Instant::now());
                    self.poke_scrollbar();
                    if !was_visible {
                        if let Some(renderer) = self.renderer.as_ref() {
                            renderer.window().request_redraw();
                        }
                    }
                }
            }

            WindowEvent::MouseInput { state, button, .. } => {
                if self.view != View::Editor {
                    return;
                }
                if button != MouseButton::Left {
                    return;
                }
                match state {
                    ElementState::Pressed => {
                        let t = Instant::now();
                        // Scrollbar interaction wins over text placement.
                        if self.try_scrollbar_press() {
                            self.poke_scrollbar();
                            if let Some(renderer) = self.renderer.as_ref() {
                                renderer.window().request_redraw();
                            }
                            return;
                        }
                        // Text press: place the caret and set the selection
                        // anchor. Shift extends from the existing anchor/caret; a
                        // plain press collapses (anchor == caret) and arms a drag.
                        // Marked in the D4 ring like a keystroke.
                        if let Some(pos) = self.pointer_to_char() {
                            let shift = self.modifiers.shift_key();
                            self.buffer.break_coalescing();
                            if shift {
                                if self.selection_anchor.is_none() {
                                    self.selection_anchor = Some(self.cursor_char);
                                }
                            } else {
                                self.selection_anchor = Some(pos);
                            }
                            self.cursor_char = pos;
                            self.selecting = true;
                            self.update_goal_col();
                            self.apply_view(true);
                            if let Some(renderer) = self.renderer.as_mut() {
                                renderer.mark_keystroke(t);
                                renderer.window().request_redraw();
                            }
                        }
                    }
                    ElementState::Released => {
                        self.selecting = false;
                        if self.scrollbar_dragging {
                            self.scrollbar_dragging = false;
                            // Start the linger countdown now the drag ended.
                            self.poke_scrollbar();
                            if let Some(renderer) = self.renderer.as_ref() {
                                renderer.window().request_redraw();
                            }
                        }
                    }
                }
            }

            WindowEvent::KeyboardInput { event, .. } => {
                if event.state != ElementState::Pressed {
                    return;
                }
                // Slice 2 dispatch: modals capture all input while open; the
                // editor path below runs only in the editor view.
                match self.view {
                    View::Editor => {}
                    View::Palette => {
                        self.palette_key(event, event_loop);
                        return;
                    }
                    View::Settings => {
                        self.settings_key(event, event_loop);
                        return;
                    }
                    View::Modules => {
                        self.modules_key(event, event_loop);
                        return;
                    }
                }
                // Timestamp at event receipt — the head of the keystroke->present
                // latency measurement (D4).
                let t = Instant::now();
                let ctrl = self.modifiers.control_key();
                let shift = self.modifiers.shift_key();
                let len = self.buffer.len_chars();
                // `changed` = buffer content changed (feeds the D4 latency ring);
                // `redraw_only` = view/banner changed without an edit (selection,
                // save marker) and just needs a repaint.
                let mut changed = false;
                let mut redraw_only = false;

                // Ctrl chords: clipboard, undo/redo, save, select-all. These
                // consume the key; the printable path below is already Ctrl-gated.
                if ctrl {
                    if let Key::Character(c) = &event.logical_key {
                        match c.to_lowercase().as_str() {
                            "p" if shift => {
                                self.open_palette();
                                return;
                            }
                            "," => {
                                self.open_settings();
                                return;
                            }
                            "q" => {
                                event_loop.exit();
                                return;
                            }
                            "a" => {
                                self.select_all();
                                redraw_only = true;
                            }
                            "c" => self.clipboard_copy(),
                            "x" => changed = self.clipboard_cut(),
                            "v" => changed = self.clipboard_paste(),
                            "z" => {
                                changed = if shift {
                                    self.do_redo()
                                } else {
                                    self.do_undo()
                                };
                            }
                            "y" => changed = self.do_redo(),
                            "s" => {
                                self.do_save();
                                redraw_only = true;
                            }
                            _ => {}
                        }
                    }
                }

                match &event.logical_key {
                    Key::Named(NamedKey::Backspace) => {
                        if self.selection_range().is_some() {
                            changed = self.delete_selection();
                        } else if self.cursor_char > 0 {
                            self.buffer.break_coalescing();
                            self.buffer
                                .remove_char_range(self.cursor_char - 1, self.cursor_char);
                            self.cursor_char -= 1;
                            self.update_goal_col();
                            changed = true;
                        }
                    }
                    Key::Named(NamedKey::Delete) => {
                        if self.selection_range().is_some() {
                            changed = self.delete_selection();
                        } else if self.cursor_char < len {
                            self.buffer.break_coalescing();
                            self.buffer
                                .remove_char_range(self.cursor_char, self.cursor_char + 1);
                            changed = true;
                        }
                    }
                    Key::Named(NamedKey::Enter) => {
                        if self.selection_range().is_some() {
                            self.replace_selection_with("\n");
                        } else {
                            self.buffer.insert_char(self.cursor_char, '\n');
                            self.cursor_char += 1;
                            self.update_goal_col();
                        }
                        changed = true;
                    }
                    Key::Named(NamedKey::Tab) => {
                        if self.selection_range().is_some() {
                            self.replace_selection_with("\t");
                        } else {
                            self.buffer.insert_char(self.cursor_char, '\t');
                            self.cursor_char += 1;
                            self.update_goal_col();
                        }
                        changed = true;
                    }
                    Key::Named(NamedKey::ArrowLeft) => {
                        self.begin_move(shift);
                        self.cursor_char = self.cursor_char.saturating_sub(1);
                        self.update_goal_col();
                        changed = true;
                    }
                    Key::Named(NamedKey::ArrowRight) => {
                        self.begin_move(shift);
                        self.cursor_char = (self.cursor_char + 1).min(len);
                        self.update_goal_col();
                        changed = true;
                    }
                    Key::Named(NamedKey::ArrowUp) => {
                        self.begin_move(shift);
                        self.move_vertical(-1);
                        changed = true;
                    }
                    Key::Named(NamedKey::ArrowDown) => {
                        self.begin_move(shift);
                        self.move_vertical(1);
                        changed = true;
                    }
                    Key::Named(NamedKey::Home) => {
                        self.begin_move(shift);
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
                        self.begin_move(shift);
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
                        // Moves the caret a page (and the view follows) so
                        // Shift+PageUp can extend the selection.
                        self.begin_move(shift);
                        let cap = self.page();
                        self.move_vertical(-(cap as i64));
                        changed = true;
                    }
                    Key::Named(NamedKey::PageDown) => {
                        self.begin_move(shift);
                        let cap = self.page();
                        self.move_vertical(cap as i64);
                        changed = true;
                    }
                    _ => {}
                }

                // Printable input arrives as `event.text` (layout-resolved).
                // Skip when Ctrl is held so chords don't type their letter, and
                // skip control chars (Enter/Tab are handled as named keys). A
                // selection is replaced atomically; otherwise chars insert with
                // typing-coalesced undo.
                if !ctrl {
                    if let Some(text) = &event.text {
                        if self.selection_range().is_some() {
                            let s: String = text.chars().filter(|c| !c.is_control()).collect();
                            if !s.is_empty() {
                                self.replace_selection_with(&s);
                                changed = true;
                            }
                        } else {
                            let mut typed = false;
                            for ch in text.chars() {
                                if !ch.is_control() {
                                    self.buffer.insert_char(self.cursor_char, ch);
                                    self.cursor_char += 1;
                                    typed = true;
                                }
                            }
                            if typed {
                                self.selection_anchor = None;
                                self.update_goal_col();
                                changed = true;
                            }
                        }
                    }
                }

                if changed || redraw_only {
                    self.apply_view(true);
                    if let Some(renderer) = self.renderer.as_mut() {
                        if changed {
                            renderer.mark_keystroke(t);
                        }
                        renderer.window().request_redraw();
                    }
                }
            }

            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let total = self.buffer.len_lines();
                let first = self.first_visible_line;
                let want_scrollbar = self.view == View::Editor
                    && self.config.scrollbar
                    && self.scrollbar_visible(now);
                let presented;
                let drew_scrollbar;
                match self.renderer.as_mut() {
                    Some(renderer) => {
                        let cap = renderer.visible_line_capacity();
                        let info = if want_scrollbar && total > cap {
                            Some(ScrollbarInfo {
                                first_line: first,
                                total_lines: total,
                            })
                        } else {
                            None
                        };
                        drew_scrollbar = info.is_some();
                        renderer.set_scrollbar(info);
                        presented = renderer.render();
                    }
                    None => return,
                }
                self.scrollbar_drawn = drew_scrollbar;
                if presented && !self.first_frame {
                    self.first_frame = true;
                    self.first_frame_at = Some(now);
                    println!(
                        "cold-start: {:.1} ms (main entry -> first frame presented)",
                        self.start.elapsed().as_secs_f64() * 1000.0
                    );
                }
                // Reschedule after updating the RSS timer + scrollbar state so
                // the idle-RSS `WaitUntil` and the scrollbar hide coexist.
                self.reschedule(event_loop, now);
            }

            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        let now = Instant::now();
        if self.first_frame && !self.rss_printed {
            if let Some(t0) = self.first_frame_at {
                if now.duration_since(t0) >= Duration::from_secs(2) {
                    match read_vmrss() {
                        Some(rss) => println!("idle RAM (VmRSS): {rss}"),
                        None => println!("idle RAM (VmRSS): unavailable"),
                    }
                    self.rss_printed = true;
                }
            }
        }
        // The scrollbar lingered out: request one more frame to erase it.
        if self.scrollbar_drawn && !self.scrollbar_visible(now) {
            if let Some(renderer) = self.renderer.as_ref() {
                renderer.window().request_redraw();
            }
        }
        self.reschedule(event_loop, now);
    }
}

/// Number of decimal digits in `n` (min 1, so 0 -> 1). Sizes the gutter column
/// from the whole file's last line number.
fn digit_count(n: usize) -> usize {
    let mut digits = 1;
    let mut v = n;
    while v >= 10 {
        v /= 10;
        digits += 1;
    }
    digits
}

/// The earlier of an optional current deadline and a candidate.
fn min_deadline(current: Option<Instant>, candidate: Instant) -> Option<Instant> {
    Some(match current {
        Some(c) => c.min(candidate),
        None => candidate,
    })
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
