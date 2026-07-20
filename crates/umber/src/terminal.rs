//! Embedded terminal session (P3): a PTY running `$SHELL`, parsed into an
//! alacritty_terminal grid on a background reader thread.
//!
//! Threading model: alacritty_terminal's own [`PtyEventLoop`] thread owns the
//! PTY fd and feeds the shared `FairMutex<Term>` parser. UI wakeups cross back
//! through [`TermNotifier`] (implemented over winit's `EventLoopProxy` by the
//! bin, and over plain atomics by the headless tests). PTY reads never happen
//! on the UI thread, and the UI never blocks on the PTY.
//!
//! Wakeup coalescing: [`EventProxy::send_event`] sets `dirty` and forwards a
//! wakeup ONLY on the false->true transition, so an output flood (`yes`) keeps
//! at most one UI wakeup in flight. The consumer must call
//! [`TerminalSession::take_dirty`] (which clears the flag) BEFORE reading the
//! grid: any parser progress that lands after the clear re-arms a fresh
//! wakeup, so no update can be lost between clear and read.
//!
//! Fidelity: visible text, cursor position, ANSI foreground colors, bold,
//! italic, and dim attributes. Cell backgrounds and Title/Clipboard events
//! remain separate follow-ups.

use std::borrow::Cow;
use std::io;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use alacritty_terminal::event::{Event as TermEvent, EventListener, WindowSize};
use alacritty_terminal::event_loop::{EventLoop as PtyEventLoop, EventLoopSender, Msg, State};
use alacritty_terminal::grid::{Dimensions, Scroll};
use alacritty_terminal::sync::FairMutex;
use alacritty_terminal::term::cell::Flags;
use alacritty_terminal::term::{Config as TermConfig, Term};
use alacritty_terminal::tty::{self, Options as PtyOptions, Pty, Shell};
use alacritty_terminal::vte::ansi::{Color, NamedColor, Rgb};

/// Grid dimensions for `Term` construction/resize. alacritty's own `TermSize`
/// helper lives in its `term::test` module, so we carry this trivial impl
/// instead of depending on test helpers in production code.
#[derive(Copy, Clone)]
struct GridSize {
    columns: usize,
    screen_lines: usize,
}

impl Dimensions for GridSize {
    fn total_lines(&self) -> usize {
        self.screen_lines
    }
    fn screen_lines(&self) -> usize {
        self.screen_lines
    }
    fn columns(&self) -> usize {
        self.columns
    }
}

/// How the session pokes the UI (or a test harness) from the reader thread.
pub trait TermNotifier: Clone + Send + 'static {
    /// New grid content is available (already coalesced — see module docs).
    fn wake(&self);
    /// The shell child exited.
    fn child_exited(&self);
}

/// [`EventListener`] bridging alacritty_terminal's events to a [`TermNotifier`].
pub struct EventProxy<N: TermNotifier> {
    notifier: N,
    dirty: Arc<AtomicBool>,
    exited: Arc<AtomicBool>,
    /// Write half for PTY responses the *parser* requests (e.g. a program
    /// queried cursor position). Filled in after the event loop exists —
    /// creation order forces the `Option`.
    writer: Arc<Mutex<Option<EventLoopSender>>>,
}

impl<N: TermNotifier> Clone for EventProxy<N> {
    fn clone(&self) -> Self {
        Self {
            notifier: self.notifier.clone(),
            dirty: self.dirty.clone(),
            exited: self.exited.clone(),
            writer: self.writer.clone(),
        }
    }
}

impl<N: TermNotifier> EventListener for EventProxy<N> {
    fn send_event(&self, event: TermEvent) {
        match event {
            TermEvent::Wakeup => {
                // Coalesce: only the false->true transition crosses threads.
                if !self.dirty.swap(true, Ordering::AcqRel) {
                    self.notifier.wake();
                }
            }
            TermEvent::PtyWrite(text) => {
                if let Ok(guard) = self.writer.lock() {
                    if let Some(sender) = guard.as_ref() {
                        let _ = sender.send(Msg::Input(Cow::Owned(text.into_bytes())));
                    }
                }
            }
            TermEvent::ChildExit(_) | TermEvent::Exit => {
                if !self.exited.swap(true, Ordering::AcqRel) {
                    self.notifier.child_exited();
                }
            }
            // Title/clipboard/color/size-request fidelity: TODO(P3).
            _ => {}
        }
    }
}

/// Rich-text style over a UTF-8 byte range in a terminal snapshot.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TerminalSpan {
    pub start: usize,
    pub end: usize,
    pub rgb: [u8; 3],
    pub bold: bool,
    pub italic: bool,
}

/// Visible terminal snapshot, including styling retained from ANSI cells.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TerminalSnapshot {
    pub text: String,
    pub cursor: Option<(usize, usize)>,
    pub spans: Vec<TerminalSpan>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CellStyle {
    rgb: [u8; 3],
    bold: bool,
    italic: bool,
}

fn indexed_rgb(index: u8) -> [u8; 3] {
    const ANSI: [[u8; 3]; 16] = [
        [35, 31, 28],
        [205, 92, 72],
        [139, 168, 116],
        [220, 174, 92],
        [112, 145, 190],
        [184, 128, 170],
        [105, 170, 176],
        [218, 211, 199],
        [112, 104, 94],
        [231, 116, 82],
        [164, 190, 130],
        [238, 194, 108],
        [135, 164, 208],
        [204, 148, 192],
        [126, 190, 194],
        [242, 236, 224],
    ];
    match index {
        0..=15 => ANSI[index as usize],
        16..=231 => {
            let n = index - 16;
            let level = |v: u8| if v == 0 { 0 } else { 55 + v * 40 };
            [level(n / 36), level((n / 6) % 6), level(n % 6)]
        }
        _ => {
            let v = 8 + (index - 232) * 10;
            [v, v, v]
        }
    }
}

fn named_index(named: NamedColor) -> Option<u8> {
    use NamedColor::*;
    Some(match named {
        Black => 0,
        Red => 1,
        Green => 2,
        Yellow => 3,
        Blue => 4,
        Magenta => 5,
        Cyan => 6,
        White => 7,
        BrightBlack => 8,
        BrightRed => 9,
        BrightGreen => 10,
        BrightYellow => 11,
        BrightBlue => 12,
        BrightMagenta => 13,
        BrightCyan => 14,
        BrightWhite => 15,
        DimBlack => 0,
        DimRed => 1,
        DimGreen => 2,
        DimYellow => 3,
        DimBlue => 4,
        DimMagenta => 5,
        DimCyan => 6,
        DimWhite => 7,
        _ => return None,
    })
}

fn fallback_named(named: NamedColor) -> [u8; 3] {
    match named {
        NamedColor::Foreground | NamedColor::BrightForeground => [220, 214, 201],
        NamedColor::DimForeground => [145, 137, 124],
        NamedColor::Background | NamedColor::Cursor => [20, 17, 14],
        _ => named_index(named)
            .map(indexed_rgb)
            .unwrap_or([220, 214, 201]),
    }
}

fn resolve_color(color: Color, colors: &alacritty_terminal::term::color::Colors) -> [u8; 3] {
    let rgb = match color {
        Color::Spec(rgb) => rgb,
        Color::Indexed(index) => colors[index as usize].unwrap_or_else(|| {
            let [r, g, b] = indexed_rgb(index);
            Rgb { r, g, b }
        }),
        Color::Named(named) => colors[named].unwrap_or_else(|| {
            let [r, g, b] = fallback_named(named);
            Rgb { r, g, b }
        }),
    };
    [rgb.r, rgb.g, rgb.b]
}

/// A live terminal: shell child, PTY reader thread, shared parsed grid.
pub struct TerminalSession<N: TermNotifier> {
    term: Arc<FairMutex<Term<EventProxy<N>>>>,
    notifier: N,
    sender: EventLoopSender,
    dirty: Arc<AtomicBool>,
    exited: Arc<AtomicBool>,
    io_thread: Option<std::thread::JoinHandle<(PtyEventLoop<Pty, EventProxy<N>>, State)>>,
}

impl<N: TermNotifier> TerminalSession<N> {
    /// Spawn the user's default shell in a PTY of `cols` x `lines` cells: on
    /// unix `$SHELL` (fallback `/bin/sh`), on Windows `%ComSpec%` (fallback
    /// `cmd.exe`) driven over ConPTY.
    pub fn spawn(
        notifier: N,
        cols: usize,
        lines: usize,
        cell_width: u16,
        cell_height: u16,
    ) -> io::Result<Self> {
        Self::spawn_with_shell(notifier, cols, lines, cell_width, cell_height, None)
    }

    /// [`TerminalSession::spawn`] with an explicit shell program + args
    /// (used by the e2e tests to run deterministic commands).
    pub fn spawn_with_shell(
        notifier: N,
        cols: usize,
        lines: usize,
        cell_width: u16,
        cell_height: u16,
        shell: Option<(String, Vec<String>)>,
    ) -> io::Result<Self> {
        let (program, args) = shell.unwrap_or_else(|| {
            // Default shell is platform-specific. alacritty_terminal drives the
            // Windows program over ConPTY; unix keeps the prior $SHELL behavior.
            #[cfg(windows)]
            let program = std::env::var("ComSpec").unwrap_or_else(|_| "cmd.exe".to_string());
            #[cfg(not(windows))]
            let program = std::env::var("SHELL").unwrap_or_else(|_| "/bin/sh".to_string());
            (program, Vec::new())
        });
        let options = PtyOptions {
            shell: Some(Shell::new(program, args)),
            ..PtyOptions::default()
        };
        let window_size = WindowSize {
            num_lines: lines.max(1) as u16,
            num_cols: cols.max(1) as u16,
            cell_width,
            cell_height,
        };
        let pty = tty::new(&options, window_size, 0)?;

        let dirty = Arc::new(AtomicBool::new(false));
        let exited = Arc::new(AtomicBool::new(false));
        let writer = Arc::new(Mutex::new(None));
        let proxy = EventProxy {
            notifier: notifier.clone(),
            dirty: dirty.clone(),
            exited: exited.clone(),
            writer: writer.clone(),
        };

        let term = Arc::new(FairMutex::new(Term::new(
            TermConfig::default(),
            &GridSize {
                columns: cols.max(1),
                screen_lines: lines.max(1),
            },
            proxy.clone(),
        )));

        let event_loop = PtyEventLoop::new(term.clone(), proxy, pty, false, false)?;
        let sender = event_loop.channel();
        // Now that the loop exists, give the parser its write-back half.
        if let Ok(mut guard) = writer.lock() {
            *guard = Some(event_loop.channel());
        }
        let io_thread = Some(event_loop.spawn());

        Ok(Self {
            term,
            notifier,
            sender,
            dirty,
            exited,
            io_thread,
        })
    }

    /// Scroll this terminal's viewport within its scrollback by `lines`
    /// (positive = toward older output). scroll_display fires no parser
    /// wakeup, so re-arm the dirty flag and poke the UI to repaint the new
    /// viewport through the normal terminal-wakeup path.
    pub fn scroll(&self, lines: i32) {
        if lines == 0 {
            return;
        }
        {
            let mut term = self.term.lock();
            term.scroll_display(Scroll::Delta(lines));
        }
        self.dirty.store(true, Ordering::Release);
        self.notifier.wake();
    }

    /// Queue `bytes` for the PTY (keyboard input, paste, control bytes).
    pub fn write(&self, bytes: Vec<u8>) {
        let _ = self.sender.send(Msg::Input(Cow::Owned(bytes)));
    }

    /// Resize both the PTY (SIGWINCH side) and the parser grid.
    pub fn resize(&self, cols: usize, lines: usize, cell_width: u16, cell_height: u16) {
        let _ = self.sender.send(Msg::Resize(WindowSize {
            num_lines: lines.max(1) as u16,
            num_cols: cols.max(1) as u16,
            cell_width,
            cell_height,
        }));
        self.term.lock().resize(GridSize {
            columns: cols.max(1),
            screen_lines: lines.max(1),
        });
    }

    /// Clear-and-return the dirty flag. Call BEFORE reading
    /// [`TerminalSession::content`] — see the module docs for the
    /// lost-wakeup ordering argument.
    pub fn take_dirty(&self) -> bool {
        self.dirty.swap(false, Ordering::AcqRel)
    }

    /// Whether the shell child has exited.
    pub fn has_exited(&self) -> bool {
        self.exited.load(Ordering::Acquire)
    }

    /// Snapshot the visible grid as trimmed lines joined by `\n`, retaining
    /// ANSI foreground and font attributes as flat UTF-8 byte spans.
    pub fn styled_content(&self) -> TerminalSnapshot {
        let term = self.term.lock();
        let lines = term.screen_lines();
        let cols = term.columns();
        let content = term.renderable_content();
        let offset = content.display_offset as i32;
        let colors = content.colors;

        let default = CellStyle {
            rgb: [220, 214, 201],
            bold: false,
            italic: false,
        };
        let mut rows: Vec<Vec<(char, CellStyle)>> = vec![Vec::with_capacity(cols); lines];
        for indexed in content.display_iter {
            let row = indexed.point.line.0 + offset;
            if row < 0 || row as usize >= lines {
                continue;
            }
            let cell = indexed.cell;
            let inverse = cell.flags.contains(Flags::INVERSE);
            let mut rgb = resolve_color(if inverse { cell.bg } else { cell.fg }, colors);
            if cell.flags.contains(Flags::DIM) {
                rgb = rgb.map(|v| ((v as u16 * 2) / 3) as u8);
            }
            rows[row as usize].push((
                indexed.c,
                CellStyle {
                    rgb,
                    bold: cell.flags.contains(Flags::BOLD),
                    italic: cell.flags.contains(Flags::ITALIC),
                },
            ));
        }
        let cursor_row = content.cursor.point.line.0 + offset;
        let cursor = if cursor_row >= 0 && (cursor_row as usize) < lines {
            Some((cursor_row as usize, content.cursor.point.column.0))
        } else {
            None
        };

        let mut text = String::new();
        let mut spans: Vec<TerminalSpan> = Vec::new();
        for (i, row) in rows.iter().enumerate() {
            if i > 0 {
                text.push('\n');
            }
            let used = row
                .iter()
                .rposition(|(c, _)| *c != ' ')
                .map_or(0, |n| n + 1);
            for &(ch, style) in &row[..used] {
                let start = text.len();
                text.push(ch);
                let end = text.len();
                if let Some(last) = spans.last_mut() {
                    if last.end == start
                        && last.rgb == style.rgb
                        && last.bold == style.bold
                        && last.italic == style.italic
                    {
                        last.end = end;
                        continue;
                    }
                }
                spans.push(TerminalSpan {
                    start,
                    end,
                    rgb: style.rgb,
                    bold: style.bold,
                    italic: style.italic,
                });
            }
            if used == 0 && i + 1 == lines {
                let _ = default;
            }
        }
        TerminalSnapshot {
            text,
            cursor,
            spans,
        }
    }

    /// Compatibility helper used by plain-text consumers and existing tests.
    pub fn content(&self) -> (String, Option<(usize, usize)>) {
        let snapshot = self.styled_content();
        (snapshot.text, snapshot.cursor)
    }

    /// Stop the reader loop and reap the shell child. Joining the IO thread is
    /// what drops the `Pty` (SIGHUP + waitpid in its `Drop`), so skipping the
    /// join would leak a zombie. The loop exits promptly on `Shutdown`
    /// (drain_on_exit = false), so the join is bounded.
    pub fn shutdown(&mut self) {
        let _ = self.sender.send(Msg::Shutdown);
        if let Some(handle) = self.io_thread.take() {
            let _ = handle.join();
        }
    }
}

impl<N: TermNotifier> Drop for TerminalSession<N> {
    fn drop(&mut self) {
        self.shutdown();
    }
}
