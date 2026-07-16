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
//! Fidelity shipped at P3: plain text grid + cursor position. Per-cell
//! color/bold/italic and Title/Clipboard/ColorRequest events are TODO(P3) —
//! see the `_ => {}` arm in [`EventProxy::send_event`].

use std::borrow::Cow;
use std::io;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use alacritty_terminal::event::{Event as TermEvent, EventListener, WindowSize};
use alacritty_terminal::event_loop::{EventLoop as PtyEventLoop, EventLoopSender, Msg, State};
use alacritty_terminal::grid::Dimensions;
use alacritty_terminal::sync::FairMutex;
use alacritty_terminal::term::{Config as TermConfig, Term};
use alacritty_terminal::tty::{self, Options as PtyOptions, Pty, Shell};

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

/// A live terminal: shell child, PTY reader thread, shared parsed grid.
pub struct TerminalSession<N: TermNotifier> {
    term: Arc<FairMutex<Term<EventProxy<N>>>>,
    sender: EventLoopSender,
    dirty: Arc<AtomicBool>,
    exited: Arc<AtomicBool>,
    io_thread: Option<std::thread::JoinHandle<(PtyEventLoop<Pty, EventProxy<N>>, State)>>,
}

impl<N: TermNotifier> TerminalSession<N> {
    /// Spawn `$SHELL` (fallback `/bin/sh`) in a PTY of `cols` x `lines` cells.
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
            (
                std::env::var("SHELL").unwrap_or_else(|_| "/bin/sh".to_string()),
                Vec::new(),
            )
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
            notifier,
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
            sender,
            dirty,
            exited,
            io_thread,
        })
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

    /// Snapshot the visible grid as trimmed lines joined by `\n`, plus the
    /// cursor `(row, col)` when it is inside the viewport. Allocates — called
    /// only on coalesced wakeups, never per frame.
    pub fn content(&self) -> (String, Option<(usize, usize)>) {
        let term = self.term.lock();
        let lines = term.screen_lines();
        let cols = term.columns();
        let content = term.renderable_content();
        let offset = content.display_offset as i32;

        let mut rows: Vec<String> = vec![String::with_capacity(cols); lines];
        for indexed in content.display_iter {
            let row = indexed.point.line.0 + offset;
            if row < 0 || row as usize >= lines {
                continue;
            }
            rows[row as usize].push(indexed.c);
        }
        let cursor_row = content.cursor.point.line.0 + offset;
        let cursor = if cursor_row >= 0 && (cursor_row as usize) < lines {
            Some((cursor_row as usize, content.cursor.point.column.0))
        } else {
            None
        };

        let mut text = String::new();
        for (i, row) in rows.iter().enumerate() {
            if i > 0 {
                text.push('\n');
            }
            text.push_str(row.trim_end());
        }
        (text, cursor)
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
