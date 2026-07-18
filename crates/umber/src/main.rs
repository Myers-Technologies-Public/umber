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

use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::{Duration, Instant};

use umber::agent_rpc::{AgentNotifier, AgentProcess, AgentRunState};
use umber::agents::{self, SessionSummary};
use umber::git::{self, LineChange};
use umber::remote::RemoteWorkspace;
use umber::search::{self, Match};
use umber::terminal::{TermNotifier, TerminalSession};
use umber_host::{HostCommand, Manifest, ModuleHost};
use umber_kernel::{Command, CommandRegistry, Config, FeatureRegistry};
use umber_text::TextBuffer;
use umber_ui::{
    OverlaySpec, Renderer, ScrollbarInfo, SelSpan, GIT_ADDED_COLOR, GIT_DELETED_COLOR,
    GIT_MODIFIED_COLOR,
};

use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop, EventLoopProxy};
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
const SETTINGS_ROWS: usize = 7;

/// Cross-thread wakeups from background machinery (P3: the terminal's PTY
/// reader thread). Delivered through winit's user-event channel.
#[derive(Debug, Clone, Copy)]
enum UserEvent {
    TerminalWakeup,
    TerminalExited,
    /// A live pi RPC agent updated its state/output (P4 slice 2).
    AgentUpdated,
}

/// [`TermNotifier`] over the winit event-loop proxy.
#[derive(Clone)]
struct UmberNotifier(EventLoopProxy<UserEvent>);

impl TermNotifier for UmberNotifier {
    fn wake(&self) {
        let _ = self.0.send_event(UserEvent::TerminalWakeup);
    }
    fn child_exited(&self) {
        let _ = self.0.send_event(UserEvent::TerminalExited);
    }
}

impl AgentNotifier for UmberNotifier {
    fn agent_updated(&self) {
        let _ = self.0.send_event(UserEvent::AgentUpdated);
    }
}

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
    /// QoL (P3b): all commands + chords, read-only.
    Help,
    /// QoL: numeric go-to-line prompt.
    GotoLine,
    /// P3b: pick (or type) an SSH host; Enter opens `ssh <host>` in the panel.
    SshPicker,
    /// P4 slice 1: read-only pi session dashboard (history from JSONL).
    Agents,
    /// P4 slice 2: type a prompt to send to the live attached agent.
    AgentPrompt,
    /// Full-thread viewer for a selected agent/session (scrollable).
    AgentThread,
    /// P3b-deep: enter an SSH host for a remote workspace.
    RemoteHost,
    /// P3b-deep: enter a remote file path to open over the workspace.
    RemotePath,
    /// P5: project-wide text search.
    Search,
}

/// What the pointer is hovering over in the document body, for hover
/// highlighting. `Line` = whitespace / past line end / empty line (separator
/// segment only); `Word` = a run of word chars or a single punctuation char
/// (gold recolor + segment). `end_col` is exclusive. Compared against the
/// previous target so a redraw fires only when the target actually changes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum HoverTarget {
    None,
    Line(usize),
    Word {
        line: usize,
        start_col: usize,
        end_col: usize,
    },
}

/// A command palette row: a built-in command or a loaded external-module
/// command, unified so the fuzzy filter and overlay treat both alike. Rebuilt
/// from the kernel registry plus the module host's live commands each time the
/// palette opens.
struct PaletteItem {
    id: String,
    title: String,
    keybinding: String,
}

/// A discovered external module (`~/.config/umber/modules/<name>/umber.toml`)
/// and its live host state, listed on the modules page beneath the built-in
/// features. A bad manifest is surfaced (`manifest: Err`), never fatal; a load
/// failure is captured in `error` so the page can show it without crashing.
struct ExternalModule {
    /// Manifest `name` when it parsed, else the directory name.
    name: String,
    base_dir: PathBuf,
    /// Parsed manifest, or the parse error text.
    manifest: Result<Manifest, String>,
    /// Currently loaded (instantiated) in the host.
    loaded: bool,
    /// Last load error, surfaced on the page; `None` when healthy.
    error: Option<String>,
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
        (
            "terminal.toggle",
            "Terminal: Toggle Panel",
            "Ctrl+` / Ctrl+J",
        ),
        ("terminal.focus", "Terminal: Focus", ""),
        ("terminal.ssh", "Terminal: SSH to Host\u{2026}", ""),
        ("terminal.maximize", "Terminal: Toggle Fullscreen", "F11"),
        ("view.nextTab", "View: Next Tab", "Ctrl+Tab"),
        ("view.closeTab", "View: Close Tab", "Ctrl+W"),
        (
            "view.toggleSidebar",
            "View: Toggle Sidebar Labels",
            "Ctrl+B",
        ),
        ("goto.line", "Go: Line\u{2026}", "Ctrl+G"),
        ("help.keys", "Help: Keyboard Shortcuts", "F1"),
        ("agents.dashboard", "Agents: pi Dashboard", "Ctrl+Shift+A"),
        ("remote.open", "Remote: Open File over SSH\u{2026}", ""),
        ("remote.disconnect", "Remote: Disconnect Workspace", ""),
        (
            "search.project",
            "Search: In Project\u{2026}",
            "Ctrl+Shift+F",
        ),
        ("view.toggle.terminal", "View: Toggle Terminal Feature", ""),
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

/// Host aliases from ssh_config text: every name after a `Host` keyword,
/// minus wildcard (`*`/`?`) and negated (`!`) patterns.
fn parse_ssh_hosts(text: &str) -> Vec<String> {
    let mut hosts = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if !line.to_ascii_lowercase().starts_with("host ") {
            continue;
        }
        for name in line.split_whitespace().skip(1) {
            if !name.contains('*') && !name.contains('?') && !name.starts_with('!') {
                hosts.push(name.to_string());
            }
        }
    }
    hosts.sort();
    hosts.dedup();
    hosts
}

/// Hosts from `~/.ssh/config` for the SSH picker. Missing file = empty (the
/// picker still accepts a typed host).
fn ssh_config_hosts() -> Vec<String> {
    let Some(home) = std::env::var_os("HOME") else {
        return Vec::new();
    };
    let path = PathBuf::from(home).join(".ssh").join("config");
    match std::fs::read_to_string(path) {
        Ok(text) => parse_ssh_hosts(&text),
        Err(_) => Vec::new(),
    }
}

/// What each visible row on the Agents page IS (parallel to the row list),
/// so clicks and Enter know their target.
#[derive(Clone, Copy, PartialEq)]
enum AgentsRow {
    /// Section header / informational — not actionable.
    Header,
    /// The live attached agent (opens its streaming thread).
    Live,
    /// `agents_sessions[i]` (opens its transcript).
    Session(usize),
    /// The History expander toggle.
    Expander,
}

/// Word-wrap one transcript message into display rows: `you ▸` / `  ●`
/// prefix on the first line, indent on continuations, blank spacer after.
fn wrap_message(rows: &mut Vec<(String, String)>, speaker: &str, text: &str) {
    const WIDTH: usize = 100;
    let prefix = if speaker == "you" {
        "you \u{25b8} "
    } else {
        "  \u{25cf} "
    };
    let indent = "      ";
    let mut line = String::new();
    let mut first = true;
    for word in text.split_whitespace() {
        if !line.is_empty() && line.chars().count() + word.chars().count() + 1 > WIDTH {
            let lead = if first { prefix } else { indent };
            rows.push((format!("{lead}{line}"), String::new()));
            first = false;
            line.clear();
        }
        if !line.is_empty() {
            line.push(' ');
        }
        line.push_str(word);
    }
    if !line.is_empty() {
        let lead = if first { prefix } else { indent };
        rows.push((format!("{lead}{line}"), String::new()));
    }
    rows.push((String::new(), String::new()));
}

/// First visible row of a windowed overlay list for a given selection (the
/// selection is kept at the window's bottom edge once the list scrolls).
fn windowed_start(sel: usize, n: usize, cap: usize) -> usize {
    let sel = if n == 0 { 0 } else { sel.min(n - 1) };
    if sel < cap {
        0
    } else {
        sel + 1 - cap
    }
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

    let event_loop = match EventLoop::<UserEvent>::with_user_event().build() {
        Ok(ev) => ev,
        Err(err) => {
            eprintln!("umber: failed to create the event loop: {err}");
            return ExitCode::FAILURE;
        }
    };
    let event_proxy = event_loop.create_proxy();

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

    // Module host (D9). A wasmtime-engine failure must not sink the editor:
    // degrade to no modules, exactly like the clipboard path above.
    let module_host = match ModuleHost::new() {
        Ok(h) => Some(h),
        Err(err) => {
            eprintln!("umber: module host unavailable ({err}); modules disabled");
            None
        }
    };

    let mut app = App {
        buffer,
        docs: vec![Document::husk()],
        active_doc: 0,
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
        module_host,
        modules: Vec::new(),
        module_commands: Vec::new(),
        modules_enabled: BTreeSet::new(),
        module_status: None,
        palette_items: Vec::new(),
        scrollbar_linger,
        cursor_char: 0,
        goal_col: 0,
        selection_anchor: None,
        selecting: false,
        clipboard,
        sel_spans: Vec::new(),
        hover: HoverTarget::None,
        first_visible_line: 0,
        modifiers: ModifiersState::empty(),
        pointer: (0.0, 0.0),
        scrollbar_deadline: None,
        scrollbar_dragging: false,
        drag_anchor_y: 0.0,
        drag_anchor_first: 0,
        scrollbar_drawn: false,
        agents_sessions: Vec::new(),
        agents_scroll: 0,
        agent_proc: None,
        agent_prompt: String::new(),
        agents_expanded: false,
        agent_thread: Vec::new(),
        agent_thread_title: String::new(),
        agent_thread_scroll: 0,
        remote: None,
        remote_file: None,
        remote_host_input: String::new(),
        remote_path_input: String::new(),
        git_status: std::collections::HashMap::new(),
        search_input: String::new(),
        search_results: Vec::new(),
        search_sel: 0,
        help_scroll: 0,
        goto_input: String::new(),
        ssh_hosts: Vec::new(),
        ssh_input: String::new(),
        ssh_filtered: Vec::new(),
        ssh_sel: 0,
        terminal: None,
        term_focused: false,
        term_resizing: false,
        sidebar_resizing: false,
        term_tab_active: false,
        last_click_at: None,
        last_click_pos: None,
        event_proxy,
        start,
        first_frame: false,
        first_frame_at: None,
        rss_printed: false,
    };

    // Discover + load enabled modules before the event loop so their commands
    // are in the palette from the first frame.
    app.init_modules();

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

/// One open editor tab's saved state. The ACTIVE document's live state lives
/// in the `App` fields directly (so the ~50 `self.buffer` call sites need no
/// change); `App::docs[active_doc]` is a husk whose buffer is swapped out.
/// Switching stashes the active fields back into its slot and swaps the target
/// in. `TextBuffer` isn't `Clone` (owns rope + undo), so this is move/swap,
/// never copy.
struct Document {
    buffer: TextBuffer,
    cursor_char: usize,
    goal_col: usize,
    selection_anchor: Option<usize>,
    first_visible_line: usize,
    git_status: std::collections::HashMap<usize, LineChange>,
    remote_file: Option<String>,
}

impl Document {
    /// An empty placeholder slot (used for the active tab, whose real state is
    /// in the `App` fields).
    fn husk() -> Self {
        Self {
            buffer: TextBuffer::empty(),
            cursor_char: 0,
            goal_col: 0,
            selection_anchor: None,
            first_visible_line: 0,
            git_status: std::collections::HashMap::new(),
            remote_file: None,
        }
    }
}

struct App {
    buffer: TextBuffer,
    /// Open editor tabs (one husk slot per tab; active tab's data is in the
    /// App fields). Always ≥ 1.
    docs: Vec<Document>,
    active_doc: usize,
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
    /// Module host (D9): wasm + lua backends behind the deny-all broker. `None`
    /// if the wasmtime engine failed to build (the editor still runs).
    module_host: Option<ModuleHost>,
    /// Discovered external modules, shown after the built-in features on the
    /// modules page; index `modules_sel - features.len()` selects one.
    modules: Vec<ExternalModule>,
    /// Commands provided by currently-loaded modules (palette + dispatch).
    module_commands: Vec<HostCommand>,
    /// Names of modules the user has enabled; persisted to the host's sidecar
    /// (`$CONFIG/umber/modules-enabled`) so the set survives restarts.
    modules_enabled: BTreeSet<String>,
    /// Last line of module output (or an error), shown in the status banner.
    module_status: Option<String>,
    /// Unified palette source (built-ins + module commands), rebuilt on open.
    palette_items: Vec<PaletteItem>,
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

    /// Current hover-highlight target under the pointer. A new target is pushed
    /// to the renderer and redrawn only when this CHANGES (never on raw
    /// `CursorMoved`); cleared when the pointer leaves the doc, a modal opens,
    /// or the text under the pointer moves (scroll/edit).
    hover: HoverTarget,

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

    // --- P4: pi agent dashboard (read-only slice) ---
    /// Parsed session summaries, newest first (refreshed on open / `r`).
    agents_sessions: Vec<SessionSummary>,
    /// Scroll offset into the sessions list.
    agents_scroll: usize,
    /// Live attached agent (P4 slice 2), spawned in the dashboard with `n`.
    agent_proc: Option<AgentProcess>,
    /// In-progress prompt text for the live agent.
    agent_prompt: String,
    /// History section expanded on the Agents page.
    agents_expanded: bool,
    /// Loaded thread rows + title + scroll for the thread viewer.
    agent_thread: Vec<(String, String)>,
    agent_thread_title: String,
    agent_thread_scroll: usize,

    // --- P3b-deep: remote workspace over umberd/SSH ---
    /// Connected remote workspace; when set, the buffer is remote-backed and
    /// Ctrl+S writes through the protocol.
    remote: Option<RemoteWorkspace>,
    /// Remote path of the open buffer (the daemon-relative path).
    remote_file: Option<String>,
    /// In-progress host / path entry for the remote-open flow.
    remote_host_input: String,
    remote_path_input: String,

    // --- P5: git gutter status (line -> change) ---
    git_status: std::collections::HashMap<usize, LineChange>,

    // --- P5: project search ---
    search_input: String,
    search_results: Vec<Match>,
    search_sel: usize,

    // --- P3b/QoL: help, go-to-line, SSH picker ---
    /// Scroll offset of the help overlay.
    help_scroll: usize,
    /// Digits typed into the go-to-line prompt.
    goto_input: String,
    /// SSH picker state: hosts from ~/.ssh/config, filter text, filtered
    /// indices, selection.
    ssh_hosts: Vec<String>,
    ssh_input: String,
    ssh_filtered: Vec<usize>,
    ssh_sel: usize,

    // --- P3: embedded terminal ---
    /// Live terminal session; spawned on first open, killed on feature
    /// disable / child exit / quit (Drop reaps the shell).
    terminal: Option<TerminalSession<UmberNotifier>>,
    /// Keyboard focus owner: `true` = terminal panel, else the editor.
    term_focused: bool,
    /// Dragging the terminal's top border to resize the split.
    term_resizing: bool,
    /// Dragging the sidebar separator to resize it.
    sidebar_resizing: bool,
    /// The terminal content-tab is active (terminal fills the content area).
    term_tab_active: bool,
    /// Double-click tracking for word-select.
    last_click_at: Option<Instant>,
    last_click_pos: Option<usize>,
    /// Proxy for background threads to wake the event loop.
    event_proxy: EventLoopProxy<UserEvent>,

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
        // Refresh the tab strip first so doc_top() reflects its height for the
        // window/geometry computed below.
        self.sync_tabs();
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
        let name = match (&self.remote, &self.remote_file) {
            // Remote-backed buffer (P3b): show host:path so the source is
            // unambiguous and Ctrl+S's remote target is visible.
            (Some(ws), Some(path)) => format!("{}:{path}", ws.label),
            _ => self
                .buffer
                .path()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "*scratch*".to_string()),
        };
        // Minimal status: position only — the filename + dirty dot live in
        // the sidebar tab, the remote host in the tab label.
        let _ = (dirty, name);
        let mut prefix = format!("Ln {}, Col {}", cl + 1, col + 1);
        // Append the last module status line, if any (simplest honest surface
        // for module text output \u{2014} the editor status banner).
        if let Some(status) = &self.module_status {
            let _ = write!(prefix, "  \u{2014}  {status}");
        }

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

        // Git gutter markers for the visible window (P5): map each changed
        // absolute line that falls in view to its window-relative row + color.
        let mut marks: Vec<(usize, [f32; 4])> = Vec::new();
        if !self.git_status.is_empty() {
            for row in 0..cap {
                let abs = self.first_visible_line + row + 1; // git lines are 1-based
                if let Some(change) = self.git_status.get(&abs) {
                    let color = match change {
                        LineChange::Added => GIT_ADDED_COLOR,
                        LineChange::Modified => GIT_MODIFIED_COLOR,
                        LineChange::Deleted => GIT_DELETED_COLOR,
                    };
                    marks.push((row, color));
                }
            }
        }

        if let Some(renderer) = self.renderer.as_mut() {
            renderer.set_gutter(&numbers, digits);
            renderer.set_document(&text);
            renderer.set_cursor(cursor);
            renderer.set_selection(&spans);
            renderer.set_gutter_marks(marks);
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
        if let Some(r) = self.renderer.as_ref() {
            if y >= r.doc_bottom() {
                return None; // terminal panel, not the document
            }
        }
        let rel_line = ((y - doc_top) / line_px).floor() as i64;
        let last = self.buffer.len_lines().saturating_sub(1) as i64;
        let line = (self.first_visible_line as i64 + rel_line).clamp(0, last) as usize;
        let col_f = ((x - text_left) / cell_w).round();
        let col = if col_f < 0.0 { 0 } else { col_f as usize };
        let col = col.min(self.buffer.visual_line_len_chars(line));
        Some(self.buffer.line_to_char(line) + col)
    }

    /// Map the current pointer to a hover target over the document body, using
    /// the same doc/cell arithmetic as click placement. Returns `None` for the
    /// banner, the gutter, or past the last line. On a non-whitespace char the
    /// target is the WORD under it; on whitespace / past line end / empty line
    /// it is the LINE.
    fn pointer_to_hover(&self) -> HoverTarget {
        let (doc_top, line_px, text_left, cell_w) = match self.renderer.as_ref() {
            Some(r) => (r.doc_top(), r.line_px(), r.text_left(), r.cell_w()),
            None => return HoverTarget::None,
        };
        let x = self.pointer.0 as f32;
        let y = self.pointer.1 as f32;
        if y < doc_top || x < text_left {
            return HoverTarget::None; // banner or gutter, not the document body
        }
        if let Some(r) = self.renderer.as_ref() {
            // P3: the terminal panel is not the document.
            if y >= r.doc_bottom() {
                return HoverTarget::None;
            }
        }
        let cap = self
            .renderer
            .as_ref()
            .map(|r| r.visible_line_capacity())
            .unwrap_or(0);
        let rel_line = ((y - doc_top) / line_px).floor() as i64;
        if rel_line < 0 || rel_line as usize >= cap {
            return HoverTarget::None;
        }
        let last = self.buffer.len_lines().saturating_sub(1) as i64;
        let abs_line = self.first_visible_line as i64 + rel_line;
        if abs_line > last {
            return HoverTarget::None;
        }
        let line = abs_line as usize;
        // The document-area guard above ensures `x >= text_left`, so the
        // division cannot go negative.
        let col_f = ((x - text_left) / cell_w).floor();
        debug_assert!(col_f >= 0.0);
        let col = col_f as usize;
        let line_len = self.buffer.visual_line_len_chars(line);
        if col >= line_len {
            return HoverTarget::Line(line);
        }
        let line_start = self.buffer.line_to_char(line);
        let line_str = self.buffer.slice_chars(line_start, line_start + line_len);
        match word_span_at(&line_str, col) {
            Some((start_col, end_col)) => HoverTarget::Word {
                line,
                start_col,
                end_col,
            },
            None => HoverTarget::Line(line),
        }
    }

    /// Window-relative line for an absolute document line, or `None` if it is
    /// scrolled out of the visible window.
    fn hover_rel_line(&self, line: usize) -> Option<usize> {
        let cap = self.renderer.as_ref()?.visible_line_capacity();
        if line >= self.first_visible_line && line < self.first_visible_line + cap {
            Some(line - self.first_visible_line)
        } else {
            None
        }
    }

    /// Push the current `hover` target into the renderer as word + segment
    /// state. Reshapes the word buffer inside the renderer only when the word
    /// text changes; never touches the document/gutter buffers.
    fn push_hover_to_renderer(&mut self) {
        let (word, line): (Option<(usize, usize, String)>, Option<usize>) = match self.hover {
            HoverTarget::None => (None, None),
            HoverTarget::Line(line) => (None, self.hover_rel_line(line)),
            HoverTarget::Word {
                line,
                start_col,
                end_col,
            } => {
                let rel = self.hover_rel_line(line);
                let ls = self.buffer.line_to_char(line);
                let text = self.buffer.slice_chars(ls + start_col, ls + end_col);
                (rel.map(|r| (r, start_col, text)), rel)
            }
        };
        if let Some(r) = self.renderer.as_mut() {
            match &word {
                Some((rel, sc, t)) => r.set_hover_word(Some((*rel, *sc, t.as_str()))),
                None => r.set_hover_word(None),
            }
            r.set_hover_line(line);
        }
    }

    /// Recompute the hover target from the pointer; if it CHANGED, push it to
    /// the renderer and request exactly one redraw. No-op (no redraw) when the
    /// target is unchanged -- this is the guard that keeps raw `CursorMoved`
    /// from triggering a frame storm.
    fn update_hover(&mut self) {
        let target = self.pointer_to_hover();
        if target == self.hover {
            return;
        }
        self.hover = target;
        self.push_hover_to_renderer();
        if let Some(r) = self.renderer.as_ref() {
            r.window().request_redraw();
        }
    }

    /// Clear the hover target, redrawing once only if something was showing.
    /// Called when the pointer leaves the doc, a modal opens, or the text under
    /// the pointer moves (scroll/edit).
    fn clear_hover(&mut self) {
        if self.hover == HoverTarget::None {
            return;
        }
        self.hover = HoverTarget::None;
        self.push_hover_to_renderer();
        if let Some(r) = self.renderer.as_ref() {
            r.window().request_redraw();
        }
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
    /// Recompute git line-status for the open local file (P5). Remote/scratch
    /// buffers clear it. Called on open + save (not per keystroke).
    fn refresh_git(&mut self) {
        if self.remote.is_some() {
            self.git_status.clear();
            return;
        }
        self.git_status = match self.buffer.path() {
            Some(p) => git::file_line_status(p),
            None => std::collections::HashMap::new(),
        };
    }

    fn do_save(&mut self) {
        // Remote-backed buffer (P3b): write the whole file back over the
        // umber-proto workspace instead of the local filesystem.
        if let (Some(ws), Some(path)) = (self.remote.as_mut(), self.remote_file.clone()) {
            let text = self.buffer.full_text();
            match ws.write_file(&path, &text) {
                Ok(_) => self.buffer.mark_saved(),
                Err(err) => eprintln!("umber: remote save failed: {err}"),
            }
            return;
        }
        match self.buffer.save() {
            Ok(true) => self.refresh_git(),
            Ok(false) => eprintln!("umber: no path to save (scratch buffer)"),
            Err(err) => eprintln!("umber: save failed: {err}"),
        }
    }

    // ===================================================================
    //  P5: project-wide search.
    // ===================================================================

    /// Project root for search: the open file's parent, else the process cwd.
    fn project_root(&self) -> PathBuf {
        self.buffer
            .path()
            .and_then(|p| p.parent().map(|d| d.to_path_buf()))
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")))
    }

    fn open_search(&mut self) {
        self.search_input.clear();
        self.search_results.clear();
        self.search_sel = 0;
        self.view = View::Search;
        self.refresh_overlay();
    }

    fn run_search(&mut self) {
        let root = self.project_root();
        self.search_results = search::search_dir(&root, self.search_input.trim(), 200);
        self.search_sel = 0;
        self.refresh_overlay();
    }

    /// Open the file for the selected match and jump to its line/col.
    // ===================================================================
    //  Multi-buffer open-editor tabs.
    // ===================================================================

    /// Display name for tab `i` (active tab reads the live buffer).
    fn tab_name(&self, i: usize) -> String {
        let path = if i == self.active_doc {
            self.buffer.path()
        } else {
            self.docs[i].buffer.path()
        };
        if let Some(name) =
            path.and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
        {
            return name;
        }
        // Remote-backed active buffer: show the remote file (marked).
        if i == self.active_doc {
            if let Some(rf) = &self.remote_file {
                let base = rf.rsplit('/').next().unwrap_or(rf);
                return format!("\u{21c5} {base}");
            }
        }
        "*scratch*".to_string()
    }

    /// Index of an already-open tab for `path`, if any.
    fn find_tab_by_path(&self, path: &std::path::Path) -> Option<usize> {
        (0..self.docs.len()).find(|&i| {
            let p = if i == self.active_doc {
                self.buffer.path()
            } else {
                self.docs[i].buffer.path()
            };
            p == Some(path)
        })
    }

    /// Stash the live editor fields back into the active tab's slot.
    fn stash_active(&mut self) {
        let d = &mut self.docs[self.active_doc];
        std::mem::swap(&mut d.buffer, &mut self.buffer);
        d.cursor_char = self.cursor_char;
        d.goal_col = self.goal_col;
        d.selection_anchor = self.selection_anchor;
        d.first_visible_line = self.first_visible_line;
        d.git_status = std::mem::take(&mut self.git_status);
        d.remote_file = self.remote_file.take();
    }

    /// Load tab `i`'s slot into the live editor fields.
    fn load_active_from(&mut self, i: usize) {
        let d = &mut self.docs[i];
        std::mem::swap(&mut self.buffer, &mut d.buffer);
        self.cursor_char = d.cursor_char;
        self.goal_col = d.goal_col;
        self.selection_anchor = d.selection_anchor;
        self.first_visible_line = d.first_visible_line;
        self.git_status = std::mem::take(&mut d.git_status);
        self.remote_file = d.remote_file.take();
        self.active_doc = i;
    }

    /// Switch to tab `i`.
    fn switch_tab(&mut self, i: usize) {
        if i == self.active_doc || i >= self.docs.len() {
            return;
        }
        self.stash_active();
        self.load_active_from(i);
        self.selecting = false;
        self.apply_view(true);
        if let Some(r) = self.renderer.as_ref() {
            r.window().request_redraw();
        }
    }

    /// Cycle to the next tab (Ctrl+Tab).
    fn next_tab(&mut self) {
        if self.term_tab_active {
            self.deactivate_terminal_tab();
            return;
        }
        if self.docs.len() > 1 {
            let next = (self.active_doc + 1) % self.docs.len();
            self.switch_tab(next);
        }
    }

    /// Open `path` in a new tab, or switch to it if already open.
    fn open_path_in_tab(&mut self, path: &std::path::Path) {
        if let Some(i) = self.find_tab_by_path(path) {
            self.switch_tab(i);
            return;
        }
        let buf = match TextBuffer::from_path(path) {
            Ok(b) => b,
            Err(err) => {
                eprintln!("umber: cannot open {:?}: {err}", path);
                return;
            }
        };
        self.stash_active();
        self.docs.push(Document::husk());
        self.active_doc = self.docs.len() - 1;
        self.buffer = buf;
        self.cursor_char = 0;
        self.goal_col = 0;
        self.selection_anchor = None;
        self.first_visible_line = 0;
        if let Some(mut ws) = self.remote.take() {
            ws.shutdown();
        }
        self.remote_file = None;
        self.selecting = false;
        self.refresh_git();
        self.apply_view(true);
        if let Some(r) = self.renderer.as_ref() {
            r.window().request_redraw();
        }
    }

    /// Close the active tab (always keeps at least one open).
    fn close_active_tab(&mut self) {
        if self.docs.len() <= 1 {
            return;
        }
        let i = self.active_doc;
        let new = if i + 1 < self.docs.len() {
            i + 1
        } else {
            i - 1
        };
        // Load the neighbor into the live fields, dropping the closing buffer.
        self.buffer = std::mem::replace(&mut self.docs[new].buffer, TextBuffer::empty());
        self.cursor_char = self.docs[new].cursor_char;
        self.goal_col = self.docs[new].goal_col;
        self.selection_anchor = self.docs[new].selection_anchor;
        self.first_visible_line = self.docs[new].first_visible_line;
        self.git_status = std::mem::take(&mut self.docs[new].git_status);
        self.remote_file = self.docs[new].remote_file.take();
        self.docs.remove(i);
        self.active_doc = if new > i { new - 1 } else { new };
        self.selecting = false;
        self.apply_view(true);
        if let Some(r) = self.renderer.as_ref() {
            r.window().request_redraw();
        }
    }

    /// Close any tab by index (keeps at least one open).
    fn close_tab(&mut self, i: usize) {
        if self.docs.len() <= 1 || i >= self.docs.len() {
            return;
        }
        if i == self.active_doc {
            self.close_active_tab();
            return;
        }
        self.docs.remove(i);
        if i < self.active_doc {
            self.active_doc -= 1;
        }
        self.apply_view(false);
        if let Some(r) = self.renderer.as_ref() {
            r.window().request_redraw();
        }
    }

    fn open_search_result(&mut self) {
        let Some(m) = self.search_results.get(self.search_sel).cloned() else {
            return;
        };
        self.view = View::Editor;
        self.close_overlay();
        // Open (or switch to) the file in its own tab, then jump to the match.
        self.open_path_in_tab(&m.path);
        let line = m.line.saturating_sub(1);
        let base = self.buffer.line_to_char(line);
        let col = m.col.min(self.buffer.visual_line_len_chars(line));
        self.cursor_char = base + col;
        self.selection_anchor = None;
        self.first_visible_line = line.saturating_sub(4);
        self.update_goal_col();
        self.apply_view(true);
        if let Some(r) = self.renderer.as_ref() {
            r.window().request_redraw();
        }
    }

    fn search_key(&mut self, event: KeyEvent) {
        match &event.logical_key {
            Key::Named(NamedKey::Escape) => self.close_overlay(),
            Key::Named(NamedKey::Enter) => self.open_search_result(),
            Key::Named(NamedKey::ArrowDown) => {
                let n = self.search_results.len();
                if n > 0 {
                    self.search_sel = (self.search_sel + 1) % n;
                }
                self.refresh_overlay();
            }
            Key::Named(NamedKey::ArrowUp) => {
                let n = self.search_results.len();
                if n > 0 {
                    self.search_sel = (self.search_sel + n - 1) % n;
                }
                self.refresh_overlay();
            }
            Key::Named(NamedKey::Backspace) => {
                self.search_input.pop();
                self.run_search();
            }
            _ => {
                if let Some(t) = &event.text {
                    let mut changed = false;
                    for ch in t.chars() {
                        if !ch.is_control() {
                            self.search_input.push(ch);
                            changed = true;
                        }
                    }
                    if changed {
                        self.run_search();
                    }
                }
            }
        }
    }

    /// Handle the remote-host entry prompt: Enter connects, then asks for a
    /// path. Typed text or a selected ssh_config host both work.
    fn remote_host_key(&mut self, event: KeyEvent) {
        match &event.logical_key {
            Key::Named(NamedKey::Escape) => self.close_overlay(),
            Key::Named(NamedKey::Enter) => {
                if self.remote_host_input.trim().is_empty() {
                    return;
                }
                self.remote_path_input.clear();
                self.view = View::RemotePath;
                self.refresh_overlay();
            }
            Key::Named(NamedKey::Backspace) => {
                self.remote_host_input.pop();
                self.refresh_overlay();
            }
            _ => {
                if let Some(t) = &event.text {
                    let mut changed = false;
                    for ch in t.chars() {
                        if !ch.is_control() && !ch.is_whitespace() {
                            self.remote_host_input.push(ch);
                            changed = true;
                        }
                    }
                    if changed {
                        self.refresh_overlay();
                    }
                }
            }
        }
    }

    /// Handle the remote-path prompt: Enter connects to the host over
    /// `ssh <host> umberd`, reads the file, and loads it as a remote-backed
    /// buffer.
    fn remote_path_key(&mut self, event: KeyEvent) {
        match &event.logical_key {
            Key::Named(NamedKey::Escape) => self.close_overlay(),
            Key::Named(NamedKey::Enter) => {
                let host = self.remote_host_input.trim().to_string();
                let path = self.remote_path_input.trim().to_string();
                if path.is_empty() {
                    return;
                }
                match RemoteWorkspace::connect_ssh(&host) {
                    Ok(mut ws) => match ws.read_file(&path) {
                        Ok(contents) => {
                            self.buffer = TextBuffer::from_string(&contents);
                            self.cursor_char = 0;
                            self.selection_anchor = None;
                            self.first_visible_line = 0;
                            self.remote = Some(ws);
                            self.remote_file = Some(path);
                            self.view = View::Editor;
                            self.close_overlay();
                            self.apply_view(true);
                            if let Some(r) = self.renderer.as_ref() {
                                r.window().request_redraw();
                            }
                        }
                        Err(err) => {
                            self.remote_host_input = format!("{host}  (read failed: {err})");
                            self.view = View::RemoteHost;
                            self.refresh_overlay();
                        }
                    },
                    Err(err) => {
                        self.remote_host_input = format!("{host}  (connect failed: {err})");
                        self.view = View::RemoteHost;
                        self.refresh_overlay();
                    }
                }
            }
            Key::Named(NamedKey::Backspace) => {
                self.remote_path_input.pop();
                self.refresh_overlay();
            }
            _ => {
                if let Some(t) = &event.text {
                    let mut changed = false;
                    for ch in t.chars() {
                        if !ch.is_control() {
                            self.remote_path_input.push(ch);
                            changed = true;
                        }
                    }
                    if changed {
                        self.refresh_overlay();
                    }
                }
            }
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
        // P3: disabling the terminal feature kills the live shell (both the
        // modules page and palette toggles funnel through here).
        if !self.config.terminal && self.terminal.is_some() {
            self.kill_terminal();
        }
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
        self.sync_activity_strip();
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
                    let c = &self.palette_items[ci];
                    rows.push((c.title.clone(), c.keybinding.clone()));
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
            View::Help => {
                let cap = self
                    .renderer
                    .as_ref()
                    .map(|r| r.overlay_row_capacity())
                    .unwrap_or(1);
                let n = self.palette_items.len();
                let start = self.help_scroll.min(n.saturating_sub(1));
                let end = (start + cap).min(n);
                let rows = self.palette_items[start..end]
                    .iter()
                    .map(|c| {
                        let key = if c.keybinding.is_empty() {
                            "\u{2014}".to_string()
                        } else {
                            c.keybinding.clone()
                        };
                        (c.title.clone(), key)
                    })
                    .collect();
                Some(OverlaySpec {
                    title: Some("Keyboard Shortcuts & Commands".to_string()),
                    input: None,
                    rows,
                    left_color: [225, 225, 230],
                    right_color: [200, 170, 110],
                    split_frac: 0.62,
                    selected: None,
                    hint: Some(format!(
                        "{n} commands \u{2014} \u{2191}\u{2193} scroll \u{2022} Esc close"
                    )),
                })
            }
            View::GotoLine => Some(OverlaySpec {
                title: None,
                input: Some(format!("Go to line: {}", self.goto_input)),
                rows: Vec::new(),
                left_color: [225, 225, 230],
                right_color: [135, 135, 150],
                split_frac: 0.62,
                selected: None,
                hint: Some(format!(
                    "1\u{2013}{} \u{2014} Enter jump \u{2022} Esc cancel",
                    self.buffer.len_lines()
                )),
            }),
            View::SshPicker => {
                let cap = self
                    .renderer
                    .as_ref()
                    .map(|r| r.overlay_row_capacity())
                    .unwrap_or(1);
                let n = self.ssh_filtered.len();
                let sel = if n == 0 { 0 } else { self.ssh_sel.min(n - 1) };
                let start = if sel < cap { 0 } else { sel + 1 - cap };
                let end = (start + cap).min(n);
                let mut rows = Vec::with_capacity(end - start);
                for &hi in &self.ssh_filtered[start..end] {
                    rows.push((self.ssh_hosts[hi].clone(), "~/.ssh/config".to_string()));
                }
                Some(OverlaySpec {
                    title: None,
                    input: Some(format!("ssh> {}", self.ssh_input)),
                    rows,
                    left_color: [225, 225, 230],
                    right_color: [135, 135, 150],
                    split_frac: 0.70,
                    selected: if n == 0 { None } else { Some(sel - start) },
                    hint: Some(
                        "Enter connect (selected or typed host) \u{2022} Esc cancel".to_string(),
                    ),
                })
            }
            View::Agents => {
                let cap = self
                    .renderer
                    .as_ref()
                    .map(|r| r.overlay_row_capacity())
                    .unwrap_or(1);
                let (all_rows, model) = self.agents_row_model();
                let n = all_rows.len();
                let sel = if n == 0 {
                    0
                } else {
                    self.agents_scroll.min(n - 1)
                };
                let start = windowed_start(sel, n, cap);
                let end = (start + cap).min(n);
                let selected = if n == 0 || matches!(model.get(sel), Some(AgentsRow::Header)) {
                    None
                } else {
                    Some(sel - start)
                };
                Some(OverlaySpec {
                    title: Some("Agents".to_string()),
                    input: None,
                    rows: all_rows[start..end].to_vec(),
                    left_color: [225, 225, 230],
                    right_color: [200, 170, 110],
                    split_frac: 0.52,
                    selected,
                    hint: Some(
                        "click/Enter open thread \u{2022} h history \u{2022} n new \u{2022} p prompt \u{2022} a abort \u{2022} k kill \u{2022} r refresh \u{2022} Esc"
                            .to_string(),
                    ),
                })
            }
            View::AgentThread => {
                let cap = self
                    .renderer
                    .as_ref()
                    .map(|r| r.overlay_row_capacity())
                    .unwrap_or(1);
                let n = self.agent_thread.len();
                let sel = if n == 0 {
                    0
                } else {
                    self.agent_thread_scroll.min(n - 1)
                };
                let start = windowed_start(sel, n, cap);
                let end = (start + cap).min(n);
                Some(OverlaySpec {
                    title: Some(self.agent_thread_title.clone()),
                    input: None,
                    rows: self.agent_thread[start..end].to_vec(),
                    left_color: [225, 225, 230],
                    right_color: [150, 143, 130],
                    split_frac: 0.85,
                    selected: None,
                    hint: Some(
                        "\u{2191}\u{2193}/wheel scroll \u{2022} Esc back to agents".to_string(),
                    ),
                })
            }
            View::AgentPrompt => Some(OverlaySpec {
                title: None,
                input: Some(format!("prompt> {}", self.agent_prompt)),
                rows: self.agent_live_rows(),
                left_color: [225, 225, 230],
                right_color: [200, 170, 110],
                split_frac: 0.62,
                selected: None,
                hint: Some("Enter send (steer if running) \u{2022} Esc back".to_string()),
            }),
            View::RemoteHost => {
                let rows = self
                    .ssh_hosts
                    .iter()
                    .take(8)
                    .map(|h| (h.clone(), "~/.ssh/config".to_string()))
                    .collect();
                Some(OverlaySpec {
                    title: Some("Remote workspace \u{2014} SSH host".to_string()),
                    input: Some(format!("host> {}", self.remote_host_input)),
                    rows,
                    left_color: [225, 225, 230],
                    right_color: [135, 135, 150],
                    split_frac: 0.7,
                    selected: None,
                    hint: Some(
                        "type a host (runs `ssh <host> umberd`) \u{2022} Enter next \u{2022} Esc"
                            .to_string(),
                    ),
                })
            }
            View::RemotePath => Some(OverlaySpec {
                title: Some(format!("Remote file on {}", self.remote_host_input.trim())),
                input: Some(format!("path> {}", self.remote_path_input)),
                rows: Vec::new(),
                left_color: [225, 225, 230],
                right_color: [135, 135, 150],
                split_frac: 0.62,
                selected: None,
                hint: Some("Enter open (daemon-relative path) \u{2022} Esc cancel".to_string()),
            }),
            View::Search => {
                let cap = self
                    .renderer
                    .as_ref()
                    .map(|r| r.overlay_row_capacity())
                    .unwrap_or(1);
                let n = self.search_results.len();
                let sel = if n == 0 {
                    0
                } else {
                    self.search_sel.min(n - 1)
                };
                let start = if sel < cap { 0 } else { sel + 1 - cap };
                let end = (start + cap).min(n);
                let root = self.project_root();
                let rows = self.search_results[start..end]
                    .iter()
                    .map(|m| {
                        let rel = m
                            .path
                            .strip_prefix(&root)
                            .unwrap_or(&m.path)
                            .display()
                            .to_string();
                        (m.text.clone(), format!("{rel}:{}", m.line))
                    })
                    .collect();
                Some(OverlaySpec {
                    title: None,
                    input: Some(format!("search> {}", self.search_input)),
                    rows,
                    left_color: [220, 220, 225],
                    right_color: [150, 150, 165],
                    split_frac: 0.66,
                    selected: if n == 0 { None } else { Some(sel - start) },
                    hint: Some(format!(
                        "{n} matches \u{2014} \u{2191}\u{2193} select \u{2022} Enter open \u{2022} Esc close"
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
                    ("Open terminal".to_string(), "\u{2192} Ctrl+J".to_string()),
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
                // External modules, tagged with kind + requested permissions.
                for m in &self.modules {
                    let state = if m.loaded { "ON " } else { "OFF" };
                    let detail = match (&m.manifest, &m.error) {
                        (Err(e), _) => format!("{state}  \u{2022}  [module] parse error: {e}"),
                        (Ok(man), Some(err)) => format!(
                            "{state}  \u{2022}  [module {}] {} \u{2014} error: {err}",
                            man.kind.as_str(),
                            man.permissions.summary()
                        ),
                        (Ok(man), None) => format!(
                            "{state}  \u{2022}  [module {}] {}",
                            man.kind.as_str(),
                            man.permissions.summary()
                        ),
                    };
                    rows.push((m.name.clone(), detail));
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
        // A modal is opening: drop any editor hover highlight.
        self.clear_hover();
        self.view = View::Palette;
        self.palette_query.clear();
        self.palette_sel = 0;
        self.rebuild_palette_items();
        self.palette_filtered = self.filter_palette("");
        self.refresh_overlay();
    }

    /// Open the settings page (Ctrl+, / "Preferences: Open Settings").
    fn open_settings(&mut self) {
        self.clear_hover();
        self.view = View::Settings;
        self.settings_sel = 0;
        self.refresh_overlay();
    }

    /// Open the modules page ("Modules: Manage").
    fn open_modules(&mut self) {
        self.clear_hover();
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

    // ===================================================================
    //  P3b/QoL: help overlay, go-to-line, SSH picker.
    // ===================================================================

    /// Open the pi agent dashboard, (re)scanning the session store.
    fn open_agents(&mut self) {
        self.agents_sessions = match agents::sessions_root() {
            Some(root) => agents::discover_sessions(&root, 50),
            None => Vec::new(),
        };
        self.agents_scroll = 0;
        self.view = View::Agents;
        self.refresh_overlay();
    }

    /// Working directory a new live agent is spawned in: the open file's
    /// parent, else the process cwd.
    fn agent_cwd(&self) -> PathBuf {
        self.buffer
            .path()
            .and_then(|p| p.parent().map(|d| d.to_path_buf()))
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")))
    }

    /// One dashboard row for `agents_sessions[i]`.
    fn agents_session_row(&self, i: usize) -> (String, String) {
        let s = &self.agents_sessions[i];
        let home = std::env::var("HOME").unwrap_or_default();
        let cwd = if !home.is_empty() && s.cwd.starts_with(&home) {
            format!("~{}", &s.cwd[home.len()..])
        } else {
            s.cwd.clone()
        };
        let marker = if s.age_secs < 120 {
            "\u{25cf} active"
        } else {
            "idle"
        };
        (
            format!("{}  \u{2014}  {}", s.model, cwd),
            format!(
                "{marker} \u{2022} {} \u{2022} {} tok \u{2022} {} msgs",
                agents::fmt_age(s.age_secs),
                agents::fmt_tokens(s.tokens_total),
                s.messages,
            ),
        )
    }

    /// The Agents page rows + a parallel action map: LIVE section, ACTIVE
    /// sessions (<2m), then a collapsed-by-default History expander.
    fn agents_row_model(&self) -> (Vec<(String, String)>, Vec<AgentsRow>) {
        let mut rows = Vec::new();
        let mut model = Vec::new();
        if self.agent_proc.is_some() {
            rows.push(("LIVE".to_string(), String::new()));
            model.push(AgentsRow::Header);
            for r in self.agent_live_rows() {
                rows.push(r);
                model.push(AgentsRow::Live);
            }
        }
        let active: Vec<usize> = (0..self.agents_sessions.len())
            .filter(|&i| self.agents_sessions[i].age_secs < 120)
            .collect();
        let hist: Vec<usize> = (0..self.agents_sessions.len())
            .filter(|&i| self.agents_sessions[i].age_secs >= 120)
            .collect();
        if !active.is_empty() {
            rows.push(("ACTIVE".to_string(), String::new()));
            model.push(AgentsRow::Header);
            for &i in &active {
                rows.push(self.agents_session_row(i));
                model.push(AgentsRow::Session(i));
            }
        }
        if self.agent_proc.is_none() && active.is_empty() && hist.is_empty() {
            rows.push((
                "No agents yet \u{2014} press n to launch pi here".to_string(),
                String::new(),
            ));
            model.push(AgentsRow::Header);
        }
        if !hist.is_empty() {
            let arrow = if self.agents_expanded {
                "\u{25be}"
            } else {
                "\u{25b8}"
            };
            rows.push((format!("{arrow} History ({})", hist.len()), String::new()));
            model.push(AgentsRow::Expander);
            if self.agents_expanded {
                for &i in &hist {
                    rows.push(self.agents_session_row(i));
                    model.push(AgentsRow::Session(i));
                }
            }
        }
        (rows, model)
    }

    /// Move the Agents selection by `delta`, skipping headers.
    fn agents_move_sel(&mut self, delta: i64) {
        let (rows, model) = self.agents_row_model();
        let n = rows.len();
        if n == 0 {
            return;
        }
        let step = if delta >= 0 { 1i64 } else { -1i64 };
        let mut sel = self.agents_scroll.min(n - 1) as i64;
        let mut remaining = delta.abs();
        while remaining > 0 {
            let next = sel + step;
            if next < 0 || next >= n as i64 {
                break;
            }
            sel = next;
            if !matches!(model[sel as usize], AgentsRow::Header) {
                remaining -= 1;
            }
        }
        while matches!(model.get(sel as usize), Some(AgentsRow::Header)) && sel + 1 < n as i64 {
            sel += 1;
        }
        self.agents_scroll = sel as usize;
        self.refresh_overlay();
    }

    /// Activate the Agents row at `idx` (open thread / toggle history).
    fn agents_activate(&mut self, idx: usize) {
        let (_, model) = self.agents_row_model();
        match model.get(idx) {
            Some(AgentsRow::Live) => self.open_agent_thread_live(),
            Some(AgentsRow::Session(i)) => self.open_agent_thread_session(*i),
            Some(AgentsRow::Expander) => {
                self.agents_expanded = !self.agents_expanded;
                self.refresh_overlay();
            }
            _ => {}
        }
    }

    /// Open the transcript viewer for `agents_sessions[i]` (active branch,
    /// chronological, scrolled to the latest).
    fn open_agent_thread_session(&mut self, i: usize) {
        let Some(s) = self.agents_sessions.get(i) else {
            return;
        };
        let title = format!(
            "{} \u{2014} {} \u{00b7} {}",
            s.model,
            s.cwd,
            agents::fmt_age(s.age_secs)
        );
        let mut rows = Vec::new();
        match std::fs::read_to_string(&s.path) {
            Ok(text) => {
                for (role, msg) in agents::session_transcript(&text, 80) {
                    wrap_message(&mut rows, &role, &msg);
                }
            }
            Err(err) => rows.push((format!("cannot read session: {err}"), String::new())),
        }
        if rows.is_empty() {
            rows.push((
                "(no conversation on the active branch)".to_string(),
                String::new(),
            ));
        }
        self.agent_thread = rows;
        self.agent_thread_title = title;
        self.agent_thread_scroll = self.agent_thread.len().saturating_sub(1);
        self.view = View::AgentThread;
        self.refresh_overlay();
    }

    /// Open the live agent's current thread (state + streamed output tail).
    fn open_agent_thread_live(&mut self) {
        let Some(proc) = self.agent_proc.as_ref() else {
            return;
        };
        let mut rows = self.agent_live_rows();
        rows.push((String::new(), String::new()));
        let tail = proc.state.output_tail();
        for l in tail.lines() {
            rows.push((l.chars().take(140).collect(), String::new()));
        }
        self.agent_thread = rows;
        self.agent_thread_title = "Live agent \u{2014} current thread".to_string();
        self.agent_thread_scroll = self.agent_thread.len().saturating_sub(1);
        self.view = View::AgentThread;
        self.refresh_overlay();
    }

    /// Thread viewer keys: scroll + Esc back to the dashboard.
    fn agent_thread_key(&mut self, event: KeyEvent) {
        match &event.logical_key {
            Key::Named(NamedKey::Escape) => {
                self.view = View::Agents;
                self.refresh_overlay();
            }
            Key::Named(NamedKey::ArrowDown) => {
                let n = self.agent_thread.len();
                self.agent_thread_scroll = (self.agent_thread_scroll + 1).min(n.saturating_sub(1));
                self.refresh_overlay();
            }
            Key::Named(NamedKey::ArrowUp) => {
                self.agent_thread_scroll = self.agent_thread_scroll.saturating_sub(1);
                self.refresh_overlay();
            }
            Key::Named(NamedKey::PageDown) => {
                let n = self.agent_thread.len();
                self.agent_thread_scroll = (self.agent_thread_scroll + 10).min(n.saturating_sub(1));
                self.refresh_overlay();
            }
            Key::Named(NamedKey::PageUp) => {
                self.agent_thread_scroll = self.agent_thread_scroll.saturating_sub(10);
                self.refresh_overlay();
            }
            _ => {}
        }
    }

    fn agents_key(&mut self, event: KeyEvent) {
        match &event.logical_key {
            Key::Named(NamedKey::Escape) => self.close_overlay(),
            Key::Named(NamedKey::ArrowDown) => self.agents_move_sel(1),
            Key::Named(NamedKey::ArrowUp) => self.agents_move_sel(-1),
            Key::Named(NamedKey::PageDown) => self.agents_move_sel(5),
            Key::Named(NamedKey::PageUp) => self.agents_move_sel(-5),
            Key::Named(NamedKey::Enter) => {
                let sel = self.agents_scroll;
                self.agents_activate(sel);
            }
            Key::Character(c) if c.as_str() == "h" => {
                self.agents_expanded = !self.agents_expanded;
                self.refresh_overlay();
            }
            Key::Character(c) if c.as_str() == "r" => {
                let scroll = self.agents_scroll;
                self.open_agents();
                self.agents_scroll = scroll;
                self.refresh_overlay();
            }
            // n: spawn a live pi agent in the working dir (P4 slice 2).
            Key::Character(c) if c.as_str() == "n" => {
                if self.agent_proc.is_none() {
                    let cwd = self.agent_cwd();
                    match AgentProcess::spawn("pi", &cwd, UmberNotifier(self.event_proxy.clone())) {
                        Ok(proc) => self.agent_proc = Some(proc),
                        Err(err) => {
                            eprintln!("umber: pi rpc spawn failed: {err}");
                        }
                    }
                }
                self.refresh_overlay();
            }
            // p: prompt the live agent.
            Key::Character(c) if c.as_str() == "p" => {
                if self.agent_proc.is_some() {
                    self.agent_prompt.clear();
                    self.view = View::AgentPrompt;
                    self.refresh_overlay();
                }
            }
            // a: abort the current run.
            Key::Character(c) if c.as_str() == "a" => {
                if let Some(proc) = self.agent_proc.as_mut() {
                    let _ = proc.abort();
                }
            }
            // k: kill/detach the live agent.
            Key::Character(c) if c.as_str() == "k" => {
                if let Some(mut proc) = self.agent_proc.take() {
                    proc.shutdown();
                }
                self.refresh_overlay();
            }
            _ => {}
        }
    }

    /// Keyboard handling for the live-agent prompt sub-view.
    fn agent_prompt_key(&mut self, event: KeyEvent) {
        match &event.logical_key {
            Key::Named(NamedKey::Escape) => {
                self.view = View::Agents;
                self.refresh_overlay();
            }
            Key::Named(NamedKey::Enter) => {
                let text = std::mem::take(&mut self.agent_prompt);
                if !text.trim().is_empty() {
                    if let Some(proc) = self.agent_proc.as_mut() {
                        // Running -> steer; idle -> a fresh prompt (§1.2).
                        let behavior = match proc.state.run_state() {
                            Some(AgentRunState::Running) | Some(AgentRunState::Queued) => {
                                Some("steer")
                            }
                            _ => None,
                        };
                        if let Err(err) = proc.prompt(&text, behavior) {
                            eprintln!("umber: prompt send failed: {err}");
                        }
                    }
                }
                self.view = View::Agents;
                self.refresh_overlay();
            }
            Key::Named(NamedKey::Backspace) => {
                self.agent_prompt.pop();
                self.refresh_overlay();
            }
            _ => {
                if let Some(t) = &event.text {
                    let mut changed = false;
                    for ch in t.chars() {
                        if !ch.is_control() {
                            self.agent_prompt.push(ch);
                            changed = true;
                        }
                    }
                    if changed {
                        self.refresh_overlay();
                    }
                }
            }
        }
    }

    /// Live-agent header rows for the dashboard: attach state, current tool,
    /// and a tail of streamed output. Empty when no agent is attached.
    fn agent_live_rows(&self) -> Vec<(String, String)> {
        let Some(proc) = self.agent_proc.as_ref() else {
            return Vec::new();
        };
        let state = match proc.state.run_state() {
            Some(AgentRunState::Starting) => "\u{25cc} starting\u{2026}",
            Some(AgentRunState::Running) => "\u{25cf} RUNNING",
            Some(AgentRunState::AwaitingInstruction) => "\u{25c9} NEEDS RESPONSE \u{2014} press p",
            Some(AgentRunState::Queued) => "\u{25d0} queued work",
            Some(AgentRunState::Exited) => "\u{2715} exited",
            None => "?",
        };
        let tool = proc
            .state
            .last_tool()
            .map(|t| format!("tool: {t}"))
            .unwrap_or_default();
        let mut rows = vec![(format!("LIVE AGENT \u{2014} {state}"), tool)];
        // Last line of streamed output as a preview.
        let tail = proc.state.output_tail();
        if let Some(last) = tail.lines().last() {
            if !last.trim().is_empty() {
                let preview: String = last
                    .chars()
                    .rev()
                    .take(60)
                    .collect::<Vec<_>>()
                    .into_iter()
                    .rev()
                    .collect();
                rows.push((format!("  {preview}"), String::new()));
            }
        }
        rows
    }

    /// Connect the SSH picker's selected (or typed) host in the terminal tab.
    fn ssh_connect_selected(&mut self) {
        let host = self
            .ssh_filtered
            .get(self.ssh_sel)
            .map(|&i| self.ssh_hosts[i].clone())
            .or_else(|| {
                let t = self.ssh_input.trim().to_string();
                if t.is_empty() {
                    None
                } else {
                    Some(t)
                }
            });
        self.view = View::Editor;
        self.close_overlay();
        if let Some(host) = host {
            self.open_terminal_session(Some(("ssh".to_string(), vec![host])));
        }
    }

    /// Hovering an overlay row moves the selection (click activates). Applied
    /// only while the list window is unscrolled — hover-selecting inside a
    /// scrolled window would shift the window under the pointer.
    fn overlay_hover_row(&mut self, row: usize) {
        let cap = self
            .renderer
            .as_ref()
            .map(|r| r.overlay_row_capacity())
            .unwrap_or(1);
        match self.view {
            View::Palette => {
                let n = self.palette_filtered.len();
                if windowed_start(self.palette_sel, n, cap) == 0
                    && row < n
                    && self.palette_sel != row
                {
                    self.palette_sel = row;
                    self.refresh_overlay();
                }
            }
            View::Settings => {
                let r2 = row.min(SETTINGS_ROWS - 1);
                if self.settings_sel != r2 {
                    self.settings_sel = r2;
                    self.refresh_overlay();
                }
            }
            View::Modules => {
                let n = self.features.features().len() + self.modules.len();
                if n > 0 {
                    let idx = row.min(n - 1);
                    if self.modules_sel != idx {
                        self.modules_sel = idx;
                        self.refresh_overlay();
                    }
                }
            }
            View::Search => {
                let n = self.search_results.len();
                if windowed_start(self.search_sel, n, cap) == 0 && row < n && self.search_sel != row
                {
                    self.search_sel = row;
                    self.refresh_overlay();
                }
            }
            View::Agents => {
                let (rows, model) = self.agents_row_model();
                let n = rows.len();
                if windowed_start(self.agents_scroll.min(n.saturating_sub(1)), n, cap) == 0
                    && row < n
                    && self.agents_scroll != row
                    && !matches!(model.get(row), Some(AgentsRow::Header) | None)
                {
                    self.agents_scroll = row;
                    self.refresh_overlay();
                }
            }
            View::SshPicker => {
                let n = self.ssh_filtered.len();
                if windowed_start(self.ssh_sel, n, cap) == 0 && row < n && self.ssh_sel != row {
                    self.ssh_sel = row;
                    self.refresh_overlay();
                }
            }
            _ => {}
        }
    }

    /// Wheel scrolling on overlay pages: bump the selection / scroll offset.
    fn overlay_scroll(&mut self, steps: i64) {
        let mag = steps.unsigned_abs() as usize;
        let down = steps > 0;
        fn bump(sel: usize, n: usize, down: bool, mag: usize) -> usize {
            if n == 0 {
                0
            } else if down {
                (sel + mag).min(n - 1)
            } else {
                sel.saturating_sub(mag)
            }
        }
        match self.view {
            View::Palette => {
                self.palette_sel = bump(self.palette_sel, self.palette_filtered.len(), down, mag)
            }
            View::Search => {
                self.search_sel = bump(self.search_sel, self.search_results.len(), down, mag)
            }
            View::SshPicker => {
                self.ssh_sel = bump(self.ssh_sel, self.ssh_filtered.len(), down, mag)
            }
            View::Settings => self.settings_sel = bump(self.settings_sel, SETTINGS_ROWS, down, mag),
            View::Modules => {
                let n = self.features.features().len() + self.modules.len();
                self.modules_sel = bump(self.modules_sel, n, down, mag)
            }
            View::Help => {
                self.help_scroll = bump(self.help_scroll, self.palette_items.len(), down, mag)
            }
            View::Agents => {
                let n = self.agents_row_model().0.len();
                self.agents_scroll = bump(self.agents_scroll, n, down, mag)
            }
            View::AgentThread => {
                let n = self.agent_thread.len();
                self.agent_thread_scroll = bump(self.agent_thread_scroll, n, down, mag)
            }
            _ => return,
        }
        self.refresh_overlay();
    }

    fn open_help(&mut self) {
        self.view = View::Help;
        self.help_scroll = 0;
        self.refresh_overlay();
    }

    fn open_goto(&mut self) {
        self.view = View::GotoLine;
        self.goto_input.clear();
        self.refresh_overlay();
    }

    fn open_ssh_picker(&mut self) {
        self.ssh_hosts = ssh_config_hosts();
        self.ssh_input.clear();
        self.view = View::SshPicker;
        self.ssh_refilter();
    }

    fn help_key(&mut self, event: KeyEvent) {
        match &event.logical_key {
            Key::Named(NamedKey::Escape) => self.close_overlay(),
            Key::Named(NamedKey::ArrowDown) | Key::Named(NamedKey::PageDown) => {
                let n = self.palette_items.len();
                self.help_scroll = (self.help_scroll + 1).min(n.saturating_sub(1));
                self.refresh_overlay();
            }
            Key::Named(NamedKey::ArrowUp) | Key::Named(NamedKey::PageUp) => {
                self.help_scroll = self.help_scroll.saturating_sub(1);
                self.refresh_overlay();
            }
            _ => {}
        }
    }

    fn goto_key(&mut self, event: KeyEvent) {
        match &event.logical_key {
            Key::Named(NamedKey::Escape) => self.close_overlay(),
            Key::Named(NamedKey::Enter) => {
                let target: Option<usize> = self.goto_input.trim().parse().ok();
                self.view = View::Editor;
                if let Some(n) = target {
                    let last = self.buffer.len_lines();
                    let line = n.clamp(1, last.max(1)) - 1;
                    self.buffer.break_coalescing();
                    self.selection_anchor = None;
                    self.cursor_char = self.buffer.line_to_char(line);
                    self.update_goal_col();
                }
                self.close_overlay();
                self.apply_view(true);
                if let Some(r) = self.renderer.as_ref() {
                    r.window().request_redraw();
                }
            }
            Key::Named(NamedKey::Backspace) => {
                self.goto_input.pop();
                self.refresh_overlay();
            }
            _ => {
                if let Some(text) = &event.text {
                    let mut changed = false;
                    for ch in text.chars() {
                        if ch.is_ascii_digit() {
                            self.goto_input.push(ch);
                            changed = true;
                        }
                    }
                    if changed {
                        self.refresh_overlay();
                    }
                }
            }
        }
    }

    /// Re-filter SSH hosts against the typed query (simple case-insensitive
    /// substring — host lists are short).
    fn ssh_refilter(&mut self) {
        let q = self.ssh_input.to_lowercase();
        self.ssh_filtered = self
            .ssh_hosts
            .iter()
            .enumerate()
            .filter(|(_, h)| q.is_empty() || h.to_lowercase().contains(&q))
            .map(|(i, _)| i)
            .collect();
        self.ssh_sel = 0;
        self.refresh_overlay();
    }

    fn ssh_key(&mut self, event: KeyEvent) {
        match &event.logical_key {
            Key::Named(NamedKey::Escape) => self.close_overlay(),
            Key::Named(NamedKey::Enter) => self.ssh_connect_selected(),
            Key::Named(NamedKey::ArrowDown) => {
                let n = self.ssh_filtered.len();
                if n > 0 {
                    self.ssh_sel = (self.ssh_sel + 1) % n;
                }
                self.refresh_overlay();
            }
            Key::Named(NamedKey::ArrowUp) => {
                let n = self.ssh_filtered.len();
                if n > 0 {
                    self.ssh_sel = (self.ssh_sel + n - 1) % n;
                }
                self.refresh_overlay();
            }
            Key::Named(NamedKey::Backspace) => {
                self.ssh_input.pop();
                self.ssh_refilter();
            }
            _ => {
                if let Some(text) = &event.text {
                    let mut changed = false;
                    for ch in text.chars() {
                        if !ch.is_control() && !ch.is_whitespace() {
                            self.ssh_input.push(ch);
                            changed = true;
                        }
                    }
                    if changed {
                        self.ssh_refilter();
                    }
                }
            }
        }
    }

    /// Recompute the palette filter after the query changed.
    fn repalette(&mut self) {
        self.palette_filtered = self.filter_palette(&self.palette_query);
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
                    .map(|&i| self.palette_items[i].id.clone());
                self.view = View::Editor;
                match id {
                    Some(id) => self.execute_command(&id, event_loop),
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
            6 => {
                // Action row: open the terminal panel from settings.
                self.close_overlay();
                self.open_terminal();
                return;
            }
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
                let n = self.features.features().len() + self.modules.len();
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
        let feature_count = self.features.features().len();
        if self.modules_sel < feature_count {
            // Built-in feature (D10): the toggle mirrors the config booleans.
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
            return;
        }
        // External module: toggle load/unload live and persist the enabled set.
        let idx = self.modules_sel - feature_count;
        if idx >= self.modules.len() {
            return;
        }
        if self.modules[idx].loaded {
            self.unload_module(idx);
            self.modules_enabled.remove(&self.modules[idx].name);
            self.modules_hint = None;
        } else {
            self.load_module(idx);
            if self.modules[idx].loaded {
                self.modules_enabled.insert(self.modules[idx].name.clone());
                self.modules_hint = None;
            } else {
                // Surface the load failure; the app stays alive.
                self.modules_hint = self.modules[idx].error.clone();
            }
        }
        self.save_modules_enabled();
        self.refresh_overlay();
    }

    /// Toggle a feature by id (from a palette command). Kernel entries no-op,
    /// leaving a hint for the modules page.
    // ===================================================================
    //  P3: embedded terminal panel.
    // ===================================================================

    /// Open (spawning the shell on first use) and focus the terminal panel.
    fn open_terminal(&mut self) {
        self.open_terminal_session(None);
    }

    /// Open + focus the panel running `shell` (None = `$SHELL`). An explicit
    /// `shell` (e.g. the SSH picker's `ssh <host>`) replaces a live session;
    /// plain opens reuse it.
    fn open_terminal_session(&mut self, shell: Option<(String, Vec<String>)>) {
        if !self.features.is_enabled("terminal") {
            self.modules_hint = Some("terminal feature is disabled".to_string());
            return;
        }
        if shell.is_some() {
            if let Some(mut old) = self.terminal.take() {
                old.shutdown();
            }
        }
        let Some(renderer) = self.renderer.as_mut() else {
            return;
        };
        renderer.set_terminal(true, true);
        self.term_focused = true;
        let (cols, lines) = renderer.term_grid_size();
        let (cw, ch) = renderer.cell_px();
        match &self.terminal {
            None => {
                match TerminalSession::spawn_with_shell(
                    UmberNotifier(self.event_proxy.clone()),
                    cols,
                    lines,
                    cw,
                    ch,
                    shell,
                ) {
                    Ok(session) => self.terminal = Some(session),
                    Err(err) => {
                        eprintln!("umber: terminal spawn failed: {err}");
                        if let Some(r) = self.renderer.as_mut() {
                            r.set_terminal(false, false);
                        }
                        self.term_focused = false;
                        return;
                    }
                }
            }
            // Reopening after a hide: re-sync the PTY to the panel grid.
            Some(session) => session.resize(cols, lines, cw, ch),
        }
        self.clear_hover();
        self.apply_view(false);
        if let Some(r) = self.renderer.as_ref() {
            r.window().request_redraw();
        }
    }

    /// Kill the shell and reap it (feature disable, child exit, quit).
    fn kill_terminal(&mut self) {
        if let Some(mut session) = self.terminal.take() {
            session.shutdown();
        }
        self.term_focused = false;
        self.term_tab_active = false;
        if let Some(renderer) = self.renderer.as_mut() {
            renderer.set_terminal_maximized(false);
            renderer.set_terminal(false, false);
        }
        self.apply_view(false);
        if let Some(r) = self.renderer.as_ref() {
            r.window().request_redraw();
        }
    }

    /// Show the terminal as a full content tab (spawning the shell if
    /// needed) and focus it.
    fn activate_terminal_tab(&mut self) {
        self.open_terminal_session(None);
        if self.terminal.is_none() {
            return; // spawn failed; open_terminal_session reported it
        }
        self.term_tab_active = true;
        let grid = if let Some(r) = self.renderer.as_mut() {
            r.set_terminal_maximized(true);
            let g = r.term_grid_size();
            let c = r.cell_px();
            r.window().request_redraw();
            Some((g, c))
        } else {
            None
        };
        if let (Some(((cols, lines), (cw, ch))), Some(s)) = (grid, self.terminal.as_ref()) {
            s.resize(cols, lines, cw, ch);
        }
        self.apply_view(false);
    }

    /// Leave the terminal tab, returning to the active document tab. The
    /// shell session stays alive for the next visit.
    fn deactivate_terminal_tab(&mut self) {
        self.term_tab_active = false;
        self.term_focused = false;
        if let Some(r) = self.renderer.as_mut() {
            r.set_terminal_maximized(false);
            r.set_terminal(false, false);
            r.window().request_redraw();
        }
        self.apply_view(false);
    }

    /// Ctrl+`/Ctrl+J: toggle the terminal content tab.
    fn terminal_toggle(&mut self) {
        if self.term_tab_active {
            self.deactivate_terminal_tab();
        } else {
            self.activate_terminal_tab();
        }
    }

    /// Drag-resize: set the terminal split from a pointer-y and resize the PTY.
    fn terminal_resize_to(&mut self, pointer_y: f64) {
        let grid = if let Some(r) = self.renderer.as_mut() {
            if !r.terminal_open() || r.terminal_maximized() {
                return;
            }
            let (_, h) = r.size();
            if h == 0 {
                return;
            }
            let frac = (((h as f64 - pointer_y) / h as f64).clamp(0.1, 0.85)) as f32;
            r.set_terminal_split_frac(frac);
            let (cols, lines) = r.term_grid_size();
            let (cw, ch) = r.cell_px();
            r.window().request_redraw();
            Some((cols, lines, cw, ch))
        } else {
            None
        };
        if let (Some((cols, lines, cw, ch)), Some(s)) = (grid, self.terminal.as_ref()) {
            s.resize(cols, lines, cw, ch);
        }
        self.apply_view(false);
    }

    /// Toggle fullscreen terminal, resizing the PTY grid to match.
    fn terminal_toggle_max(&mut self) {
        let grid = if let Some(r) = self.renderer.as_mut() {
            if !r.terminal_open() {
                return;
            }
            let m = !r.terminal_maximized();
            r.set_terminal_maximized(m);
            let (cols, lines) = r.term_grid_size();
            let (cw, ch) = r.cell_px();
            r.window().request_redraw();
            Some((cols, lines, cw, ch))
        } else {
            None
        };
        if let (Some((cols, lines, cw, ch)), Some(s)) = (grid, self.terminal.as_ref()) {
            s.resize(cols, lines, cw, ch);
        }
        self.apply_view(false);
    }

    /// Activate a left tab-bar tab (0 palette, 1 find, 2 agents, 3 terminal,
    /// 4 settings) — the mouse backup for the keyboard commands.
    /// Expand/collapse the left activity bar (icons only <-> icons + labels).
    fn toggle_sidebar(&mut self) {
        if let Some(r) = self.renderer.as_mut() {
            let e = !r.sidebar_expanded();
            r.set_sidebar_expanded(e);
        }
        self.apply_view(false);
        if let Some(r) = self.renderer.as_ref() {
            r.window().request_redraw();
        }
    }

    /// Push the open-document tab labels (with dirty markers) + active index
    /// to the renderer's tab strip.
    fn sync_tabs(&mut self) {
        let mut labels: Vec<String> = Vec::with_capacity(self.docs.len());
        for i in 0..self.docs.len() {
            let dirty = if i == self.active_doc {
                self.buffer.is_dirty()
            } else {
                self.docs[i].buffer.is_dirty()
            };
            let name = self.tab_name(i);
            labels.push(if dirty {
                format!("\u{2022} {name}")
            } else {
                name
            });
        }
        // The terminal (when spawned) appears as the last tab in the list.
        if self.terminal.is_some() {
            labels.push("\u{25b8} Terminal".to_string());
        }
        let active = if self.term_tab_active {
            self.docs.len()
        } else {
            self.active_doc
        };
        if let Some(r) = self.renderer.as_mut() {
            r.set_sidebar_tabs(&labels, active);
        }
        self.sync_activity_strip();
    }

    /// Feed the TOP activity strip (Palette/Find/Agents/Terminal/Settings)
    /// and tint the action matching the current view. `usize::MAX` = none
    /// (the renderer's active lookup is range-checked).
    fn sync_activity_strip(&mut self) {
        let labels: Vec<String> = ["Palette", "Find", "Agents", "Terminal", "Settings"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let active = match self.view {
            View::Palette => 0,
            View::Search => 1,
            View::Agents | View::AgentPrompt | View::AgentThread => 2,
            View::Editor if self.term_focused => 3,
            View::Settings => 4,
            _ => usize::MAX,
        };
        if let Some(r) = self.renderer.as_mut() {
            r.set_tabs(&labels, active);
        }
    }

    fn sidebar_tab_activate(&mut self, tab: usize) {
        match tab {
            0 => self.open_palette(),
            1 => self.open_search(),
            2 => self.open_agents(),
            3 => {
                self.close_overlay();
                self.terminal_toggle();
            }
            4 => self.open_settings(),
            _ => {}
        }
    }

    /// Left-click activation on an overlay list row (window-relative `row`):
    /// select it, or activate it when it was already selected.
    fn overlay_click_row(&mut self, row: usize, event_loop: &ActiveEventLoop) {
        match self.view {
            View::Settings => {
                if self.settings_sel == row {
                    self.settings_adjust(1); // toggle bool / bump numeric
                } else {
                    self.settings_sel = row.min(SETTINGS_ROWS - 1);
                }
                self.refresh_overlay();
            }
            View::Modules => {
                let n = self.features.features().len() + self.modules.len();
                let idx = row.min(n.saturating_sub(1));
                if self.modules_sel == idx {
                    self.modules_toggle_current();
                } else {
                    self.modules_sel = idx;
                    self.refresh_overlay();
                }
            }
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
                let abs = start + row;
                if abs < n {
                    let id = self.palette_items[self.palette_filtered[abs]].id.clone();
                    self.view = View::Editor;
                    self.execute_command(&id, event_loop);
                }
            }
            View::Search => {
                let cap = self
                    .renderer
                    .as_ref()
                    .map(|r| r.overlay_row_capacity())
                    .unwrap_or(1);
                let n = self.search_results.len();
                let sel = if n == 0 {
                    0
                } else {
                    self.search_sel.min(n - 1)
                };
                let start = if sel < cap { 0 } else { sel + 1 - cap };
                let abs = start + row;
                if abs < n {
                    self.search_sel = abs;
                    self.open_search_result();
                }
            }
            View::Agents => {
                let cap = self
                    .renderer
                    .as_ref()
                    .map(|r| r.overlay_row_capacity())
                    .unwrap_or(1);
                let n = self.agents_row_model().0.len();
                let abs = windowed_start(self.agents_scroll.min(n.saturating_sub(1)), n, cap) + row;
                if abs < n {
                    self.agents_scroll = abs;
                    self.agents_activate(abs);
                }
            }
            View::SshPicker => {
                let cap = self
                    .renderer
                    .as_ref()
                    .map(|r| r.overlay_row_capacity())
                    .unwrap_or(1);
                let n = self.ssh_filtered.len();
                let abs = windowed_start(self.ssh_sel, n, cap) + row;
                if abs < n {
                    if self.ssh_sel == abs {
                        self.ssh_connect_selected();
                    } else {
                        self.ssh_sel = abs;
                        self.refresh_overlay();
                    }
                }
            }
            _ => {}
        }
    }

    /// Encode a terminal-focused keystroke as PTY bytes. Esc never reaches the
    /// shell (it returns focus to the editor); Ctrl+letter becomes the C0
    /// control byte, so Ctrl+C is SIGINT to the PTY, deliberately NOT the
    /// editor copy command.
    fn term_key_bytes(event: &KeyEvent, ctrl: bool) -> Option<Vec<u8>> {
        match &event.logical_key {
            Key::Named(NamedKey::Enter) => Some(b"\r".to_vec()),
            Key::Named(NamedKey::Backspace) => Some(vec![0x7f]),
            Key::Named(NamedKey::Tab) => Some(b"\t".to_vec()),
            Key::Named(NamedKey::ArrowUp) => Some(b"\x1b[A".to_vec()),
            Key::Named(NamedKey::ArrowDown) => Some(b"\x1b[B".to_vec()),
            Key::Named(NamedKey::ArrowRight) => Some(b"\x1b[C".to_vec()),
            Key::Named(NamedKey::ArrowLeft) => Some(b"\x1b[D".to_vec()),
            Key::Named(NamedKey::Home) => Some(b"\x1b[H".to_vec()),
            Key::Named(NamedKey::End) => Some(b"\x1b[F".to_vec()),
            Key::Named(NamedKey::PageUp) => Some(b"\x1b[5~".to_vec()),
            Key::Named(NamedKey::PageDown) => Some(b"\x1b[6~".to_vec()),
            Key::Named(NamedKey::Delete) => Some(b"\x1b[3~".to_vec()),
            Key::Character(c) if ctrl => {
                let ch = c.chars().next()?;
                let lower = ch.to_ascii_lowercase();
                if lower.is_ascii_alphabetic() {
                    Some(vec![(lower as u8) & 0x1f])
                } else {
                    None
                }
            }
            _ => event.text.as_ref().map(|t| t.as_bytes().to_vec()),
        }
    }

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

    // ===================================================================
    //  P2: external module host wiring (D9).
    // ===================================================================

    /// Discover `~/.config/umber/modules/*/umber.toml` and load the enabled
    /// ones. The enabled set persists in the host's sidecar; on first run (no
    /// sidecar yet) it is seeded from each manifest's `default_on`. Called once
    /// at startup, before the event loop. Never fatal: a missing dir, a bad
    /// manifest, or a load failure is recorded, not raised.
    fn init_modules(&mut self) {
        let Some(dir) = umber_host::modules_dir() else {
            return;
        };
        let enabled_path = umber_host::enabled_path();
        let had_sidecar = enabled_path.as_ref().map(|p| p.exists()).unwrap_or(false);
        self.modules_enabled = enabled_path
            .as_ref()
            .map(|p| umber_host::load_enabled(p))
            .unwrap_or_default();

        for d in umber_host::discover(&dir) {
            let name = d.name().to_string();
            let manifest = d.manifest.map_err(|e| e.to_string());
            // First run: seed the enabled set from `default_on` so a freshly
            // dropped-in module appears on without the user toggling it.
            if !had_sidecar {
                if let Ok(m) = &manifest {
                    if m.default_on {
                        self.modules_enabled.insert(name.clone());
                    }
                }
            }
            self.modules.push(ExternalModule {
                name,
                base_dir: d.base_dir,
                manifest,
                loaded: false,
                error: None,
            });
        }

        for idx in 0..self.modules.len() {
            if self.modules_enabled.contains(&self.modules[idx].name) {
                self.load_module(idx);
            }
        }
        if !had_sidecar {
            self.save_modules_enabled();
        }
    }

    /// Load module `idx` into the host, appending its commands. Records a load
    /// error on the entry instead of raising. A module whose manifest failed to
    /// parse cannot be loaded.
    fn load_module(&mut self, idx: usize) {
        let (manifest, base_dir) = match &self.modules[idx].manifest {
            Ok(m) => (m.clone(), self.modules[idx].base_dir.clone()),
            Err(_) => {
                self.modules[idx].error = Some("manifest failed to parse".to_string());
                return;
            }
        };
        let Some(host) = self.module_host.as_mut() else {
            self.modules[idx].error = Some("module host unavailable".to_string());
            return;
        };
        match host.load(manifest, &base_dir) {
            Ok(cmds) => {
                self.module_commands.extend(cmds);
                self.modules[idx].loaded = true;
                self.modules[idx].error = None;
            }
            Err(e) => {
                self.modules[idx].loaded = false;
                self.modules[idx].error = Some(e.to_string());
            }
        }
    }

    /// Unload module `idx`, dropping its commands from the palette source.
    fn unload_module(&mut self, idx: usize) {
        let name = self.modules[idx].name.clone();
        if let Some(host) = self.module_host.as_mut() {
            let removed = host.unload(&name);
            self.module_commands
                .retain(|c| !removed.iter().any(|r| r == &c.id));
        }
        self.modules[idx].loaded = false;
        self.modules[idx].error = None;
    }

    /// Persist the enabled-module set to the host's sidecar file.
    fn save_modules_enabled(&self) {
        if let Some(path) = umber_host::enabled_path() {
            let _ = umber_host::save_enabled(&path, &self.modules_enabled);
        }
    }

    /// Invoke an external-module command by id, capturing its first output line
    /// (or the error) into the status banner. Never panics on a bad module.
    fn invoke_module(&mut self, id: &str) {
        let status = match self.module_host.as_mut() {
            Some(host) => match host.invoke(id) {
                Ok(text) => {
                    let line = text.lines().next().unwrap_or("").trim();
                    if line.is_empty() {
                        format!("{id}: ran (no output)")
                    } else {
                        format!("{id}: {line}")
                    }
                }
                Err(e) => format!("{id} failed: {e}"),
            },
            None => format!("{id}: module host unavailable"),
        };
        self.module_status = Some(status);
    }

    /// Rebuild the unified palette source: built-in commands followed by the
    /// currently-loaded module commands.
    fn rebuild_palette_items(&mut self) {
        let mut items =
            Vec::with_capacity(self.commands.commands().len() + self.module_commands.len());
        for c in self.commands.commands() {
            items.push(PaletteItem {
                id: c.id.to_string(),
                title: c.title.to_string(),
                keybinding: c.keybinding.to_string(),
            });
        }
        for c in &self.module_commands {
            items.push(PaletteItem {
                id: c.id.clone(),
                title: c.title.clone(),
                keybinding: "module".to_string(),
            });
        }
        self.palette_items = items;
    }

    /// Filter+rank the unified palette items against `query`, reusing the
    /// kernel's fuzzy scorer so built-ins and module commands rank alike.
    fn filter_palette(&self, query: &str) -> Vec<usize> {
        if query.trim().is_empty() {
            return (0..self.palette_items.len()).collect();
        }
        let mut scored: Vec<(usize, i32)> = self
            .palette_items
            .iter()
            .enumerate()
            .filter_map(|(i, it)| umber_kernel::fuzzy_score(&it.title, query).map(|s| (i, s)))
            .collect();
        scored.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        scored.into_iter().map(|(i, _)| i).collect()
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
            "view.toggle.terminal" => self.toggle_feature("terminal"),
            "terminal.toggle" => {
                self.close_overlay();
                self.terminal_toggle();
                return;
            }
            "terminal.focus" => {
                self.close_overlay();
                self.open_terminal();
                return;
            }
            "terminal.maximize" => {
                self.close_overlay();
                if !self
                    .renderer
                    .as_ref()
                    .map(|r| r.terminal_open())
                    .unwrap_or(false)
                {
                    self.open_terminal();
                }
                self.terminal_toggle_max();
                return;
            }
            "view.toggleSidebar" => {
                self.toggle_sidebar();
                return;
            }
            "view.nextTab" => {
                self.next_tab();
                return;
            }
            "view.closeTab" => {
                self.close_active_tab();
                return;
            }
            "terminal.ssh" => {
                self.open_ssh_picker();
                return;
            }
            "goto.line" => {
                self.open_goto();
                return;
            }
            "help.keys" => {
                self.open_help();
                return;
            }
            "agents.dashboard" => {
                self.open_agents();
                return;
            }
            "remote.open" => {
                self.remote_host_input.clear();
                self.ssh_hosts = ssh_config_hosts();
                self.view = View::RemoteHost;
                self.refresh_overlay();
                return;
            }
            "remote.disconnect" => {
                if let Some(mut ws) = self.remote.take() {
                    ws.shutdown();
                }
                self.remote_file = None;
                self.close_overlay();
                return;
            }
            "search.project" => {
                self.open_search();
                return;
            }
            // Not a built-in id: route to the module host (external command).
            other => self.invoke_module(other),
        }
        // In-place command finished: return to the editor and repaint.
        self.close_overlay();
        if follow {
            self.apply_view(true);
        }
    }
}

impl ApplicationHandler<UserEvent> for App {
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

    fn user_event(&mut self, _event_loop: &ActiveEventLoop, event: UserEvent) {
        match event {
            UserEvent::TerminalWakeup => {
                let Some(session) = self.terminal.as_ref() else {
                    return;
                };
                // take_dirty BEFORE content(): the coalescing contract — any
                // parser progress after the clear re-arms a fresh wakeup.
                if session.take_dirty() {
                    let (text, cursor) = session.content();
                    if let Some(renderer) = self.renderer.as_mut() {
                        renderer.set_terminal_text(&text, cursor);
                        if renderer.terminal_open() {
                            renderer.window().request_redraw();
                        }
                    }
                }
            }
            UserEvent::TerminalExited => {
                // Shell ended (exit / Ctrl+D): close the panel and reap.
                self.kill_terminal();
            }
            UserEvent::AgentUpdated => {
                // Live agent changed state/output: refresh the dashboard if
                // it (or its prompt sub-view) is open.
                if matches!(self.view, View::Agents | View::AgentPrompt) {
                    self.refresh_overlay();
                }
            }
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
                // P3: keep the PTY grid in step with the resized panel.
                if let (Some(session), Some(renderer)) =
                    (self.terminal.as_ref(), self.renderer.as_ref())
                {
                    if renderer.terminal_open() {
                        let (cols, lines) = renderer.term_grid_size();
                        let (cw, ch) = renderer.cell_px();
                        session.resize(cols, lines, cw, ch);
                    }
                }
                // Modal overlays are shaped to the surface width at set_overlay
                // time; a resize while one is open must re-spec it or its text
                // stays laid out for the old geometry.
                if self.view != View::Editor {
                    self.refresh_overlay();
                }
                self.apply_view(false);
                // The window geometry changed, so the pointer now maps to a
                // different cell: drop the (possibly stale) hover highlight.
                self.clear_hover();
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
                self.clear_hover();
                if let Some(renderer) = self.renderer.as_ref() {
                    renderer.window().request_redraw();
                }
            }

            WindowEvent::ModifiersChanged(modifiers) => {
                self.modifiers = modifiers.state();
            }

            WindowEvent::MouseWheel { delta, .. } => {
                // Overlay pages: the wheel moves the selection / scroll.
                if self.view != View::Editor {
                    let steps = match delta {
                        MouseScrollDelta::LineDelta(_, y) => -y as i64,
                        MouseScrollDelta::PixelDelta(p) => (-p.y / BASE_LINE_PX) as i64,
                    };
                    if steps != 0 {
                        self.overlay_scroll(steps);
                    }
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
                    // The document scrolled under the pointer: the text the
                    // hover pointed at moved, so drop it (redraws once).
                    self.clear_hover();
                }
                if let Some(renderer) = self.renderer.as_ref() {
                    renderer.window().request_redraw();
                }
            }

            WindowEvent::CursorLeft { .. } => {
                // Pointer left the window: drop the hover highlight, or the
                // last gold word would linger until an edit or re-entry.
                self.clear_hover();
            }

            WindowEvent::CursorMoved { position, .. } => {
                self.pointer = (position.x, position.y);
                // Sidebar separator: dragging resizes; hovering swaps in a
                // col-resize cursor and lights the line.
                if self.sidebar_resizing {
                    if let Some(r) = self.renderer.as_mut() {
                        r.set_sidebar_width_px(position.x as f32);
                        r.window().request_redraw();
                    }
                    self.apply_view(false);
                    return;
                }
                let edge_hot = self
                    .renderer
                    .as_ref()
                    .map(|r| r.sidebar_edge_hit(position.x as f32))
                    .unwrap_or(false);
                if let Some(r) = self.renderer.as_mut() {
                    if r.set_sidebar_edge_hot(edge_hot) {
                        r.window().set_cursor(if edge_hot {
                            winit::window::CursorIcon::ColResize
                        } else {
                            winit::window::CursorIcon::Default
                        });
                        r.window().request_redraw();
                    }
                }
                // Sidebar hover highlight (redraw only on change).
                if let Some(r) = self.renderer.as_mut() {
                    let tab = r.sidebar_tab_at(position.x as f32, position.y as f32);
                    if r.set_sidebar_hover(tab) {
                        r.window().request_redraw();
                    }
                }
                // Top-strip action hover (any view; redraw only on change).
                if let Some(r) = self.renderer.as_mut() {
                    let tab = r.tabstrip_at(position.x as f32, position.y as f32);
                    if r.set_tabstrip_hover(tab) {
                        r.window().request_redraw();
                    }
                }
                // Overlay pages: hovering a row moves the selection.
                if self.view != View::Editor {
                    if let Some(row) = self
                        .renderer
                        .as_ref()
                        .and_then(|r| r.overlay_row_at(position.y as f32))
                    {
                        self.overlay_hover_row(row);
                    }
                    return;
                }
                if self.term_resizing {
                    self.terminal_resize_to(position.y);
                } else if self.scrollbar_dragging {
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
                    // Pointer is over the scrollbar chrome, not text: drop any
                    // word/segment hover (redraws once only if one was showing).
                    self.clear_hover();
                } else if self.view == View::Editor {
                    // Document hover: map pointer -> target; redraw ONLY when the
                    // target changes, never on raw motion.
                    self.update_hover();
                }
            }

            WindowEvent::MouseInput { state, button, .. } => {
                // Middle-click a file tab (left bar) closes it.
                if button == MouseButton::Middle && state == ElementState::Pressed {
                    if let Some(i) = self.renderer.as_ref().and_then(|r| {
                        r.sidebar_tab_at(self.pointer.0 as f32, self.pointer.1 as f32)
                    }) {
                        if i >= self.docs.len() {
                            // Terminal tab: middle-click kills the shell.
                            self.kill_terminal();
                        } else {
                            self.close_tab(i);
                        }
                    }
                    return;
                }
                if button != MouseButton::Left {
                    return;
                }
                // Left tab bar: works from any view (mouse backup for the
                // palette / find / agents / terminal / settings commands).
                if state == ElementState::Released {
                    // End a separator drag no matter which view is up.
                    self.sidebar_resizing = false;
                }
                if state == ElementState::Pressed {
                    // Grab the sidebar separator to resize it.
                    if self
                        .renderer
                        .as_ref()
                        .map(|r| r.sidebar_edge_hit(self.pointer.0 as f32))
                        .unwrap_or(false)
                    {
                        self.sidebar_resizing = true;
                        return;
                    }
                    // Left bar = open file tabs: click switches.
                    if let Some(tab) = self.renderer.as_ref().and_then(|r| {
                        r.sidebar_tab_at(self.pointer.0 as f32, self.pointer.1 as f32)
                    }) {
                        if self.view != View::Editor {
                            self.close_overlay();
                        }
                        if tab >= self.docs.len() {
                            // Last row = the terminal tab.
                            self.activate_terminal_tab();
                        } else {
                            if self.term_tab_active {
                                self.deactivate_terminal_tab();
                            }
                            self.switch_tab(tab);
                        }
                        return;
                    }
                    // Top strip = activity actions: click activates.
                    if let Some(i) = self
                        .renderer
                        .as_ref()
                        .and_then(|r| r.tabstrip_at(self.pointer.0 as f32, self.pointer.1 as f32))
                    {
                        self.sidebar_tab_activate(i);
                        return;
                    }
                }
                // Overlay pages: click outside the panel closes; a click on a
                // row selects it, and clicking the already-selected row
                // activates it (toggle / open / run).
                if self.view != View::Editor {
                    if state == ElementState::Pressed {
                        if let Some((px, py, pw, ph)) =
                            self.renderer.as_ref().map(|r| r.overlay_panel_bounds())
                        {
                            let mx = self.pointer.0 as f32;
                            let my = self.pointer.1 as f32;
                            if mx < px || mx > px + pw || my < py || my > py + ph {
                                self.close_overlay();
                                return;
                            }
                        }
                        if let Some(row) = self
                            .renderer
                            .as_ref()
                            .and_then(|r| r.overlay_row_at(self.pointer.1 as f32))
                        {
                            self.overlay_click_row(row, event_loop);
                        }
                    }
                    return;
                }
                match state {
                    ElementState::Pressed => {
                        let t = Instant::now();
                        // A press changes the caret/selection context under the
                        // pointer: drop any hover highlight (redraws once).
                        self.clear_hover();
                        // Terminal top border: start a drag-resize (a few px
                        // band around the border, when not maximized).
                        if let Some(r) = self.renderer.as_ref() {
                            if r.terminal_open() && !r.terminal_maximized() {
                                let py = self.pointer.1 as f32;
                                if (py - r.term_top()).abs() <= 5.0 {
                                    self.term_resizing = true;
                                    return;
                                }
                            }
                        }
                        // P3: clicks in the terminal panel move focus there;
                        // clicks in the document return it to the editor.
                        if let Some(renderer) = self.renderer.as_ref() {
                            if renderer.terminal_open()
                                && self.pointer.1 as f32 >= renderer.term_top()
                            {
                                self.term_focused = true;
                                if let Some(r) = self.renderer.as_mut() {
                                    r.set_terminal(true, true);
                                    r.window().request_redraw();
                                }
                                return;
                            }
                        }
                        if self.term_focused {
                            self.term_focused = false;
                            if let Some(r) = self.renderer.as_mut() {
                                if r.terminal_open() {
                                    r.set_terminal(true, false);
                                }
                            }
                        }
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
                            // Double-click selects the word under the pointer.
                            let now2 = Instant::now();
                            let double = self
                                .last_click_at
                                .map(|t0| now2.duration_since(t0) < Duration::from_millis(400))
                                .unwrap_or(false)
                                && self.last_click_pos == Some(pos);
                            self.last_click_at = Some(now2);
                            self.last_click_pos = Some(pos);
                            if double {
                                let line = self.buffer.char_to_line(pos);
                                let ls = self.buffer.line_to_char(line);
                                let text = self.buffer.visible_text(line, 1);
                                let col = pos - ls;
                                if let Some((ws, we)) =
                                    word_span_at(text.lines().next().unwrap_or(""), col)
                                {
                                    self.buffer.break_coalescing();
                                    self.selection_anchor = Some(ls + ws);
                                    self.cursor_char = ls + we;
                                    self.update_goal_col();
                                    self.selecting = false;
                                    self.apply_view(true);
                                    if let Some(r) = self.renderer.as_ref() {
                                        r.window().request_redraw();
                                    }
                                    return;
                                }
                            }
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
                        self.term_resizing = false;
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
                // P3: terminal focus owns the keyboard exclusively. Ctrl+`
                // stays global (toggle/unfocus) and Esc returns to the editor;
                // everything else becomes PTY bytes. These keys are NOT D4
                // latency samples — the ring measures editor keystrokes only.
                if self.view == View::Editor && self.term_focused {
                    let ctrl = self.modifiers.control_key();
                    if ctrl && matches!(&event.logical_key, Key::Character(c) if c.as_str() == "`")
                    {
                        self.terminal_toggle();
                        return;
                    }
                    if matches!(&event.logical_key, Key::Named(NamedKey::Escape)) {
                        // Esc leaves the terminal tab back to the document.
                        self.deactivate_terminal_tab();
                        return;
                    }
                    // F11: toggle fullscreen terminal.
                    if matches!(&event.logical_key, Key::Named(NamedKey::F11)) {
                        self.terminal_toggle_max();
                        return;
                    }
                    if let (Some(session), Some(bytes)) =
                        (self.terminal.as_ref(), Self::term_key_bytes(&event, ctrl))
                    {
                        session.write(bytes);
                    }
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
                    View::Help => {
                        self.help_key(event);
                        return;
                    }
                    View::GotoLine => {
                        self.goto_key(event);
                        return;
                    }
                    View::SshPicker => {
                        self.ssh_key(event);
                        return;
                    }
                    View::Agents => {
                        self.agents_key(event);
                        return;
                    }
                    View::AgentPrompt => {
                        self.agent_prompt_key(event);
                        return;
                    }
                    View::AgentThread => {
                        self.agent_thread_key(event);
                        return;
                    }
                    View::RemoteHost => {
                        self.remote_host_key(event);
                        return;
                    }
                    View::RemotePath => {
                        self.remote_path_key(event);
                        return;
                    }
                    View::Search => {
                        self.search_key(event);
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

                // QoL: F1 opens the help overlay from the editor.
                if matches!(&event.logical_key, Key::Named(NamedKey::F1)) {
                    self.open_help();
                    return;
                }

                // Ctrl+Tab cycles open editor tabs (Tab is a Named key, so it
                // is handled before the Character chords below).
                if ctrl && matches!(&event.logical_key, Key::Named(NamedKey::Tab)) {
                    self.next_tab();
                    return;
                }

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
                            "`" | "j" => {
                                self.terminal_toggle();
                                return;
                            }
                            "g" => {
                                self.open_goto();
                                return;
                            }
                            "b" => {
                                self.toggle_sidebar();
                                return;
                            }
                            "w" => {
                                self.close_active_tab();
                                return;
                            }
                            "q" => {
                                event_loop.exit();
                                return;
                            }
                            "a" if shift => {
                                self.open_agents();
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
                    let prev_first = self.first_visible_line;
                    self.apply_view(true);
                    if let Some(renderer) = self.renderer.as_mut() {
                        if changed {
                            renderer.mark_keystroke(t);
                        }
                        renderer.window().request_redraw();
                    }
                    // Edit changed the text, or the caret move scrolled the
                    // view: either way the text under the pointer moved, so drop
                    // the hover highlight (redraws once, coalesced above).
                    if changed || self.first_visible_line != prev_first {
                        self.clear_hover();
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

/// True for a "word" char: alphanumeric (Unicode) or underscore. Punctuation
/// and whitespace are not word chars.
fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// Column span (char indices, `end` exclusive) of the hover target at `col` in
/// `line`. `None` when `col` is past the last char or on whitespace (the caller
/// treats that as a line-only hover). A word char expands left/right over the
/// maximal run of word chars; a punctuation char is a single-char span. `col`
/// and the returned bounds are char indices, not bytes.
fn word_span_at(line: &str, col: usize) -> Option<(usize, usize)> {
    let chars: Vec<char> = line.chars().collect();
    let c = *chars.get(col)?;
    if c.is_whitespace() {
        return None;
    }
    if is_word_char(c) {
        let mut start = col;
        while start > 0 && is_word_char(chars[start - 1]) {
            start -= 1;
        }
        let mut end = col + 1;
        while end < chars.len() && is_word_char(chars[end]) {
            end += 1;
        }
        Some((start, end))
    } else {
        // Punctuation: a single-char word.
        Some((col, col + 1))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn word_span_expands_over_word_chars() {
        // Hovering anywhere inside an identifier selects the whole run;
        // underscores and digits are word chars.
        let line = "foo_bar baz";
        assert_eq!(word_span_at(line, 0), Some((0, 7)));
        assert_eq!(word_span_at(line, 3), Some((0, 7)));
        assert_eq!(word_span_at(line, 6), Some((0, 7)));
        assert_eq!(word_span_at(line, 8), Some((8, 11)));
    }

    #[test]
    fn word_span_whitespace_is_none() {
        // A space is not a word char -> line-only hover.
        assert_eq!(word_span_at("foo bar", 3), None);
    }

    #[test]
    fn word_span_past_end_is_none() {
        assert_eq!(word_span_at("hi", 2), None);
        assert_eq!(word_span_at("hi", 5), None);
        assert_eq!(word_span_at("", 0), None);
    }

    #[test]
    fn word_span_punctuation_is_single_char() {
        // Punctuation counts as a single-char word, even when adjacent.
        let line = "a::b";
        assert_eq!(word_span_at(line, 0), Some((0, 1)));
        assert_eq!(word_span_at(line, 1), Some((1, 2)));
        assert_eq!(word_span_at(line, 2), Some((2, 3)));
        assert_eq!(word_span_at(line, 3), Some((3, 4)));
    }

    #[test]
    fn word_span_unicode_word_by_char_index() {
        // is_alphanumeric() is Unicode-aware; col is a char index, not a byte.
        assert_eq!(word_span_at("h\u{e9}llo", 1), Some((0, 5)));
    }

    #[test]
    fn hover_target_change_detection() {
        let w = HoverTarget::Word {
            line: 2,
            start_col: 0,
            end_col: 3,
        };
        // Identical targets are equal (no redraw).
        assert_eq!(
            w,
            HoverTarget::Word {
                line: 2,
                start_col: 0,
                end_col: 3
            }
        );
        // Any field differing is a change (redraw).
        assert_ne!(
            w,
            HoverTarget::Word {
                line: 2,
                start_col: 0,
                end_col: 4
            }
        );
        assert_ne!(
            w,
            HoverTarget::Word {
                line: 3,
                start_col: 0,
                end_col: 3
            }
        );
        // Word vs line on the same line is a change (word recolor vs segment).
        assert_ne!(w, HoverTarget::Line(2));
        // Line vs line, and None vs anything.
        assert_ne!(HoverTarget::Line(1), HoverTarget::Line(2));
        assert_eq!(HoverTarget::Line(5), HoverTarget::Line(5));
        assert_ne!(HoverTarget::None, HoverTarget::Line(0));
    }
}

#[cfg(test)]
mod ssh_config_tests {
    use super::parse_ssh_hosts;

    #[test]
    fn parses_hosts_skipping_wildcards_and_negations() {
        let cfg = "# comment\nHost moo\n  HostName 1.2.3.4\n\nhost dev staging\nHost *\n  Compression yes\nHost *.internal !bastion prod\n";
        assert_eq!(parse_ssh_hosts(cfg), vec!["dev", "moo", "prod", "staging"]);
    }

    #[test]
    fn empty_or_hostless_config_yields_nothing() {
        assert!(parse_ssh_hosts("").is_empty());
        assert!(parse_ssh_hosts("Port 22\nUser root\n").is_empty());
    }
}
