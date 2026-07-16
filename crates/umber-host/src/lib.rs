//! umber-host — sandboxed module host + permission broker; mlua runtime for
//! the Lua tier (D9). The P2 "identity phase" core: load/unload modules at
//! runtime, register their commands, invoke them under a time budget, and read
//! their manifest-declared config — all behind a deny-by-default permission
//! broker.
//!
//! Two backends share one **Host ABI v1** (see [`wasm`] and [`lua`] module
//! docs): Rust/TS/AssemblyScript compiled to a core wasm module, and Lua via
//! mlua. v1 is deliberately tiny — plain wasmtime funcs with a stable naming
//! convention, *not* the component model / WIT machinery (that is planned v2).
//!
//! A module can: (a) declare the commands it provides `(id, title)`,
//! (b) be invoked when one of its commands runs, (c) emit UTF-8 text output,
//! and (d) read its own manifest-declared `[config]` keys. It has no other I/O:
//! the [`PermissionBroker`] stores every requested capability but grants none.
//!
//! # Interruption
//!
//! A hung module must never freeze the editor. Both backends enforce a
//! per-invocation budget (default [`DEFAULT_BUDGET`], ~100 ms, configurable):
//! - **wasm:** wasmtime *epoch* interruption. A single background ticker thread
//!   advances the engine epoch every [`TICK`]; each invoke sets an epoch
//!   deadline, so a runaway guest traps and only that call unwinds.
//! - **lua:** an mlua instruction-count hook checks the wall clock and aborts
//!   the handler once the budget elapses.

mod broker;
mod lua;
mod manifest;
mod wasm;

pub use broker::{Capability, Decision, PermissionBroker};
pub use manifest::{Manifest, ManifestError, ModuleKind, Permissions};

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

use wasmtime::{Config as WasmConfig, Engine};

use lua::LuaModule;
use wasm::WasmModule;

/// Default per-invocation time budget (~100 ms). Configurable via
/// [`ModuleHost::with_budget`].
pub const DEFAULT_BUDGET: Duration = Duration::from_millis(100);

/// Engine-epoch ticker interval. The wasm invocation deadline is expressed as a
/// whole number of these ticks.
pub const TICK: Duration = Duration::from_millis(10);

/// Everything that can go wrong loading or invoking a module.
#[derive(Debug)]
pub enum HostError {
    /// The invocation ran past its time budget and was interrupted.
    Timeout,
    /// Compilation / instantiation / script-load failure.
    Load(String),
    /// The module violated the ABI (missing export, bad version, ...).
    Abi(String),
    /// The command handler trapped or errored at runtime.
    Invoke(String),
    /// No loaded module provides this command id.
    NoSuchCommand(String),
    /// The manifest was invalid.
    Manifest(ManifestError),
    /// A module with this name is already loaded.
    AlreadyLoaded(String),
    /// The entry file could not be read.
    Io(String),
}

impl fmt::Display for HostError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HostError::Timeout => write!(f, "invocation exceeded its time budget"),
            HostError::Load(e) => write!(f, "load failed: {e}"),
            HostError::Abi(e) => write!(f, "ABI error: {e}"),
            HostError::Invoke(e) => write!(f, "invocation failed: {e}"),
            HostError::NoSuchCommand(id) => write!(f, "no loaded module provides `{id}`"),
            HostError::Manifest(e) => write!(f, "{e}"),
            HostError::AlreadyLoaded(n) => write!(f, "module `{n}` is already loaded"),
            HostError::Io(e) => write!(f, "cannot read module entry: {e}"),
        }
    }
}

impl std::error::Error for HostError {}

impl From<ManifestError> for HostError {
    fn from(e: ManifestError) -> Self {
        HostError::Manifest(e)
    }
}

/// A command a loaded module provides.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HostCommand {
    pub id: String,
    pub title: String,
}

/// Which backend runs a loaded module.
enum Backend {
    Wasm(WasmModule),
    Lua(LuaModule),
}

/// A module currently resident in the host.
struct LoadedModule {
    manifest: Manifest,
    #[allow(dead_code)] // v1: stored + surfaced; consulted once the ABI has I/O (v2).
    broker: PermissionBroker,
    backend: Backend,
    command_ids: Vec<String>,
}

/// The module host: one wasmtime [`Engine`] (+ its epoch ticker) and the set of
/// resident modules, keyed by manifest name.
pub struct ModuleHost {
    engine: Engine,
    budget: Duration,
    /// `ceil(budget / TICK)`, the wasm epoch deadline in ticks.
    deadline_ticks: u64,
    modules: BTreeMap<String, LoadedModule>,
    /// command id -> owning module name (so [`ModuleHost::invoke`] routes).
    command_owner: BTreeMap<String, String>,
    ticker_stop: Arc<AtomicBool>,
    ticker: Option<JoinHandle<()>>,
}

impl ModuleHost {
    /// Build a host with the [`DEFAULT_BUDGET`].
    pub fn new() -> Result<Self, HostError> {
        Self::with_budget(DEFAULT_BUDGET)
    }

    /// Build a host with an explicit per-invocation `budget`.
    pub fn with_budget(budget: Duration) -> Result<Self, HostError> {
        let mut config = WasmConfig::new();
        config.epoch_interruption(true);
        let engine = Engine::new(&config).map_err(|e| HostError::Load(e.to_string()))?;

        // One ticker thread drives epoch interruption for every module/invoke.
        let ticker_stop = Arc::new(AtomicBool::new(false));
        let ticker = {
            let engine = engine.clone();
            let stop = ticker_stop.clone();
            std::thread::Builder::new()
                .name("umber-host-epoch".to_string())
                .spawn(move || {
                    while !stop.load(Ordering::Relaxed) {
                        std::thread::sleep(TICK);
                        engine.increment_epoch();
                    }
                })
                .map_err(|e| HostError::Load(e.to_string()))?
        };

        let ticks = budget.as_millis().div_ceil(TICK.as_millis()).max(1) as u64;
        Ok(ModuleHost {
            engine,
            budget,
            deadline_ticks: ticks,
            modules: BTreeMap::new(),
            command_owner: BTreeMap::new(),
            ticker_stop,
            ticker: Some(ticker),
        })
    }

    /// Load the module described by `manifest`, resolving `entry` relative to
    /// `base_dir`. Instantiates the backend, runs its registration, and records
    /// the commands it provides. Returns the provided commands (which the caller
    /// registers into the command palette).
    pub fn load(
        &mut self,
        manifest: Manifest,
        base_dir: &Path,
    ) -> Result<Vec<HostCommand>, HostError> {
        if self.modules.contains_key(&manifest.name) {
            return Err(HostError::AlreadyLoaded(manifest.name.clone()));
        }

        let entry_path = base_dir.join(&manifest.entry);
        let broker = PermissionBroker::new(manifest.permissions.clone());
        let config = manifest.config.clone();

        let (backend, registered) = match manifest.kind {
            ModuleKind::Wasm => {
                let bytes = std::fs::read(&entry_path).map_err(|e| HostError::Io(e.to_string()))?;
                let (module, registered) =
                    WasmModule::load(&self.engine, &bytes, config, self.deadline_ticks)?;
                (Backend::Wasm(module), registered)
            }
            ModuleKind::Lua => {
                let script = std::fs::read_to_string(&entry_path)
                    .map_err(|e| HostError::Io(e.to_string()))?;
                let (module, registered) = LuaModule::load(&script, config)?;
                (Backend::Lua(module), registered)
            }
        };

        let mut commands = Vec::with_capacity(registered.len());
        let mut command_ids = Vec::with_capacity(registered.len());
        for (id, title) in registered {
            self.command_owner.insert(id.clone(), manifest.name.clone());
            command_ids.push(id.clone());
            commands.push(HostCommand { id, title });
        }

        self.modules.insert(
            manifest.name.clone(),
            LoadedModule {
                manifest,
                broker,
                backend,
                command_ids,
            },
        );
        Ok(commands)
    }

    /// Convenience: parse `<dir>/umber.toml` and [`load`](Self::load) it.
    pub fn load_dir(&mut self, dir: &Path) -> Result<Vec<HostCommand>, HostError> {
        let manifest = Manifest::from_path(&dir.join("umber.toml"))?;
        self.load(manifest, dir)
    }

    /// Unload `name`, dropping its instance and deregistering its commands.
    /// Returns the command ids that were removed (so the caller can drop them
    /// from the palette). Unknown names return an empty list.
    pub fn unload(&mut self, name: &str) -> Vec<String> {
        match self.modules.remove(name) {
            Some(module) => {
                for id in &module.command_ids {
                    self.command_owner.remove(id);
                }
                module.command_ids
            }
            None => Vec::new(),
        }
    }

    /// Invoke `command_id`, returning the UTF-8 text the module emitted.
    pub fn invoke(&mut self, command_id: &str) -> Result<String, HostError> {
        let owner = self
            .command_owner
            .get(command_id)
            .cloned()
            .ok_or_else(|| HostError::NoSuchCommand(command_id.to_string()))?;
        let module = self
            .modules
            .get_mut(&owner)
            .ok_or_else(|| HostError::NoSuchCommand(command_id.to_string()))?;
        match &mut module.backend {
            Backend::Wasm(m) => m.invoke(command_id, self.deadline_ticks),
            Backend::Lua(m) => m.invoke(command_id, self.budget),
        }
    }

    /// Whether a module with `name` is loaded.
    pub fn is_loaded(&self, name: &str) -> bool {
        self.modules.contains_key(name)
    }

    /// The manifest of a loaded module.
    pub fn manifest(&self, name: &str) -> Option<&Manifest> {
        self.modules.get(name).map(|m| &m.manifest)
    }

    /// Names of all loaded modules.
    pub fn loaded(&self) -> Vec<String> {
        self.modules.keys().cloned().collect()
    }

    /// The per-invocation time budget.
    pub fn budget(&self) -> Duration {
        self.budget
    }
}

impl Drop for ModuleHost {
    fn drop(&mut self) {
        self.ticker_stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.ticker.take() {
            let _ = handle.join();
        }
    }
}

// ===========================================================================
// Discovery + enabled-set persistence
// ===========================================================================

/// `$XDG_CONFIG_HOME/umber/modules`, else `$HOME/.config/umber/modules`. Mirrors
/// umber-kernel's `Config::path` resolution so modules sit beside the config.
pub fn modules_dir() -> Option<PathBuf> {
    config_root().map(|r| r.join("modules"))
}

/// Path of the newline-delimited enabled-module set
/// (`$CONFIG/umber/modules-enabled`). Kept as a sidecar rather than in
/// `config.toml` so the kernel's scalar-only config parser stays untouched.
pub fn enabled_path() -> Option<PathBuf> {
    config_root().map(|r| r.join("modules-enabled"))
}

fn config_root() -> Option<PathBuf> {
    if let Some(xdg) = std::env::var_os("XDG_CONFIG_HOME") {
        if !xdg.is_empty() {
            return Some(PathBuf::from(xdg).join("umber"));
        }
    }
    let home = std::env::var_os("HOME")?;
    Some(PathBuf::from(home).join(".config").join("umber"))
}

/// One entry found by [`discover`]: its directory and its parsed (or failed)
/// manifest. A bad `umber.toml` is surfaced, never fatal.
pub struct Discovered {
    /// Directory name (fallback identity when the manifest failed to parse).
    pub dir_name: String,
    pub base_dir: PathBuf,
    pub manifest: Result<Manifest, ManifestError>,
}

impl Discovered {
    /// The user-facing name: the manifest `name` when it parsed, else the
    /// directory name.
    pub fn name(&self) -> &str {
        match &self.manifest {
            Ok(m) => &m.name,
            Err(_) => &self.dir_name,
        }
    }
}

/// Scan `dir` for `*/umber.toml`, parsing each. Missing `dir` yields an empty
/// list. Results are sorted by directory name for a stable modules page.
pub fn discover(dir: &Path) -> Vec<Discovered> {
    let mut found = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return found;
    };
    for entry in entries.flatten() {
        let base_dir = entry.path();
        if !base_dir.is_dir() {
            continue;
        }
        let toml = base_dir.join("umber.toml");
        if !toml.is_file() {
            continue;
        }
        let dir_name = entry.file_name().to_string_lossy().into_owned();
        found.push(Discovered {
            dir_name,
            manifest: Manifest::from_path(&toml),
            base_dir,
        });
    }
    found.sort_by(|a, b| a.dir_name.cmp(&b.dir_name));
    found
}

/// Read the enabled-module set from `path` (one name per line; missing file =
/// empty set).
pub fn load_enabled(path: &Path) -> BTreeSet<String> {
    match std::fs::read_to_string(path) {
        Ok(text) => text
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .map(str::to_string)
            .collect(),
        Err(_) => BTreeSet::new(),
    }
}

/// Persist the enabled-module set to `path`, creating parent directories.
pub fn save_enabled(path: &Path, enabled: &BTreeSet<String>) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut body =
        String::from("# Umber enabled modules (one per line). Managed by the modules page.\n");
    for name in enabled {
        body.push_str(name);
        body.push('\n');
    }
    std::fs::write(path, body)
}
