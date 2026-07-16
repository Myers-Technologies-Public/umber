//! Lua module backend (D9) — the same ABI v1 surface as WASM, expressed
//! idiomatically through a host-provided `umber` table.
//!
//! # Lua ABI v1
//!
//! The host injects a global table `umber` before running the module chunk:
//!
//! - `umber.register(id, title[, handler])` — declare a command. `handler` is
//!   an optional Lua function run on invocation; if omitted, invocation calls a
//!   global `umber_invoke(id)` fallback (mirroring the WASM export).
//! - `umber.emit(text)` — append UTF-8 text to the current invocation output.
//! - `umber.config_get(key) -> string | nil` — read a manifest-declared config
//!   value.
//!
//! Loading executes the chunk (declaring commands); invocation runs the stored
//! handler. Every invocation runs under an instruction-count hook that checks a
//! wall-clock deadline and aborts a runaway script — the Lua-tier equivalent of
//! the WASM epoch deadline.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;
use std::time::{Duration, Instant};

use mlua::{HookTriggers, Lua, LuaOptions, MultiValue, StdLib, Value, VmState};

use crate::HostError;

/// Marker embedded in the deadline error so [`LuaModule::invoke`] can tell a
/// timeout apart from an ordinary script error.
const TIMEOUT_MARK: &str = "umber-host:deadline";

/// A loaded Lua module. Commands' handler functions live in a Lua table kept in
/// the registry (`_umber_handlers`), keyed by command id.
pub struct LuaModule {
    lua: Lua,
    output: Rc<RefCell<String>>,
    ids: Vec<String>,
}

impl LuaModule {
    /// Run `script`, wiring up the `umber` table, and collect the commands it
    /// declares. `config` is captured for `umber.config_get`.
    pub fn load(
        script: &str,
        config: BTreeMap<String, String>,
    ) -> Result<(Self, Vec<(String, String)>), HostError> {
        // Deny-all sandbox (D9): only pure-computation stdlibs. `Lua::new()`
        // would load mlua's ALL_SAFE set, which includes `io` and `os`
        // (filesystem + os.execute) — exactly what modules must not have.
        // `package`/require and `debug` are likewise excluded.
        let lua = Lua::new_with(
            StdLib::STRING | StdLib::TABLE | StdLib::MATH | StdLib::COROUTINE,
            LuaOptions::default(),
        )
        .map_err(|e| HostError::Load(format!("lua init: {e}")))?;
        let output = Rc::new(RefCell::new(String::new()));
        // Declared commands accumulate here during chunk execution.
        let declared: Rc<RefCell<Vec<(String, String)>>> = Rc::new(RefCell::new(Vec::new()));

        // Handler table, stored globally so handlers survive past `load`.
        let handlers = lua
            .create_table()
            .map_err(|e| HostError::Load(format!("lua table: {e}")))?;

        let umber = lua
            .create_table()
            .map_err(|e| HostError::Load(format!("lua table: {e}")))?;

        // umber.emit(text)
        {
            let out = output.clone();
            let emit = lua
                .create_function(move |_, text: String| {
                    out.borrow_mut().push_str(&text);
                    Ok(())
                })
                .map_err(|e| HostError::Load(format!("lua emit: {e}")))?;
            umber
                .set("emit", emit)
                .map_err(|e| HostError::Load(format!("lua set emit: {e}")))?;
        }

        // umber.config_get(key) -> string | nil
        {
            let cfg = config.clone();
            let get = lua
                .create_function(move |_, key: String| Ok(cfg.get(&key).cloned()))
                .map_err(|e| HostError::Load(format!("lua config_get: {e}")))?;
            umber
                .set("config_get", get)
                .map_err(|e| HostError::Load(format!("lua set config_get: {e}")))?;
        }

        // umber.register(id, title[, handler])
        {
            let decl = declared.clone();
            let handlers_ref = handlers.clone();
            let register = lua
                .create_function(
                    move |_, (id, title, handler): (String, String, Option<mlua::Function>)| {
                        decl.borrow_mut().push((id.clone(), title));
                        if let Some(h) = handler {
                            handlers_ref.set(id, h)?;
                        }
                        Ok(())
                    },
                )
                .map_err(|e| HostError::Load(format!("lua register: {e}")))?;
            umber
                .set("register", register)
                .map_err(|e| HostError::Load(format!("lua set register: {e}")))?;
        }

        lua.globals()
            .set("umber", umber)
            .map_err(|e| HostError::Load(format!("lua set umber: {e}")))?;
        lua.globals()
            .set("_umber_handlers", handlers)
            .map_err(|e| HostError::Load(format!("lua set handlers: {e}")))?;

        lua.load(script)
            .exec()
            .map_err(|e| HostError::Load(format!("lua chunk: {e}")))?;

        let commands = declared.borrow().clone();
        let ids = commands.iter().map(|(id, _)| id.clone()).collect();
        Ok((LuaModule { lua, output, ids }, commands))
    }

    /// Run the command `id` under a `budget` wall-clock deadline. A runaway
    /// script is aborted by the instruction hook and reported as
    /// [`HostError::Timeout`]; the host (and this module) stay usable.
    pub fn invoke(&mut self, id: &str, budget: Duration) -> Result<String, HostError> {
        if !self.ids.iter().any(|i| i == id) {
            return Err(HostError::NoSuchCommand(id.to_string()));
        }
        self.output.borrow_mut().clear();

        let deadline = Instant::now() + budget;
        // Installing the deadline hook is what makes a runaway script
        // interruptible; if it can't be installed we must not run unguarded.
        self.lua
            .set_hook(
                HookTriggers::new().every_nth_instruction(10_000),
                move |_lua, _debug| {
                    if Instant::now() >= deadline {
                        Err(mlua::Error::runtime(TIMEOUT_MARK))
                    } else {
                        Ok(VmState::Continue)
                    }
                },
            )
            .map_err(|e| HostError::Invoke(format!("set hook: {e}")))?;

        let result = self.call_handler(id);
        self.lua.remove_hook();

        match result {
            Ok(()) => Ok(self.output.borrow().clone()),
            Err(e) => {
                if e.to_string().contains(TIMEOUT_MARK) {
                    Err(HostError::Timeout)
                } else {
                    Err(HostError::Invoke(format!("{e}")))
                }
            }
        }
    }

    /// Call the stored handler for `id`, or the `umber_invoke(id)` global
    /// fallback if none was registered.
    fn call_handler(&self, id: &str) -> Result<(), mlua::Error> {
        let handlers: mlua::Table = self.lua.globals().get("_umber_handlers")?;
        let handler: Value = handlers.get(id)?;
        match handler {
            Value::Function(f) => f.call::<()>(id.to_string()),
            _ => {
                let fallback: Value = self.lua.globals().get("umber_invoke")?;
                match fallback {
                    Value::Function(f) => f.call::<()>(id.to_string()),
                    _ => Err(mlua::Error::runtime(format!(
                        "no handler for command `{id}` and no umber_invoke fallback"
                    ))),
                }
            }
        }
    }
}

/// Silence an unused-import lint for `MultiValue` on toolchains that don't need
/// it; kept because mlua's call sites are version-sensitive.
#[allow(dead_code)]
fn _abi_touch(_: Option<MultiValue>) {}
