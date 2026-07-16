//! WASM module backend — Host ABI v1 (deliberately tiny; the full
//! component-model/WIT machinery is the planned v2).
//!
//! # ABI v1 — exact surface
//!
//! A module is a **core** WebAssembly module (not a component). Strings are
//! passed as `(ptr: i32, len: i32)` pairs into the module's exported `memory`,
//! UTF-8, no allocator required on either side (the guest hands the host its
//! own static buffers).
//!
//! ## Imports the host provides (module namespace `"umber"`)
//!
//! - `register(id_ptr: i32, id_len: i32, title_ptr: i32, title_len: i32)`
//!   — called from within `umber_register`; declares one command
//!   `(id, title)`. Registration order defines the command's invocation index.
//! - `emit(ptr: i32, len: i32)` — append UTF-8 text to the current
//!   invocation's output buffer.
//! - `config_get(key_ptr: i32, key_len: i32, out_ptr: i32, out_cap: i32) -> i32`
//!   — copy the manifest-declared config value for `key` into guest memory at
//!   `out_ptr` (up to `out_cap` bytes); returns the byte length written, `-1`
//!   if the key is absent, or `-2` if the value did not fit in `out_cap`.
//!
//! ## Exports the host requires / calls
//!
//! - `memory` — the guest linear memory (required).
//! - `umber_abi_version() -> i32` — optional; if present it **must** return `1`.
//! - `umber_register()` — called once after instantiation; the guest declares
//!   its commands here via the `register` import.
//! - `umber_invoke(cmd_index: i32)` — called to run the command at its
//!   registration index; the guest produces output via `emit`.
//!
//! Every invocation runs under an epoch deadline (see [`crate`]): a hung module
//! traps with `Trap::Interrupt` and the host reports a timeout instead of
//! freezing.

use std::collections::BTreeMap;
use std::collections::HashMap;

use wasmtime::{Caller, Engine, Linker, Memory, Module, Store, Trap, TypedFunc};

use crate::HostError;

/// Per-store host state the ABI imports read and write.
struct WasmState {
    /// Guest linear memory, set once after instantiation.
    memory: Option<Memory>,
    /// Manifest-declared config keys (`config_get` source).
    config: BTreeMap<String, String>,
    /// Commands declared during `umber_register` (id, title), in order.
    registered: Vec<(String, String)>,
    /// Output accumulated during the current `umber_invoke`.
    output: String,
}

/// A loaded, instantiated WASM module.
pub struct WasmModule {
    store: Store<WasmState>,
    invoke: TypedFunc<i32, ()>,
    /// Command id -> registration index (the arg passed to `umber_invoke`).
    index_of: HashMap<String, i32>,
}

impl WasmModule {
    /// Instantiate `bytes` (a `.wasm` binary or, with the `wat` feature, `.wat`
    /// text) and run `umber_register`. Returns the module and its declared
    /// `(id, title)` commands in registration order. `deadline_ticks` bounds the
    /// guest code run *during load* (the ABI handshake + `umber_register`), just
    /// as [`WasmModule::invoke`] bounds each call.
    pub fn load(
        engine: &Engine,
        bytes: &[u8],
        config: BTreeMap<String, String>,
        deadline_ticks: u64,
    ) -> Result<(Self, Vec<(String, String)>), HostError> {
        let module =
            Module::new(engine, bytes).map_err(|e| HostError::Load(format!("compile: {e}")))?;

        let mut store = Store::new(
            engine,
            WasmState {
                memory: None,
                config,
                registered: Vec::new(),
                output: String::new(),
            },
        );

        // Epoch interruption is always armed on this engine and the ticker has
        // already advanced the epoch, so guest code run here (the abi handshake
        // and `umber_register`) needs its own deadline or the very first call
        // traps immediately against the store's default (zero) deadline.
        store.set_epoch_deadline(deadline_ticks);

        let mut linker: Linker<WasmState> = Linker::new(engine);
        define_imports(&mut linker)?;

        let instance = linker
            .instantiate(&mut store, &module)
            .map_err(|e| HostError::Load(format!("instantiate: {e}")))?;

        let memory = instance
            .get_memory(&mut store, "memory")
            .ok_or_else(|| HostError::Abi("module does not export `memory`".to_string()))?;
        store.data_mut().memory = Some(memory);

        // Optional ABI version handshake: if the export exists it must be v1.
        if let Some(ver) = instance
            .get_typed_func::<(), i32>(&mut store, "umber_abi_version")
            .ok()
        {
            let v = ver
                .call(&mut store, ())
                .map_err(|e| HostError::Abi(format!("umber_abi_version trapped: {e}")))?;
            if v != 1 {
                return Err(HostError::Abi(format!(
                    "unsupported ABI version {v} (host speaks v1)"
                )));
            }
        }

        // Declare commands.
        let register = instance
            .get_typed_func::<(), ()>(&mut store, "umber_register")
            .map_err(|_| HostError::Abi("module does not export `umber_register`".to_string()))?;
        register
            .call(&mut store, ())
            .map_err(|e| HostError::Abi(format!("umber_register trapped: {e}")))?;

        let invoke = instance
            .get_typed_func::<i32, ()>(&mut store, "umber_invoke")
            .map_err(|_| HostError::Abi("module does not export `umber_invoke`".to_string()))?;

        let registered = std::mem::take(&mut store.data_mut().registered);
        let mut index_of = HashMap::new();
        for (i, (id, _title)) in registered.iter().enumerate() {
            index_of.insert(id.clone(), i as i32);
        }

        Ok((
            WasmModule {
                store,
                invoke,
                index_of,
            },
            registered,
        ))
    }

    /// Run the command `id`. The epoch deadline is armed by the caller via
    /// `store.set_epoch_deadline`; a deadline trap surfaces as
    /// [`HostError::Timeout`]. Returns the text the module emitted.
    pub fn invoke(&mut self, id: &str, deadline_ticks: u64) -> Result<String, HostError> {
        let index = *self
            .index_of
            .get(id)
            .ok_or_else(|| HostError::NoSuchCommand(id.to_string()))?;

        self.store.data_mut().output.clear();
        self.store.set_epoch_deadline(deadline_ticks);

        match self.invoke.call(&mut self.store, index) {
            Ok(()) => Ok(std::mem::take(&mut self.store.data_mut().output)),
            Err(err) => {
                if err.downcast_ref::<Trap>() == Some(&Trap::Interrupt) {
                    Err(HostError::Timeout)
                } else {
                    Err(HostError::Invoke(format!("{err}")))
                }
            }
        }
    }
}

/// Install the three ABI v1 imports into `linker` under namespace `umber`.
fn define_imports(linker: &mut Linker<WasmState>) -> Result<(), HostError> {
    let map = |e: wasmtime::Error| HostError::Load(format!("link: {e}"));

    linker
        .func_wrap(
            "umber",
            "register",
            |mut caller: Caller<'_, WasmState>,
             id_ptr: i32,
             id_len: i32,
             title_ptr: i32,
             title_len: i32| {
                let mem = caller
                    .data()
                    .memory
                    .ok_or_else(|| wasmtime::Error::msg("no memory"))?;
                let id = read_utf8(&mem, &caller, id_ptr, id_len)?;
                let title = read_utf8(&mem, &caller, title_ptr, title_len)?;
                caller.data_mut().registered.push((id, title));
                Ok(())
            },
        )
        .map_err(map)?;

    linker
        .func_wrap(
            "umber",
            "emit",
            |mut caller: Caller<'_, WasmState>, ptr: i32, len: i32| {
                let mem = caller
                    .data()
                    .memory
                    .ok_or_else(|| wasmtime::Error::msg("no memory"))?;
                let text = read_utf8(&mem, &caller, ptr, len)?;
                caller.data_mut().output.push_str(&text);
                Ok(())
            },
        )
        .map_err(map)?;

    linker
        .func_wrap(
            "umber",
            "config_get",
            |mut caller: Caller<'_, WasmState>,
             key_ptr: i32,
             key_len: i32,
             out_ptr: i32,
             out_cap: i32|
             -> Result<i32, wasmtime::Error> {
                let mem = caller
                    .data()
                    .memory
                    .ok_or_else(|| wasmtime::Error::msg("no memory"))?;
                let key = read_utf8(&mem, &caller, key_ptr, key_len)?;
                let value = match caller.data().config.get(&key) {
                    Some(v) => v.clone(),
                    None => return Ok(-1),
                };
                let bytes = value.as_bytes();
                if bytes.len() > out_cap.max(0) as usize {
                    return Ok(-2);
                }
                mem.write(&mut caller, out_ptr as usize, bytes)
                    .map_err(|e| wasmtime::Error::msg(format!("config_get write: {e}")))?;
                Ok(bytes.len() as i32)
            },
        )
        .map_err(map)?;

    Ok(())
}

/// Read a UTF-8 string of `len` bytes at `ptr` from guest memory, bounds- and
/// UTF-8-checked (a bad pointer or non-UTF-8 payload is a guest error, not a
/// host panic).
fn read_utf8(
    mem: &Memory,
    caller: &Caller<'_, WasmState>,
    ptr: i32,
    len: i32,
) -> Result<String, wasmtime::Error> {
    if ptr < 0 || len < 0 {
        return Err(wasmtime::Error::msg("negative ptr/len"));
    }
    let (ptr, len) = (ptr as usize, len as usize);
    let data = mem.data(caller);
    let end = ptr
        .checked_add(len)
        .ok_or_else(|| wasmtime::Error::msg("ptr+len overflow"))?;
    let slice = data
        .get(ptr..end)
        .ok_or_else(|| wasmtime::Error::msg("string out of bounds"))?;
    String::from_utf8(slice.to_vec()).map_err(|_| wasmtime::Error::msg("string not UTF-8"))
}
