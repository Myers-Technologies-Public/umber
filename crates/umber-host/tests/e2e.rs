//! End-to-end tests for the module host: the two backends' full load ->
//! register -> invoke -> emit round trip, per-invocation interruption in both
//! tiers, manifest parsing, and load/unload command (de)registration.
//!
//! Fixtures live in `tests/fixtures/`. The `.wat` modules are compiled directly
//! by wasmtime's `wat` feature (no toolchain needed), so the wasm tier is
//! exercised without a build step. Manifests are constructed in-code (all
//! `Manifest` fields are public) pointing `entry` at a fixture file, so the
//! tests need no scratch directories or `/tmp` at all.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use umber_host::{HostError, Manifest, ModuleHost, ModuleKind, Permissions};

/// Absolute path to `tests/fixtures/`.
fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
}

/// A minimal manifest pointing `entry` at a fixture file. `config` starts empty;
/// callers needing `config_get` coverage insert keys before loading.
fn manifest(name: &str, kind: ModuleKind, entry: &str) -> Manifest {
    Manifest {
        name: name.to_string(),
        version: "0.1.0".to_string(),
        kind,
        entry: entry.to_string(),
        default_on: false,
        permissions: Permissions::default(),
        ui_panels: Vec::new(),
        ui_commands: Vec::new(),
        ui_surfaces: Vec::new(),
        config: BTreeMap::new(),
    }
}

/// Wall-clock ceiling for an interruption test: the invoke must trap and unwind
/// well inside this, or the budget/hook is not doing its job.
const INTERRUPT_CEILING: Duration = Duration::from_secs(5);

// ---------------------------------------------------------------------------
// (a) wasm round trip
// ---------------------------------------------------------------------------

#[test]
fn wasm_command_round_trip() {
    let mut host = ModuleHost::new().unwrap();
    let cmds = host
        .load(
            manifest("hello-wasm", ModuleKind::Wasm, "hello.wat"),
            &fixtures(),
        )
        .unwrap();
    assert_eq!(cmds.len(), 1);
    assert_eq!(cmds[0].id, "hello.greet");
    assert_eq!(cmds[0].title, "Hello: Greet");

    let out = host.invoke("hello.greet").unwrap();
    assert_eq!(out, "hello from wasm");
}

// ---------------------------------------------------------------------------
// (b) lua round trip
// ---------------------------------------------------------------------------

#[test]
fn lua_command_round_trip() {
    let mut host = ModuleHost::new().unwrap();
    let cmds = host
        .load(
            manifest("hello-lua", ModuleKind::Lua, "hello.lua"),
            &fixtures(),
        )
        .unwrap();
    assert_eq!(cmds.len(), 1);
    assert_eq!(cmds[0].id, "hello.lua");
    assert_eq!(cmds[0].title, "Hello: Lua");

    let out = host.invoke("hello.lua").unwrap();
    assert_eq!(out, "hello from lua");
}

/// The Lua tier reads its manifest-declared config through `umber.config_get`.
#[test]
fn lua_reads_manifest_config() {
    let mut m = manifest("cfg-lua", ModuleKind::Lua, "config.lua");
    m.config.insert("greeting".to_string(), "howdy".to_string());

    let mut host = ModuleHost::new().unwrap();
    host.load(m, &fixtures()).unwrap();
    assert_eq!(host.invoke("cfg.show").unwrap(), "howdy");
}

// ---------------------------------------------------------------------------
// (c) infinite-loop interruption in BOTH tiers
// ---------------------------------------------------------------------------

#[test]
fn wasm_infinite_loop_is_interrupted() {
    // Short budget so the epoch deadline trips quickly; the wall-clock assert
    // bounds the whole test.
    let mut host = ModuleHost::with_budget(Duration::from_millis(50)).unwrap();
    host.load(
        manifest("loop-wasm", ModuleKind::Wasm, "loop.wat"),
        &fixtures(),
    )
    .unwrap();

    let start = Instant::now();
    let res = host.invoke("loop.spin");
    let elapsed = start.elapsed();

    assert!(
        matches!(res, Err(HostError::Timeout)),
        "expected Timeout, got {res:?}"
    );
    assert!(
        elapsed < INTERRUPT_CEILING,
        "wasm interruption took too long: {elapsed:?}"
    );

    // The host survives a runaway guest: a healthy module still loads + runs.
    host.load(
        manifest("hello-wasm", ModuleKind::Wasm, "hello.wat"),
        &fixtures(),
    )
    .unwrap();
    assert_eq!(host.invoke("hello.greet").unwrap(), "hello from wasm");
}

#[test]
fn lua_infinite_loop_is_interrupted() {
    let mut host = ModuleHost::with_budget(Duration::from_millis(50)).unwrap();
    host.load(
        manifest("loop-lua", ModuleKind::Lua, "loop.lua"),
        &fixtures(),
    )
    .unwrap();

    let start = Instant::now();
    let res = host.invoke("loop.lua");
    let elapsed = start.elapsed();

    assert!(
        matches!(res, Err(HostError::Timeout)),
        "expected Timeout, got {res:?}"
    );
    assert!(
        elapsed < INTERRUPT_CEILING,
        "lua interruption took too long: {elapsed:?}"
    );

    // The host survives: a healthy module still loads + runs afterward.
    host.load(
        manifest("hello-lua", ModuleKind::Lua, "hello.lua"),
        &fixtures(),
    )
    .unwrap();
    assert_eq!(host.invoke("hello.lua").unwrap(), "hello from lua");
}

// ---------------------------------------------------------------------------
// (d) manifest parsing: valid, missing required, unknown kind, permission lists
// ---------------------------------------------------------------------------

#[test]
fn manifest_valid_parses_from_file() {
    let m = Manifest::from_path(&fixtures().join("valid.toml")).unwrap();
    assert_eq!(m.name, "sample");
    assert_eq!(m.version, "0.2.0");
    assert_eq!(m.kind, ModuleKind::Wasm);
    assert_eq!(m.entry, "hello.wat");
    assert!(m.default_on);
    // Permission lists round-trip as declared.
    assert_eq!(m.permissions.fs, vec!["read:workspace", "write:tmp"]);
    assert_eq!(m.permissions.net, vec!["localhost"]);
    assert_eq!(m.permissions.exec, vec!["pi"]);
    assert_eq!(m.permissions.summary(), "fs:2 net:1 exec:1");
    assert_eq!(m.ui_commands, vec!["hello.greet"]);
    assert_eq!(m.config.get("greeting").map(String::as_str), Some("howdy"));
}

#[test]
fn manifest_missing_required_field_errors() {
    let text = "[module]\nname = \"x\"\nkind = \"lua\"\nentry = \"x.lua\"\n";
    assert!(matches!(
        Manifest::parse(text),
        Err(umber_host::ManifestError::MissingField("module.version"))
    ));
}

#[test]
fn manifest_unknown_kind_errors() {
    let text = "[module]\nname=\"x\"\nversion=\"1\"\nkind=\"python\"\nentry=\"x.py\"\n";
    match Manifest::parse(text) {
        Err(umber_host::ManifestError::UnknownKind(k)) => assert_eq!(k, "python"),
        other => panic!("expected UnknownKind, got {other:?}"),
    }
}

#[test]
fn manifest_empty_permission_lists_default_empty() {
    let text = "[module]\nname=\"x\"\nversion=\"1\"\nkind=\"lua\"\nentry=\"x.lua\"\n";
    let m = Manifest::parse(text).unwrap();
    assert!(m.permissions.fs.is_empty());
    assert!(m.permissions.net.is_empty());
    assert!(m.permissions.exec.is_empty());
    assert_eq!(m.permissions.summary(), "no perms");
}

// ---------------------------------------------------------------------------
// (e) load -> unload deregisters commands
// ---------------------------------------------------------------------------

#[test]
fn load_then_unload_deregisters_commands() {
    let mut host = ModuleHost::new().unwrap();
    host.load(
        manifest("hello-wasm", ModuleKind::Wasm, "hello.wat"),
        &fixtures(),
    )
    .unwrap();
    assert!(host.is_loaded("hello-wasm"));
    assert_eq!(host.invoke("hello.greet").unwrap(), "hello from wasm");

    let removed = host.unload("hello-wasm");
    assert_eq!(removed, vec!["hello.greet".to_string()]);
    assert!(!host.is_loaded("hello-wasm"));

    // The command no longer routes once its module is gone.
    assert!(matches!(
        host.invoke("hello.greet"),
        Err(HostError::NoSuchCommand(_))
    ));

    // Unloading an unknown module is a harmless empty removal.
    assert!(host.unload("never-loaded").is_empty());
}

/// A second module with the same manifest name is refused (no silent clobber).
#[test]
fn duplicate_load_is_rejected() {
    let mut host = ModuleHost::new().unwrap();
    host.load(manifest("dup", ModuleKind::Wasm, "hello.wat"), &fixtures())
        .unwrap();
    let again = host.load(manifest("dup", ModuleKind::Wasm, "hello.wat"), &fixtures());
    assert!(matches!(again, Err(HostError::AlreadyLoaded(_))));
}

#[test]
fn lua_sandbox_exposes_no_ambient_authority() {
    // Regression guard for the deny-all sandbox (D9): `os`, `io`, `package`,
    // `require`, and `debug` must be absent inside module scripts.
    let mut host = ModuleHost::new().unwrap();
    host.load(
        manifest("sandbox-lua", ModuleKind::Lua, "sandbox.lua"),
        &fixtures(),
    )
    .unwrap();
    assert_eq!(host.invoke("sandbox.check").unwrap(), "clean");
}
