//! Headless e2e for the remote workspace (P3b): drives the REAL `umberd`
//! binary as a local subprocess over the umber-proto protocol — the same code
//! path SSH uses, minus the ssh hop. Proves open/save/list/stat round-trip and
//! that path confinement rejects escapes.

use std::path::PathBuf;

use umber::remote::RemoteWorkspace;

/// Path to the umberd binary built by cargo for this test run.
fn umberd_bin() -> PathBuf {
    // CARGO_BIN_EXE_<name> is set by cargo for integration tests of a workspace
    // that builds the bin; fall back to the conventional target path.
    if let Some(p) = std::option_env!("CARGO_BIN_EXE_umberd") {
        return PathBuf::from(p);
    }
    let mut p = std::env::current_exe().unwrap();
    p.pop(); // test exe
    if p.ends_with("deps") {
        p.pop();
    }
    p.push("umberd");
    p
}

fn workspace_root(name: &str) -> PathBuf {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/test-tmp")
        .join(name);
    std::fs::create_dir_all(&root).unwrap();
    root
}

#[test]
fn write_read_list_roundtrip() {
    let root = workspace_root("remote_rt");
    let _ = std::fs::remove_file(root.join("hello.txt"));
    let mut ws =
        RemoteWorkspace::connect_local(&umberd_bin(), &root).expect("connect to local umberd");

    let n = ws
        .write_file("hello.txt", "remote edit\nsecond line\n")
        .unwrap();
    assert_eq!(n, "remote edit\nsecond line\n".len());

    let contents = ws.read_file("hello.txt").unwrap();
    assert_eq!(contents, "remote edit\nsecond line\n");

    let entries = ws.list_dir(".").unwrap();
    assert!(entries.iter().any(|e| e.name == "hello.txt" && !e.is_dir));

    let st = ws.stat("hello.txt").unwrap();
    assert!(st.exists && !st.is_dir);
    let missing = ws.stat("nope.txt").unwrap();
    assert!(!missing.exists);
}

#[test]
fn nested_write_creates_parent_dirs() {
    let root = workspace_root("remote_nested");
    let mut ws = RemoteWorkspace::connect_local(&umberd_bin(), &root).expect("connect");
    ws.write_file("sub/dir/deep.txt", "deep").unwrap();
    assert_eq!(ws.read_file("sub/dir/deep.txt").unwrap(), "deep");
}

#[test]
fn path_escape_is_rejected() {
    let root = workspace_root("remote_confine");
    let mut ws = RemoteWorkspace::connect_local(&umberd_bin(), &root).expect("connect");
    // Climbing out of the root must fail, not read the host's /etc/passwd.
    let err = ws.read_file("../../../../../../etc/passwd").unwrap_err();
    assert!(
        err.to_lowercase().contains("escape"),
        "expected an escape rejection, got: {err}"
    );
}
