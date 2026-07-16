//! umberd — headless workspace daemon: fs, search, PTYs, LSP.
//!
//! P-phase: **P3** (the remote workspace backend; local backend is in-process
//! from P1 over the same umber-proto boundary — Rule 1). Stub `main` for the
//! P0 render spike.

fn main() {
    // TODO(P3): serve the umber-proto workspace protocol over a transport
    // (in-process for local, SSH for remote — same code path, Rule 1 / D7).
    eprintln!("umberd: not implemented yet (P3). See docs/PLAN.md.");
}
