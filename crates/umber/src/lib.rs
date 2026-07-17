//! umber — library surface of the main binary.
//!
//! The bin target (`main.rs`) owns the window/event loop; this lib target
//! exists so integration tests can exercise window-free machinery directly.
//! Currently that is the embedded terminal session (P3): PTY + parser +
//! grid, fully headless.

pub mod agent_rpc;
pub mod agents;
pub mod terminal;
