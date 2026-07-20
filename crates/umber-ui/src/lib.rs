//! umber-ui — wgpu renderer + glyph atlas + cosmic-text shaping.
//!
//! P-phase: **P0** render spike (this crate), growing into the retained UI
//! layer (damage tracking, panel layout, plain/tui surfaces per D5) in P1+.
//!
//! The spike owns the GPU side: a wgpu surface plus a glyphon text renderer
//! (cosmic-text shaping -> etagere glyph atlas texture -> wgpu draw). The
//! winit window and event loop live in the `umber` bin (per the architecture
//! sketch in docs/PLAN.md); this crate is handed an `Arc<Window>`.

mod popout;
mod renderer;

pub use popout::{PopoutWindow, RenameOutcome};
pub use renderer::{
    OverlaySpec, PaneDividerSpec, Renderer, ScrollbarGeom, ScrollbarInfo, SelSpan,
    TerminalTextSpan, GIT_ADDED_COLOR, GIT_DELETED_COLOR, GIT_MODIFIED_COLOR,
};
