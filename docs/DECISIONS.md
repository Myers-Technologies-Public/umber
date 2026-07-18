# Umber — Decision Log

Locked 2026-07-16 (research-phase kickoff). Change requires an entry here, not
a silent edit.

| # | Decision | Choice | Notes |
|---|----------|--------|-------|
| D1 | Name | **Umber** | crates `umber-*`, config `~/.config/umber/`, module manifest `umber.toml` |
| D2 | Repo | local `~/MyersTechnologies/umber/` | no remote for now |
| D3 | License | OSS core (MIT OR Apache-2.0), marketplace modules may be paid/closed | Zed/cfx model; drives adoption toward paid tier |
| D4 | UI stack | **own engine: raw wgpu + custom UI layer** | cx-engine (Bidfall) experience transfers; text shaping via cosmic-text |
| D5 | TUI directive | per-panel render-mode toggle: plain command-line I/O vs rich TUI widgets | one GUI app; NOT a separate terminal build of the editor |
| D6 | Keybindings | VS Code-style chords; `Ctrl+Shift+P` = command palette | modal editing can come later as a third-party-able module |
| D7 | Remote SSH | **day-1 module** (in v0.1 release scope) | forces the core/workspace-backend split from the start |
| D8 | Platform | Linux first (Wayland primary, X11 free via winit); Windows later | dev box: CachyOS/Wayland |
| D9 | Module tiers | **all three day 1**: Rust→WASM, TypeScript/AssemblyScript→WASM, Lua (mlua) | biggest scope add; see P2 |
| D10 | Kernel boundary | unremovable: render loop, module host, config, keybind engine, command palette | everything else — editor pane, file tree, terminal, git, LSP, SSH, agents — is a default-on removable module |
| D11 | Agent dashboard scope | local **and** remote agents day 1 | needs an agent-registry protocol, not just local process introspection |
| D12 | Agent backends | pi first; `AgentBackend` trait so others slot in later | nothing proprietary planned now; user adds as he sees fit |
| D13 | Config format | TOML | helix-style |
| D14 | Cadence | research now → build shortly after → ship soon | P0 spike is the first build artifact |

## Ground truth captured at kickoff

- Dev box: CachyOS (Arch), Wayland, `rustc 1.97.0` / `cargo 1.97.0` installed.
- pi integration surface (verified against pi `docs/sdk.md`, 2026-07-16):
  - SDK is **TypeScript, in-process only** — not directly usable from Rust.
  - Sanctioned language-agnostic path: **`pi --mode rpc` (JSON-RPC over
    stdio)** — see pi `docs/rpc.md` (still to be read; P4 research task).
  - Sessions persist as JSONL trees under `~/.pi/agent/sessions/`
    (`SessionManager`); token/usage field layout must be verified against pi
    `docs/session-format.md` before dashboard design — **do not assume**.
  - SDK events relevant to a dashboard: `agent_start`/`agent_end`,
    `turn_start`/`turn_end`, `tool_execution_*`, `message_update`,
    `queue_update` (steering/follow-up = "awaiting instruction" signals).

## 2026-07-18 — Ghostty-style tiling + context menus
- **Pane tree lives in the app (`umber::panes`), not the renderer.** Binary
  split tree with normalized rects; renderer only receives laid-out pane
  rects/dividers (`set_panes`) and owns per-tile buffers. Editor geometry
  accessors (`editor_card_rect` → `left_edge`/`doc_top`/`doc_size`) switch on
  the pane rect, so all caret/selection/scrollbar math followed for free.
- **Tiles spawn at their real grid.** Split the tree → sync the renderer →
  read `term_pane_grid` → spawn the PTY at that size. Spawn-then-resize
  garbled the first paint. Spawn failure rolls the split back.
- **Context menus composite post-text.** Card quads ride a vertex range past
  `ctx_quad_start`, labels ride the overlay text renderer — pre-text drawing
  let terminal glyphs paint over the menu.
- **One row pitch.** Context-menu labels, hover pill, hit-test, and card
  height all use `line_px()`; a 1.35× hit pitch made a "Close Pane" click
  execute "Split Down".
- Split chords (Ctrl+Shift+O/E) stay live under terminal focus, or a tiled
  shell could never be split again.
