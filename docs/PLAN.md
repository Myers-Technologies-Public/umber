# Umber — Phase Plan

Spike-and-slice. Every phase has a GO/NO-GO or exit criteria. Decisions
referenced as D# from [DECISIONS.md](DECISIONS.md).

## Architecture sketch

```
umber (bin)            window, event loop, wires kernel + host + backend
crates/
  umber-ui             wgpu renderer + retained UI layer: glyph atlas,
                       cosmic-text shaping, damage tracking, panel layout
  umber-text           buffer model on ropey: edits, undo tree, marks
  umber-syntax         tree-sitter incremental parse + highlight
  umber-kernel         command registry, keymap engine (chords, D6),
                       command palette, TOML config (D13), module registry
  umber-host           wasmtime component host + permission broker;
                       mlua runtime for the Lua tier (D9)
  umber-proto          workspace protocol types (core <-> backend)
  umberd (bin)         headless workspace daemon: fs, search, PTYs, LSP
modules/               first-party, compiled to WASM, default-on (D10)
  editor-pane/  file-tree/  terminal/  git/  lsp/
  ssh-remote/   agent-dashboard/  theme-umber-dark/
```

**Rule 1 (from D7):** the editor core never touches the filesystem directly.
It speaks `umber-proto` to a workspace backend — in-process for local
projects, `umberd` over SSH for remote ones. Same code path for both, one
async boundary. (Xi-editor died of async-everything; Zed lives with exactly
this split. We copy Zed's shape, not xi's.)

**Rule 2 (from D10):** if a feature can be a module, it is a module. P2
proves it by re-shipping P1's built-ins as modules.

**TUI directive (D5):** the panel API exposes two render surfaces —
`plain` (line-oriented, stdout-style) and `tui` (widget tree: boxes, lists,
sparklines). Modules implement one or both; the user toggles per panel.
Both surfaces render through umber-ui — there is no embedded web/terminal
emulation for module UI.

## Module manifest (`umber.toml`) — first sketch

```toml
[module]
name = "agent-dashboard"
version = "0.1.0"
kind = "wasm"              # wasm | lua
entry = "agent_dashboard.wasm"
default_on = true

[permissions]              # deny-by-default, user-granted (cfx-style)
fs = ["read:workspace"]
net = ["localhost"]
exec = ["pi"]

[ui]
panels = ["agents"]
commands = ["agents.dashboard.open", "agents.session.new"]
surfaces = ["tui", "plain"]
```

## Phases

### P0 — Render spike (GO/NO-GO on D4)
winit window (Wayland) + wgpu surface + cosmic-text shaping + glyph atlas.
Load a large file into ropey, draw it, type into it. Instrument
keystroke→present latency (same discipline as Bidfall cx-engine).
- **Exit:** p99 keystroke→present ≤ 8 ms; smooth scroll on a 100 MB file;
  idle RAM ≤ 150 MB; cold start ≤ 300 ms.
- **NO-GO fallback:** gpui (revisit D4).

### P1 — Editor core
umber-text (undo tree, multi-cursor data model), umber-syntax highlighting,
open/save through umber-proto (local backend), umber-kernel: command
registry, chord keymap engine, Ctrl+Shift+P palette, TOML config with live
reload. Minimal chrome: near-zero title bar, small edge padding (D5 spirit).
- **Exit:** daily-drivable for single files; every action reachable from
  the palette; zero mouse required.

### P2 — Module host (the identity phase)
wasmtime component host, `umber.toml` manifest, permission broker,
load/unload/toggle at runtime without restart. All three author tiers (D9):
Rust→WASM, TS/AssemblyScript→WASM, Lua via mlua. Then **re-ship editor-pane
and file-tree as modules** and delete the built-in paths.
- **Exit:** kernel boots to an empty shell with zero modules; enabling
  modules brings the editor back; a third-party "hello panel" exists in all
  three tiers.

### P3 — Terminal + SSH remote (v0.1 release scope, D7)
`terminal` module on alacritty_terminal (PTYs owned by the workspace
backend, so remote terminals come free). `ssh-remote` module: russh
transport, `umberd` deploy/handshake/reconnect.
- **Exit:** open a remote project on moo, edit + terminal over SSH, survive
  a network drop.

### P4 — Agent dashboard (pi first, D11/D12)
`AgentBackend` trait: list sessions, live state (running / awaiting
instruction), runtime duration, token usage, streamed output, send prompt /
steer / follow-up. Backends:
- **pi-local:** spawn/attach `pi --mode rpc` (JSON-RPC over stdio); read
  `~/.pi/agent/sessions/*.jsonl` for history + token accounting.
- **pi-remote:** same, executed on the workspace backend (`umberd` on moo) —
  remote agents ride the existing SSH channel; plus a lightweight beacon so
  non-workspace hosts can register.
Research tasks before design: read pi `docs/rpc.md` and
`docs/session-format.md`; verify where token/usage counts actually live.
- **Exit:** dashboard panel (tui + plain surfaces) showing local + moo pi
  sessions: state, elapsed, tokens, live output; prompt/steer from Umber.

### P5 — Real-editor tier
`lsp` module (completion, diagnostics, go-to-def), `git` module (gutter,
stage/commit), fuzzy project search. All as modules over umber-proto.
- **Exit:** Umber can develop Umber.

### P6 — Marketplace
Registry server on nexeon (SvelteKit or Bun — web stack, user's call
closer to build), module signing (ed25519), in-editor browse/install/update,
premium licensing + entitlements.
- **Exit:** a paid module can be bought, installed, and updated end-to-end.

## Risks

| Risk | Mitigation |
|---|---|
| Text shaping/IME/bidi swamp | cosmic-text carries shaping; IME deferred behind a P1 flag, not silently dropped |
| Three plugin tiers day 1 (D9) balloon P2 | one shared host ABI; TS and Lua tiers are bindings over the same WIT interface, not separate hosts |
| Own UI engine (D4) is the long pole | P0 gate is exactly this; gpui fallback pre-agreed |
| SSH day 1 (D7) | proto split from P1 means SSH is a transport, not a rewrite |
| pi internals shift under us | integrate only via documented surfaces (rpc mode, session format doc) |

## Reading list (research phase)

- Xi-editor retrospective (Raph Levien) — the async-everything cautionary tale
- Zed source: `gpui` renderer internals, remote dev (`remote_server`), WASM extension host
- Lapce: WASI plugin system (closest existing marketplace model)
- Helix: architecture docs (what must be core when there are no plugins)
- wasmtime component model + WIT docs
- alacritty_terminal embedding API; russh examples
- pi: `docs/rpc.md`, `docs/session-format.md`, `docs/extensions.md`

## Open questions (park until relevant)

1. TS tier mechanics: AssemblyScript→WASM only, or embed QuickJS-in-WASM to
   run plain JS/TS? (P2)
2. Theme format + default theme design language. (P1/P2)
3. Marketplace branding/domain; payment rails. (P6)
4. Windows timeline. (post-v0.1, D8)

## Status ledger (updated 2026-07-16)

- P0 render spike: DONE (user-verified on Wayland; formal D4 numbers still unreported)
- P1 editor core: DONE (selection/clipboard/undo/save/palette/settings/modules)
- P2 module host: DONE slice 1 (wasm+lua sandboxed, runtime load/unload; editor-pane-as-module + ABI v2 I/O still open)
- P3 terminal: DONE (embedded panel + headless PTY e2e)
- P3b SSH: DONE slice 1 (ssh-in-terminal via host picker; umberd remote *workspace* still open)
- P4 agents: DONE slice 1 (read-only session dashboard from JSONL); slice 2 = live RPC control (pi --mode rpc)
- P5 LSP/git/search: NOT STARTED
- P6 marketplace: NOT STARTED
- QoL: F1 help, Ctrl+G goto-line, Ctrl+J panel toggle, hover/separator, Ghostty scrollbar
