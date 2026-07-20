# Umber

A minimalist, keyboard-first, GPU-rendered code editor for Linux and Windows.
Rust core, everything-is-a-module architecture, WASM-sandboxed marketplace
modules, first-class SSH remote development, and built-in AI agent
orchestration (pi first).

[![CI](https://github.com/Myers-Technologies-Public/umber/actions/workflows/ci.yml/badge.svg)](https://github.com/Myers-Technologies-Public/umber/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)
[![Releases](https://img.shields.io/github/v/release/Myers-Technologies-Public/umber?include_prereleases&sort=semver)](https://github.com/Myers-Technologies-Public/umber/releases)

> **Status:** research phase (kicked off 2026-07-16). Umber is under active,
> pre-1.0 development — expect rough edges, moving APIs, and features landing
> incrementally. The pieces below describe the architecture and where it's
> headed; not everything is wired end-to-end yet.

<!-- PLACEHOLDER: replace with a real hero screenshot (docs/images/hero.png).
     Recommended: a wide shot of the editor with a couple of tiled panes and
     the command palette open. -->
<p align="center">
  <img src="docs/images/hero.png" alt="Umber editor — hero screenshot (placeholder)" width="820">
</p>

## Features

- **GPU-rendered, latency-first** — the whole surface is drawn through `wgpu`
  (Vulkan/DX12/Metal). Keystroke→pixel latency is treated as the product, not
  an afterthought.
- **Keyboard-first** — a fuzzy command palette (`Ctrl+Shift+P`) drives
  everything, with mouse-hover fallbacks. Minimal chrome, small padding, no
  bloat.
- **Everything is a module** — default-on, toggleable, uninstallable. The
  unremovable kernel is only: render loop, module host, config, keybind engine,
  and command palette. Nothing else.
- **Sandboxed modules** — third-party modules run WASM-sandboxed (`wasmtime`)
  with an explicit capability manifest (fs / net / exec); a `mlua` Lua backend
  is available for lighter extensions.
- **Tiling panes** — four-direction splits, right-click splits anywhere,
  drag-a-tab docking with edge-zone previews, and pop-out independent windows.
- **Embedded terminals** — a real PTY per tile (`alacritty_terminal`), running
  your default shell — `$SHELL` on unix, ConPTY + `%ComSpec%` on Windows.
- **First-class SSH remote development** — edit on a remote host over a thin
  `umberd` request/response protocol.
- **Built-in AI agent orchestration** — drive coding agents (pi first) from
  inside the editor: threads, prompts, steering, history.
- **Syntax + language tooling** — Tree-sitter highlighting, LSP integration,
  fuzzy file picker, and project search.

<!-- PLACEHOLDER: a couple of supporting shots. Swap these in when available. -->
<p align="center">
  <img src="docs/images/tiling.png" alt="Tiling panes (placeholder)" width="400">
  <img src="docs/images/agents.png" alt="AI agent panel (placeholder)" width="400">
</p>

## Principles

1. Near-zero performance hit. Keystroke→pixel latency is the product.
2. Everything is a module — default-on, toggleable, uninstallable. The
   unremovable kernel is: render loop, module host, config, keybind engine,
   command palette. Nothing else.
3. Minimalist UI: small padding, minimal title bar, no bloat, keyboard-first
   with mouse-hover fallbacks.
4. OSS core; marketplace modules may be paid (hosted on nexeon, later).

## Platforms

| Platform | Status | Notes |
| --- | --- | --- |
| Linux | Primary | Wayland-first (`wayland-data-control` clipboard), X11 supported. |
| Windows | Supported | ConPTY terminals, `%APPDATA%` config, DX12/Vulkan via `wgpu`. |
| macOS | Untested | Should build (nothing macOS-hostile), but not yet exercised. |

## Building from source

Requires a recent stable Rust toolchain.

```sh
git clone https://github.com/Myers-Technologies-Public/umber.git
cd umber
cargo build --release
./target/release/umber
```

**Windows:** build with the MSVC toolchain. The `tree-sitter` grammars and the
vendored Lua (`mlua`) compile C, so you need the **Visual Studio Build Tools**
(the "Desktop development with C++" workload) installed.

```powershell
rustup target add x86_64-pc-windows-msvc
cargo build --release --target x86_64-pc-windows-msvc
```

## Project layout

| Crate | Role |
| --- | --- |
| `umber` | The editor binary: windowing, input, panes, terminal, agents. |
| `umber-ui` | `wgpu`/`glyphon` renderer, tiling, pop-out surfaces. |
| `umber-text` | Rope-backed text buffer (`ropey`). |
| `umber-syntax` | Tree-sitter highlighting. |
| `umber-kernel` | Config, keybinds, command registry — the unremovable core. |
| `umber-host` | Module host: WASM (`wasmtime`) + Lua (`mlua`), capability manifest. |
| `umber-proto` | Wire protocol shared with the remote daemon. |
| `umberd` | Remote-development daemon (SSH-first). |

See [docs/PLAN.md](docs/PLAN.md) for the phase plan, architecture, and exit
criteria, and [docs/DECISIONS.md](docs/DECISIONS.md) for the locked decision
log.

## Releases

Prebuilt binaries for Linux and Windows are published on the
[Releases page](https://github.com/Myers-Technologies-Public/umber/releases). Releases
are cut by pushing a `v*` tag, which triggers the release workflow to build and
attach artifacts. See [CHANGELOG.md](CHANGELOG.md) for what's in each release.

## Contributing

Contributions are welcome. By submitting a contribution you agree to license it
under the project's dual MIT/Apache-2.0 terms (inbound = outbound). Please run
`cargo fmt`, `cargo clippy`, and `cargo test` before opening a pull request.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or
  <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or
  <http://opensource.org/licenses/MIT>)

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
