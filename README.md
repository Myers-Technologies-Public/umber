# Umber

A minimalist, keyboard-first, GPU-rendered code editor for Linux. Rust core,
everything-is-a-module architecture, WASM-sandboxed marketplace modules,
first-class SSH remote development, and built-in AI agent orchestration
(pi first).

**Status:** research phase (kicked off 2026-07-16).

- [docs/PLAN.md](docs/PLAN.md) — phase plan, architecture, exit criteria
- [docs/DECISIONS.md](docs/DECISIONS.md) — locked decision log

## Principles

1. Near-zero performance hit. Keystroke→pixel latency is the product.
2. Everything is a module — default-on, toggleable, uninstallable. The
   unremovable kernel is: render loop, module host, config, keybind engine,
   command palette. Nothing else.
3. Minimalist UI: small padding, minimal title bar, no bloat, keyboard-first
   with mouse-hover fallbacks.
4. OSS core; marketplace modules may be paid (hosted on nexeon, later).
