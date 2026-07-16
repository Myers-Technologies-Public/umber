# Umber — Prior-Art Research

Research-phase distillation for the Umber editor. Each system below has a short
**how it works** grounded in code we actually read, then **STEAL** (adopt) and
**AVOID** lists tied back to Umber's [PLAN.md](PLAN.md) phases (P0–P6) and
[DECISIONS.md](DECISIONS.md) decisions (D1–D14).

**Method.** Shallow-cloned into `/tmp` on 2026-07-16 and read source directly:
- `zed` — `git clone --depth 1 https://github.com/zed-industries/zed` ✅
- `lapce` — `git clone --depth 1 https://github.com/lapce/lapce` ✅
- `helix` — `git clone --depth 1 https://github.com/helix-editor/helix` ✅

All three clones succeeded; every claim below cites a file path in one of these
trees. The xi-editor section is the one exception — see its note. Repo paths are
relative to each clone root (e.g. `zed:crates/remote/...`).

---

## 1. Zed — remote development architecture

### How it works

Zed's remote story is a **thin UI client + a headless daemon that owns all
project state**, exactly the shape PLAN.md Rule 1 (D7) prescribes.

**Two processes from one binary.** `remote_server` dispatches on
`run | proxy | version` (`zed:crates/remote_server/src/main.rs:59` prints
`usage: remote <run|proxy|version>`; `Commands` enum + `run()` in
`remote_server/src/lib.rs`). The long-lived **`run`** process is the daemon: it
binds *three* Unix domain sockets (stdin/stdout/stderr) via
`ServerListeners::new` → `UnixListener::bind`
(`zed:crates/remote_server/src/server.rs:388-399`) and hosts a `HeadlessProject`.
Each incoming SSH connection instead spawns a short-lived **`proxy`** process
that connects to those sockets and shuttles bytes between SSH stdio and the
persistent `run` process. This split is *the* reason a dropped network survives:
project state lives in `run`, which outlives any `proxy`/SSH churn.

**All state is server-side.** `HeadlessProject`
(`zed:crates/remote_server/src/headless_project.rs:52-74`) owns `fs`,
`worktree_store`, `buffer_store`, `lsp_store`, `dap_store`, `git_store`,
`task_store`, `breakpoint_store`, `agent_server_store`, `context_server_store`,
and `extensions` (`HeadlessExtensionStore`). The UI holds none of it; it
registers RPC handlers against a `session` —
`session.add_request_handler(cx.weak_entity(), Self::handle_list_remote_directory)`,
`handle_get_path_metadata`, `handle_ping`, `handle_get_processes`, …
(`headless_project.rs:292-296`).

**One wire format, framed simply.** The transport is length-prefixed protobuf:
`write_message` emits a 4-byte little-endian length then an encoded `Envelope`;
`read_message` reads the 4 bytes, then that many bytes
(`zed:crates/remote/src/protocol.rs:9-51`). The `Envelope`/`TypedEnvelope` types
(`rpc::proto`) are the *same* RPC types Zed uses for its collab server — the
remote path reuses the multiplayer protocol, it isn't a bespoke second one.

**Deploy + liveness.** The client downloads the correct server binary for the
remote's platform/arch (`download_server_binary_locally`,
`zed:crates/remote/src/remote_client.rs:149`), uploads it, and starts the proxy
(`start_proxy`, `remote_client.rs:447`/`690`). Liveness is a heartbeat:
`HEARTBEAT_INTERVAL = 5s`, `MAX_MISSED_HEARTBEATS = 5`,
`MAX_RECONNECT_ATTEMPTS = 3`, driven by an explicit connection `State` machine
(`Connecting → Connected → HeartbeatMissed → Reconnecting → ReconnectFailed →
ReconnectExhausted`, `remote_client.rs:159-193`). `ProxyLaunchError::ServerNotRunning`
uses exit code 90 to signal "reconnect attempted, daemon gone"
(`zed:crates/remote/src/proxy.rs:5-23`).

### STEAL

- **The proxy/daemon split is the model for `umberd` (P3, D7).** A persistent
  `umberd run` that owns worktree/buffer/search/PTY/LSP state, plus a thin
  `umberd proxy` spawned per SSH session, gives reconnect-survival almost for
  free — the daemon never learns the client blinked. Copy this two-subcommand
  shape verbatim (`server.rs` `ServerListeners` + `main.rs` dispatch).
- **Local and remote MUST be the same code path (PLAN Rule 1, D7).** Zed's UI
  talks to `HeadlessProject` through `session.add_request_handler` regardless of
  transport; the in-process case is just a different `session`. Make
  `umber-proto` the only way the core reaches project state, in-process backend
  and `umberd`-over-SSH behind the identical trait — this is already Rule 1;
  Zed confirms it scales.
- **Length-prefixed binary Envelope over stdio (P3).** `protocol.rs`'s 4-byte-LE
  length + encoded message is trivial, transport-agnostic (works over an SSH
  channel, a Unix socket, or an in-process pipe), and dodges JSON entirely. Use
  this framing for `umber-proto` rather than newline-delimited JSON-RPC.
- **Explicit connection state machine + heartbeat (P3 exit: "survive a network
  drop").** Zed's `State` enum with bounded reconnect attempts and a 5s/5-miss
  heartbeat is exactly the "survive a network drop" acceptance test. Lift the
  constants and the state names.
- **Ship the daemon binary from the client, keyed by remote arch (P3).**
  `download_server_binary_locally(platform, …)` + upload + version check is the
  deploy/handshake step PLAN P3 calls out; don't expect the daemon to be
  pre-installed.

### AVOID

- **Don't fork the wire format per feature.** Zed pays for its single `Envelope`
  union with a huge generated proto surface, but the discipline is right — one
  message type, one framing. Resist a JSON side-channel "just for terminals" or
  "just for agents"; route PTY (P3) and agent (P4) traffic through the same
  `umber-proto` envelope.
- **Don't let any project state leak to the UI side.** The moment the core caches
  buffer contents or worktree trees locally (instead of treating the backend as
  the source of truth), the local/remote paths diverge and reconnect gets hard.
  Zed keeps `BufferStore`/`WorktreeStore` server-side even for local projects;
  match that — the core is a view, not an owner.
- **Don't over-scope the daemon at P3.** `HeadlessProject` also carries DAP,
  context servers, agent servers, prettier, toolchains. For Umber P3 the daemon
  needs only fs + search + PTY + LSP (PLAN's `umberd` line); LSP itself is a P5
  module. Add stores as phases land, not up front.

---

## 2. Zed — WASM extension host

### How it works

**Component-model WASM, one WIT world.** Extensions are WASI-P0 *components*, not
core modules. The API is a single WIT world `extension`
(`zed:crates/extension_api/wit/<version>/extension.wit`) that `import`s host
capabilities (`http-client`, `process`, `platform`, `nodejs`, `github`, `dap`,
`context-server`) and `export`s the hooks the host calls: `init-extension`,
`language-server-command`, `labels-for-completions`, `run-slash-command`,
`context-server-command`, `get-dap-binary`, etc. Host-provided resources are
handed in as capability handles — e.g. `resource worktree { read-text-file:
func(path) -> result<string,string>; which: func(binary-name) -> option<string>;
shell-env: func() -> env-vars; }`. An extension can only touch a worktree it was
*given*.

**Build pipeline.** `ExtensionBuilder` compiles Rust extensions with
`--target wasm32-wasip2` (`zed:crates/extension/src/extension_builder.rs:28-29`),
auto-installs the target via `rustup target add`
(`extension_builder.rs:448-476`), fetches a pinned `wasi-sdk-25.0` for the C
grammar toolchain, and stamps/reads an API-version custom section from the
produced `.wasm` (`parse_wasm_extension_version`,
`zed:crates/extension_host/src/wasm_host.rs:807`). Grammars are compiled
separately with clang.

**Runtime + sandbox.** `WasmHost` builds a `wasmtime::Engine` with **epoch
interruption** so a runaway extension can be pre-empted
(`wasm_host.rs:546-560`), builds a per-extension `WasiCtx` scoped to a
work-dir (`build_wasi_ctx`, `wasm_host.rs:729-742`;
`test_writeable_path_rejects_escape_attempts` guards path escapes), and caches
compiled artifacts (`IncrementalCompilationCache`). Each instance carries a
`WasmState { table, ctx: WasiCtx, capability_granter }`.

**Deny-by-default capabilities.** The manifest declares capabilities
(`extension.toml`: `[[capabilities]] kind = "process:exec" command = "echo"
args = ["hello…"]`, plus `schema_version = 1`). At the call site the host asks
`CapabilityGranter`: `grant_exec`, `grant_download_file`,
`grant_npm_install_package` each scan granted capabilities and `bail!` if none
match (`zed:crates/extension_host/src/capability_granter.rs:23-81`). The matcher
is precise and wildcard-aware — `ProcessExecCapability::allows` supports exact
args, `*` (one arg), and `**` (trailing rest)
(`zed:crates/extension/src/capabilities/process_exec.rs`), and
`DownloadFileCapability::allows` matches host + path segments the same way
(`.../capabilities/download_file.rs`).

### STEAL

- **Single WIT world = the "one shared host ABI" P2 already committed to (D9).**
  Zed proves TS/Lua/Rust don't each need a bespoke host: define Umber's host as
  one WIT interface, and make the TS/AssemblyScript and Lua tiers *bindings over
  that same interface* (PLAN's stated mitigation for the "three tiers balloon
  P2" risk). Model the WIT after Zed's `import` host caps / `export` hooks split.
- **Capability handles, not ambient authority (P2 permission broker, D10).** Pass
  the module a `worktree` resource it can only read *because you gave it that
  handle* — mirrors Umber's `fs = ["read:workspace"]` manifest intent. A module
  with no handle can't reach the fs at all. This is a cleaner enforcement point
  than string-matching paths after the fact.
- **Deny-by-default + wildcard capability matcher (P2, D10; manifest sketch in
  PLAN).** `ProcessExecCapability`/`DownloadFileCapability` `allows()` with
  `*`/`**` is a ready-made design for Umber's `exec = ["pi"]` /
  `net = ["localhost"]` grants — copy the exact/`*`/`**` semantics and the
  "scan granted, else `bail!`" broker shape. It's ~40 lines and well-tested.
- **Epoch interruption for pre-emption (P2).** A default-on module that hangs
  must not freeze the render loop (P0's ≤8ms budget, D4). Wasmtime epoch
  interruption (`wasm_host.rs:546`) is the lever; wire it into `umber-host` from
  the start.
- **Stamp an ABI version into the artifact (P2/P6).** Reading an API-version
  custom section from the `.wasm` lets the host reject incompatible modules
  cleanly — essential once the P6 marketplace ships modules built against older
  host ABIs.
- **`--target wasm32-wasip2` + component model is the confirmed target triple
  (P2, D9).** Zed already fought this battle; use wasip2/component model, not the
  older reactor/preview1 style Lapce is stuck on (see §3 AVOID).

### AVOID

- **Don't grow the WIT world unboundedly.** Zed's `extension.wit` has sprawled to
  LSP + DAP + slash-commands + context-servers + docs-indexing in one world.
  That's an LSP-flavored extension API, not a general module API. Umber's kernel
  boundary (D10) makes the *editor pane itself* a module, so the WIT must expose
  UI/panel/command primitives (D5 surfaces, palette) — resist letting it become
  a language-tooling-only interface like Zed's.
- **Don't require a C toolchain (wasi-sdk-25) in the common path.** Zed pulls a
  ~hundreds-of-MB SDK to build tree-sitter grammars. Umber ships grammars with
  `umber-syntax` (P1) rather than per-extension, so keep the module build path
  pure-`cargo build --target wasm32-wasip2` and don't inherit the clang/wasi-sdk
  dependency for the module tiers.
- **Don't tie capability grants to install-time only.** Zed grants from the
  manifest at load. Umber's manifest is "deny-by-default, user-granted (cfx-style)"
  (PLAN) — keep a *runtime* consent step (broker prompts the user) rather than
  trusting the manifest's declared `[permissions]` blindly; the manifest declares
  intent, the user grants authority.

---

## 3. Lapce — WASI plugin system + marketplace

### How it works

**Plugins run in the proxy, not the UI.** Lapce is already core/proxy-split
(`lapce-app` UI ⇄ `lapce-proxy` backend over `lapce-rpc`), and *plugins load in
the proxy* (`lapce-proxy/src/plugin/`), so remote plugins come free — same lesson
as Zed §1, arrived at independently. Plugin lifecycle lives in
`catalog.rs` (`PluginCatalog`: `start_unactivated_volts`,
`check_unactivated_volts`, `shutdown_volt`, `handle_server_request`).

**"Volt" = the plugin unit.** A plugin is a **volt** declared by a `volt.toml`
manifest; metadata is `VoltMetadata`/`VoltInfo`
(`lapce-rpc/src/plugin.rs:60`, `wasm: Option<String>` field). Volts may be pure
WASI or pure config/theme (`wasm: bool`, `plugin.rs:35`); the test fixtures show
both — `some_author.test-plugin-one/` ships `lapce.wasm` + `volt.toml` +
`Dark.toml`/`Light.toml` themes, while `test-plugin-three/` is `volt.toml` only
(`lapce-proxy/src/plugin/wasi/plugins/`).

**WASI preview1, classic host functions.** `start_volt`
(`lapce-proxy/src/plugin/wasi.rs`) builds a `wasmtime::Engine::default()`,
`Module::from_file`, and a `WasiCtxBuilder` with **manual stdio pipes**
(`WasiPipe`) for stdin/stdout/stderr, then `preopened_dir(volt_path, "/")`. The
host↔plugin protocol is JSON-over-those-pipes: the plugin writes to stdout, the
host's `linker.func_wrap("lapce", "host_handle_rpc", …)` reads the string,
`handle_plugin_server_message` processes it, and the response is written back to
the plugin's stdin. It's an LSP-shaped JSON-RPC bridge tunneled through WASI
stdio.

**Sandbox is coarse.** The plugin gets:
- `.inherit_env()` — **the host's entire environment is handed to the plugin**
  (`wasi.rs`, `WasiCtxBuilder::new().inherit_env()`), plus `VOLT_OS/ARCH/LIBC/URI`.
- `preopened_dir(volt_path, "/")` — filesystem scoped to the volt's own dir.
- HTTP via `wasi-experimental-http` with `allowed_hosts: Some(vec!["insecure:allow-all"])`
  and `max_concurrent_requests: 100` — i.e. **network is allow-all**, no
  per-host gate (`wasi.rs`).

**Marketplace.** A real registry exists at `plugins.lapce.dev`. Discovery/install
hit a REST API: `GET …/api/v1/plugins?q={query}&offset={offset}`,
`…/plugins/{author}/{name}/latest`, `…/{version}/download`, `…/icon`, `…/readme`
(`lapce-app/src/plugin.rs:289,416,457,472`; download URL
`lapce-proxy/src/plugin/mod.rs:1557`). Install downloads a gzip/tar, unpacks it,
and `install_volt` registers it. No signing or entitlement layer is visible.

### STEAL

- **Load modules in the workspace backend so remote ⇒ free (P3/P5, D7).** Lapce
  independently confirms Zed's lesson: because volts run in `lapce-proxy`, they
  work over remote transparently. Umber's LSP/git/search modules (P5) should run
  against `umber-proto`/`umberd`, not in the UI process — then remote LSP is not
  a second implementation.
- **A `volt.toml` where WASM is optional (P2/P1).** Lapce's manifest cleanly
  covers *config-only* plugins (themes: `test-plugin-three` is `volt.toml` only,
  `wasm: false`). Umber's `theme-umber-dark` module (PLAN module list) and TOML
  themes (open question #2) fit this — let `umber.toml` `kind` cover a
  no-code/asset module, not only `wasm`/`lua`.
- **A REST registry with query/latest/download/icon/readme is the P6 minimum.**
  `plugins.lapce.dev`'s endpoint shape (`?q=&offset=`, `/latest`, `/download`,
  `/readme`, `/icon`) is a proven, small surface for the P6 marketplace on
  nexeon (SvelteKit/Bun per PLAN P6). Copy the endpoint layout as the baseline,
  then add what Lapce lacks (below).

### AVOID

- **Don't inherit the host environment into modules.** `.inherit_env()` leaks
  every host env var (tokens, keys) into every plugin — a straight-up sandbox
  hole. Umber's broker is deny-by-default (D10); pass only explicitly-granted
  env, never `inherit_env`. This is the single clearest "AVOID" in the whole
  survey.
- **Don't ship network as `allow-all`.** `allowed_hosts: ["insecure:allow-all"]`
  defeats the point of a permission model. Umber's manifest already scopes
  `net = ["localhost"]`; enforce it host-side with a Zed-style
  `NetCapability::allows(host)` matcher (§2), not a blanket allow.
- **Don't build on WASI preview1 + hand-rolled JSON-over-stdio pipes.** Lapce's
  `WasiPipe` stdio-tunneled JSON-RPC (`wasi.rs`) predates the component model;
  it's brittle (manual framing, string parsing) and can't express typed
  capability handles. Umber should use wasip2/component + WIT (§2) so the ABI is
  typed and capabilities are resources, not stdout strings. (Lapce is pinned to
  `wasmtime 14`; the component model is the modern path.)
- **Don't launch a marketplace without signing/entitlements (P6, D3).** Lapce's
  registry has no visible signing or paid-tier support. Umber P6 explicitly needs
  ed25519 module signing + premium licensing/entitlements — design those into the
  registry schema from day one, since retrofitting signatures onto an unsigned
  archive format is painful.

---

## 4. Helix — what stays in core when there is no plugin system

### How it works

Helix has (at this clone) **no runtime plugin system** — everything ships in
core, split across purpose crates (`helix/docs/architecture.md`):
`helix-core` (functional editing primitives), `helix-view` (frontend-agnostic-*ish*
editor logic), `helix-term` (the terminal frontend), `helix-tui` (TUI widgets,
forked from tui-rs), `helix-lsp`/`helix-dap`, `helix-event`, `helix-loader`.

Key core primitives (`architecture.md`, `docs/vision.md`):
- **Rope buffers** via re-exported `ropey`; cheap clone/snapshot — same choice as
  Umber's `umber-text` (PLAN).
- **Multiple selections are the core primitive**: a `Selection` is a set of
  `Range`s, each a moving `head` + immovable `anchor`; a single cursor is a
  1-range selection. Umber's `umber-text` "multi-cursor data model" (P1) is the
  same idea.
- **OT-like `Transaction`s**: every edit is a `Transaction` that applies to a
  rope and can be *inverted* to produce undo; selections/marks *map over* a
  transaction to translate positions. This is Umber's "undo tree, marks" (P1).
- **Compositor + `Component` layers**: the UI is a `Vec<Component>` stack the
  `Compositor` renders bottom-to-top, giving pickers/popups over the editor
  (`architecture.md` "View"/"TUI"). Components draw into a `Surface` (a buffer of
  `Rect`s).
- **`helix-event`**: typed events (`events!` macro), synchronous `register_hook!`,
  and `AsyncHook` for *debounced, cancellable* post-event work — their concession
  to async, deliberately narrow.
- **Vision explicitly rejects Electron/DOM** and targets low RAM ("shouldn't
  consume half your RAM on a Raspberry Pi"), edit-anything (200MB XML, minified
  JS on one line), batteries-included.

Honest caveat from their own docs: `helix-view` "was *supposed* to be a
frontend-agnostic imperative library … Currently it's tied to the terminal UI"
(`architecture.md`, "View"). The frontend-agnostic seam they wanted, they didn't
fully get.

### STEAL

- **Rope + Selection(head/anchor) + invertible Transaction is the P1 core.**
  Helix, Zed, and Lapce all land on ropey; Helix's `Transaction`-inverts-to-undo
  and selection-maps-over-transaction is the cleanest articulation of Umber's
  `umber-text` P1 scope (undo tree, multi-cursor, marks). Build `umber-text` on
  exactly these primitives.
- **A Compositor/Component layer stack for panels (D5, P0/P1 chrome).** Umber's
  D5 two-surface panel API (`plain`/`tui`) and "file picker over the editor" map
  directly onto Helix's `Vec<Component>` + `Surface`/`Rect` model. Steal the
  layer-stack rendering discipline for `umber-ui`'s retained panel layout — it's
  proven for pickers/popups/palette (D6).
- **A narrow, typed, debounced event/hook system (kernel).** `helix-event`'s
  `events!` + `register_hook!` + `AsyncHook` (debounced, cancellable) is a
  disciplined way to let modules react to editor events without an async
  free-for-all (see §5). Model `umber-kernel`'s module event dispatch on this —
  *synchronous hooks by default, explicit debounced async only where needed.*
- **"Batteries-included core, extend within reason" validates D10's boundary.**
  Helix's vision — basics built-in, plugins for the rest — is Umber's D10 kernel
  boundary stated as product philosophy. Their "must be core" set (rope,
  selections, transactions, LSP client, fuzzy picker) is a good sanity-check on
  what Umber keeps unremovable vs. ships as default-on modules.

### AVOID

- **Don't claim a frontend-agnostic seam you don't test.** Helix's own docs admit
  `helix-view` got "tied to the terminal UI" despite the intent. Umber's D5 says
  *one GUI app* with per-panel plain/tui toggle — so `umber-ui` must render both
  surfaces (PLAN: "Both surfaces render through umber-ui"). Don't let `tui` become
  a second, half-maintained frontend; the P0/P1 discipline is that both surfaces
  are the *same* renderer, enforced by actually shipping a module that uses each.
- **Don't defer extensibility to "someday" — it distorts the core.** Helix's
  no-plugin stance means language/theme/tooling changes all touch core crates.
  Umber's identity is the opposite (D10, P2: re-ship editor-pane and file-tree as
  modules and *delete the built-in paths*). Don't drift toward Helix's
  everything-in-core convenience during P1; keep the P2 module seam visible from
  the first built-in.
- **Don't copy modal-first as a default (D6).** Helix's core interaction is
  modal/selection-first by vision. Umber D6 is VS Code chords + `Ctrl+Shift+P`,
  with modal editing as a *later third-party module*. Take Helix's primitives,
  not its interaction model.

---

## 5. xi-editor — the async-everything / JSON-RPC-split postmortem

> **Grounding note.** Network fetch of Raph Levien's retrospective was not
> attempted from inside this run; the clones (§1–4) are code-grounded, but this
> section is reconstructed from prior knowledge of the xi-editor "Retrospective"
> essay and the project's `docs`/rope-science writings. **Treat the specific
> claims here as unverified recollection, not cited source** — the *conclusions*
> are corroborated by how Zed (§1) and Lapce (§3) actually structure their splits.

### How it worked (and why it stalled)

xi-editor (Raph Levien, ~2016–2020) was a Rust "core" exposing editor state to
*any* frontend over **asynchronous JSON-RPC**. Design tenets:

- **Async everything.** Every operation (even within-core) was modeled as async
  message passing, aiming for never-block-the-UI and a CRDT-ready collaborative
  core.
- **Frontend/core split over JSON-RPC.** The core owned the rope and edit logic;
  frontends (Cocoa, Electron/xi-mac, others) were separate processes talking
  JSON. A `xi-rope` with clever data structures underneath.
- **CRDT-based concurrent edit model** in the core for consistency under async.

Reported lessons from the retrospective (recollected):

1. **Async-everywhere made simple things hard.** Modeling *all* editor state
   transitions as async message passing exploded complexity — features that are
   trivially synchronous (a keystroke → visible glyph) had to round-trip through
   async plumbing, hurting latency and reasoning. The async boundary belonged at
   the *I/O edges*, not threaded through the whole core.
2. **The JSON-RPC frontend split was too chatty and too coarse.** Fine-grained
   editor state over JSON serialization added latency and a large, awkward
   protocol surface; keeping frontend and core in lockstep across the boundary
   was a persistent tax, and no frontend ever became fully first-class.
3. **CRDT-for-everything was premature.** The collaborative/async data model was a
   large upfront cost for a payoff most single-user editing never needed.
4. The project effectively wound down without shipping a mainstream editor;
   Levien's takeaways fed into later Rust-UI work (druid → xilem).

### STEAL

- **Put the async boundary at exactly one place — and Umber already has (PLAN
  Rule 1, D7).** PLAN's own words: *"Xi-editor died of async-everything; Zed
  lives with exactly this split. We copy Zed's shape, not xi's."* The `umber-proto`
  core↔backend seam is the *one* async boundary; everything inside the core
  (P0/P1 render + edit loop) stays synchronous and latency-bounded (P0's ≤8ms
  keystroke→present, D4). Keep it that way — this is the single most important
  inherited lesson.
- **Use a compact binary framing, not JSON, on the one boundary you keep (P3).**
  xi's chattiness over JSON is why §1 STEALs Zed's length-prefixed protobuf
  `Envelope`. Umber's proto types (`umber-proto`) should serialize compactly
  (protobuf/bincode/postcard), never fine-grained JSON per keystroke.
- **Synchronous-by-default, debounced-async-only-where-needed (kernel).** Pair
  this with Helix's `helix-event`/`AsyncHook` (§4): modules react via synchronous
  hooks except for genuinely long-running work, which is explicitly debounced —
  not the reverse.

### AVOID

- **Never make within-core editing async.** The keystroke→glyph→present path
  (P0, D4) must be a straight synchronous line; do not model local edits as
  messages to be awaited. Async is for fs/search/LSP/PTY/SSH/agents crossing
  `umber-proto` — nothing else.
- **Don't build a general JSON-RPC frontend protocol for the UI.** Umber's UI is
  in-process (D4, own wgpu engine); the *only* RPC is core↔workspace-backend. Do
  not turn the panel/module boundary into a second JSON-RPC frontend split — that
  is precisely xi's mistake. (Note the tension: pi integration in P4 *is*
  JSON-RPC over stdio — that's fine, because it's an external-process I/O edge
  behind `AgentBackend`/`umber-proto`, D11/D12, not the editor's own frontend.)
- **Don't adopt a CRDT/collaborative-edit core speculatively.** Umber is
  single-user local+remote (D7/D11), not multiplayer. Skip the CRDT machinery xi
  paid for up front; if real-time collab ever matters, it's a post-v0.1 module,
  not a core substrate.

---

## Cross-cutting summary (what all four teach Umber)

| Lesson | Sources | Umber tie |
|---|---|---|
| One async boundary at the backend seam; core stays synchronous | Zed §1, xi §5 (Lapce §3 concurs) | Rule 1, D7, P0/P1/P3 |
| State lives in the (headless) backend; UI is a thin client | Zed §1, Lapce §3 | D7, P3, P5 |
| Compact binary framing (length-prefixed), not per-keystroke JSON | Zed §1, xi §5 | umber-proto, P3 |
| One WIT/host ABI; TS+Lua are bindings over it | Zed §2 | D9, P2 |
| Deny-by-default capabilities w/ `*`/`**` matcher + capability handles | Zed §2 (Lapce §3 = counter-example) | D10, P2 permission broker |
| WASI **preview2 / component model**, not preview1 stdio-JSON | Zed §2 vs Lapce §3 | P2, D9 |
| Never `inherit_env`; never `allow-all` network | Lapce §3 (what NOT to do) | D10, P2 |
| Rope + Selection(head/anchor) + invertible Transaction | Helix §4 (all three concur) | umber-text, P1 |
| Compositor/Component layer stack for panels/pickers/palette | Helix §4 | D5, D6, P0/P1 |
| Signing + entitlements designed into the registry from day one | Lapce §3 (gap) | P6, D3 |
