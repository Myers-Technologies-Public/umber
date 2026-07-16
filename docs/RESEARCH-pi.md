# Research — pi Integration Surface for the P4 Agent Dashboard

Ground-truth deep-dive for [PLAN.md](PLAN.md) **P4** and [DECISIONS.md](DECISIONS.md)
**D11** (local + remote agents day 1) / **D12** (pi first, `AgentBackend` trait).

Every field name below was verified against real artifacts, not assumed:

- `docs/rpc.md` and `docs/session-format.md` shipped with
  `@earendil-works/pi-coding-agent`
  (`/home/asmartcow/.local/lib/node_modules/@earendil-works/pi-coding-agent/docs/`).
- Live session JSONL under
  `/home/asmartcow/.pi/agent/sessions/--home-asmartcow--/` (8 real sessions,
  read on 2026-07-16).

Where the docs and the on-disk reality diverge, the on-disk reality is called
out explicitly (see [§2.4](#24-doc-vs-reality-deltas-verified-on-disk)).

---

## 1. RPC protocol summary (`pi --mode rpc`)

The sanctioned, language-agnostic control surface — the only one usable from
Rust (the SDK is TypeScript, in-process only; confirmed in DECISIONS kickoff).

### 1.1 Transport & framing

- Launch: `pi --mode rpc [options]`. Relevant options:
  - `--provider <name>`, `--model <pattern>` (`provider/id[:thinking]`)
  - `--name <name>` / `-n <name>` — initial session display name
  - `--no-session` — disable persistence (no JSONL written)
  - `--session-dir <path>` — custom session storage dir
- **Commands** → written to **stdin**, one JSON object per line.
- **Responses** → JSON on **stdout**, `type: "response"`, one per command.
- **Events** → JSON on **stdout**, streamed asynchronously.
- **Framing is strict JSONL, LF (`\n`) only.** Split on `\n` only; strip a
  trailing `\r`. Do **not** use a reader that also breaks on `U+2028`/`U+2029`
  (they are valid inside JSON strings). Node `readline` is explicitly called
  non-compliant. → Our Rust reader must split on `\n` byte only.
- Correlation: commands may carry an optional `id`; the matching response
  echoes it. **Events never carry `id`** — that is how you tell a response
  from an event on the same stream.

### 1.2 Commands we care about (verified in rpc.md)

| Command | Purpose | Key response data |
|---|---|---|
| `prompt` | Send a user prompt. During streaming requires `streamingBehavior: "steer" \| "followUp"` or it errors. | `{success}` — accept/queue ack only; failures after acceptance come as events |
| `steer` | Queue a steering message mid-run (delivered after current turn's tool calls, before next LLM call) | `{success}` |
| `follow_up` | Queue a message delivered only when the agent stops | `{success}` |
| `abort` | Abort current operation | `{success}` |
| `get_state` | **Live status snapshot** | see [§1.4](#14-get_state-the-live-status-oracle) |
| `get_session_stats` | **Token/cost/context snapshot** | see [§1.5](#15-get_session_stats-the-token-oracle) |
| `get_messages` | Full conversation (active branch) | `{messages: AgentMessage[]}` |
| `get_entries` | Append-order entries **with a durable `since` cursor** | `{entries[], leafId}` |
| `get_tree` | Full session tree | `{tree, leafId}` |
| `get_last_assistant_text` | Last assistant text | `{text \| null}` |
| `set_session_name` | Name the session | `{success}` |
| `new_session` / `switch_session` / `fork` / `clone` | Session lifecycle | cancellable by extensions |
| `set_model` / `cycle_model` / `get_available_models` | Model control | full `Model` object(s) |
| `set_steering_mode` / `set_follow_up_mode` | Queue delivery policy (`all` / `one-at-a-time`) | `{success}` |
| `compact` / `set_auto_compaction` | Context compaction | token deltas |

`get_entries { since: <lastEntryId> }` is the ideal incremental cursor for a
dashboard tailing a live session: entry ids are stable, so the same cursor
survives client restarts, and `leafId` tells us in one round-trip whether the
active branch moved. If `since` is unknown the response is `success:false`.

### 1.3 Event stream (verified event table)

Streamed to stdout as JSON lines, no `id`. The dashboard-relevant ones:

| Event | Dashboard meaning |
|---|---|
| `agent_start` | run began → **state = running** |
| `agent_end` | one low-level run ended (may retry/continue) — **not** a reliable "idle" |
| `agent_settled` | fully settled, nothing queued/retrying → **state = awaiting instruction** |
| `turn_start` / `turn_end` | turn boundaries; `turn_end` carries `{message, toolResults}` |
| `message_start` / `message_end` | message lifecycle; `message` is an `AgentMessage` (carries `usage`) |
| `message_update` | streaming deltas — `assistantMessageEvent.{type,delta,...}` for `text_delta`, `thinking_delta`, `toolcall_delta`, `done`, `error` → **live output** |
| `tool_execution_start` / `_update` / `_end` | tool run + streaming partial output, correlate by `toolCallId`; `_update.partialResult` is cumulative (replace, don't append) |
| `queue_update` | `{steering:[], followUp:[]}` — pending queue changed → **"awaiting instruction" signal** |
| `compaction_start` / `_end` | context compaction |
| `auto_retry_start` / `_end` | transient-error retry (overloaded/429/5xx) |
| `extension_error` | an extension threw |

**State-machine rule (verified):** use `agent_start` → running,
`agent_settled` → idle/awaiting. Do **not** treat `agent_end` as idle — the doc
explicitly warns a retry, compaction retry, or queued continuation may follow.

### 1.4 `get_state` — the live status oracle

Verified response `data` shape:

```json
{
  "model": { ...Model|null },
  "thinkingLevel": "medium",
  "isStreaming": false,
  "isCompacting": false,
  "steeringMode": "all",
  "followUpMode": "one-at-a-time",
  "sessionFile": "/path/to/session.jsonl",
  "sessionId": "abc123",
  "sessionName": "my-feature-work",
  "autoCompactionEnabled": true,
  "messageCount": 5,
  "pendingMessageCount": 0
}
```

For state without tailing events: `isStreaming` ⇒ running; `!isStreaming`
⇒ awaiting; `pendingMessageCount > 0` ⇒ has queued steering/follow-up work.
`sessionFile` links the live process to its JSONL on disk.

### 1.5 `get_session_stats` — the token oracle

Verified response `data` shape:

```json
{
  "sessionFile": "/path/to/session.jsonl",
  "sessionId": "abc123",
  "userMessages": 5, "assistantMessages": 5,
  "toolCalls": 12, "toolResults": 12, "totalMessages": 22,
  "tokens": { "input": 50000, "output": 10000,
              "cacheRead": 40000, "cacheWrite": 5000, "total": 105000 },
  "cost": 0.45,
  "contextUsage": { "tokens": 60000, "contextWindow": 200000, "percent": 30 }
}
```

- `tokens` = assistant usage **totals** for the session (pi does the summation
  for us — we do not have to re-add per-message usage ourselves for a live
  session).
- `contextUsage` = the **current** context-window occupancy used for the
  footer/compaction gauge. Omitted when no model/context window; `.tokens` and
  `.percent` are `null` right after compaction until the next assistant reply.
- **This single call is the authoritative token/cost source for a *live*
  (RPC-attached) session.** JSONL summation (§2.3) is the fallback only for
  sessions with no live process.

### 1.6 Extension UI sub-protocol (relevant to the beacon, §4)

In RPC mode, extension UI calls surface as `extension_ui_request` on stdout.
Two classes:

- **Dialog** (`select`, `confirm`, `input`, `editor`) — block until the client
  replies with `extension_ui_response {id, ...}`. A dashboard that ignores
  these will stall the agent unless it answers (or the request carries a
  `timeout`, which auto-resolves).
- **Fire-and-forget** (`notify`, `setStatus`, `setWidget`, `setTitle`,
  `set_editor_text`) — pushed, no reply expected. `setStatus {statusKey,
  statusText}` and `notify {message, notifyType}` are exactly the "live status
  line" primitives a dashboard can surface for free.

---

## 2. Session file anatomy (JSONL on disk)

### 2.1 Location & naming (verified on disk)

```
~/.pi/agent/sessions/--<cwd-with-/-as-->--/<ISO-stamp>_<uuid>.jsonl
```

Real example: the cwd `/home/asmartcow` is encoded as the directory
`--home-asmartcow--`, containing e.g.
`2026-07-15T22-32-31-197Z_019f67e9-00dd-7b30-9ddd-b07fb9417500.jsonl`.
**Note:** the encoded dir is a real nesting level — a naive
`sessions/*.jsonl` glob finds nothing; you must descend into the per-cwd dirs.
Discovering sessions = walk `sessions/*/*.jsonl`.

### 2.2 Structure

- **Line 1 = `SessionHeader`** (metadata, not part of the tree, no
  `id`/`parentId`). Verified real header:

  ```json
  {"type":"session","version":3,"id":"019f67e9-00dd-7b30-9ddd-b07fb9417500",
   "timestamp":"2026-07-15T22:32:31.197Z","cwd":"/home/asmartcow"}
  ```

  `version:3` is current (v1 linear, v2 tree, v3 renamed `hookMessage`→`custom`);
  older files auto-migrate on load. `parentSession` appears here for
  forked/cloned sessions.

- **Remaining lines = tree entries**, each `{type, id (8-hex), parentId
  (|null), timestamp (ISO), ...}`. Branching is in-place via `parentId`; the
  "leaf" is the current position. Entry `type` values seen/known:
  `message`, `model_change`, `thinking_level_change`, `compaction`,
  `branch_summary`, `custom`, `custom_message`, `label`, `session_info`.

  Verified entry-type census of a 350-message real session:
  `message` ×350, `session` ×1, `model_change` ×1, `thinking_level_change` ×1.

- **`message` entry** wraps an `AgentMessage` in `.message`:
  `user` / `assistant` / `toolResult` / `bashExecution` / `custom` /
  `branchSummary` / `compactionSummary`.

- **`model_change` entry** (verified real):
  ```json
  {"type":"model_change","id":"afc715d1","parentId":null,
   "timestamp":"2026-07-15T22:32:32.140Z",
   "provider":"claude-max","modelId":"claude-fable-5"}
  ```

### 2.3 Where token/usage/model actually live (VERIFIED)

Token accounting lives on **each `assistant` message**, in `message.usage` —
**per message, not cumulative.** Verified assistant `message` keys:
`role, content, api, provider, model, usage, stopReason, timestamp, responseId`.

Verified `usage` keys (superset of what session-format.md documents):

```
input, output, cacheRead, cacheWrite, totalTokens, cost,
cacheWrite1h, reasoning
```

and `usage.cost` keys: `input, output, cacheRead, cacheWrite, total`.

**Redacted real assistant line** (content blanked; usage/meta verbatim):

```json
{"type":"message","id":"3f7ce726","parentId":"4a60aab6",
 "timestamp":"2026-07-15T22:33:12.865Z",
 "message":{"role":"assistant","content":[{"type":"text","text":"<redacted>"}],
   "api":"anthropic-messages","provider":"claude-max","model":"claude-fable-5",
   "usage":{"input":2,"output":566,"cacheRead":6309,"cacheWrite":5204,
            "totalTokens":12081,
            "cost":{"input":0,"output":0,"cacheRead":0,"cacheWrite":0,"total":0},
            "cacheWrite1h":5204,"reasoning":258},
   "stopReason":"toolUse","timestamp":1784154781207,
   "responseId":"msg_011Cd4dy1NX5E1gS6WGSeDb8"}}
```

To reconstruct **session totals from JSONL only** (no live RPC): sum
`message.usage.*` across all `assistant` entries on the active branch. Model in
force = latest `model_change` entry, else the `model`/`provider` on the assistant
messages. Current-context estimate ≈ the **last** assistant message's
`totalTokens` (cacheRead grows across the run — in a real session the last
message showed `cacheRead:137780, totalTokens:143586`), whereas the running
*spend* is the **sum** of per-message usage. Keep these two distinct — they are
the JSONL analogue of RPC's `tokens` (sum) vs `contextUsage` (last).

### 2.4 Doc-vs-reality deltas (verified on disk)

1. **`usage.cost.total` is `0` here** because the session ran on the
   `claude-max` subscription (`provider:"claude-max"`, `model:"claude-fable-5"`)
   — a flat-rate plan with no per-token price. **The dashboard must not present
   cost as "free/$0" — treat `cost==0` as "not metered on this plan" and lead
   with token counts.** Cost is only meaningful for metered API providers.
2. Real `usage` carries two keys **not** in session-format.md's `Usage`
   interface: **`cacheWrite1h`** (1-hour cache-write tier) and **`reasoning`**
   (reasoning-token count). The parser must ignore-unknown, not reject.
3. Assistant messages carry a top-level **`responseId`** not shown in the doc's
   `AssistantMessage`. Same tolerance rule.
4. Entry `message.timestamp` is a **Unix-ms number** while the enclosing
   entry `timestamp` is an **ISO string** — two different types, both present.
   Use the entry-level ISO timestamps for elapsed-time math.

> Design rule from these deltas (echoes PLAN "integrate only via documented
> surfaces"): our JSONL parser must be **permissive/forward-compatible** —
> deserialize known fields, ignore unknown, never hard-fail a session because a
> new key appeared.

---

## 3. Data-availability matrix — Umber dashboard requirement → verified pi source

Each P4 requirement mapped to its concrete pi source, with status. "Live" =
RPC-attached process; "History" = reading JSONL for a session we don't control.

| # | Umber requirement | Live source (RPC) | History source (JSONL) | Status |
|---|---|---|---|---|
| R1 | **State: running** | event `agent_start`; `get_state.isStreaming==true` | — (no reliable liveness in file) | **VERIFIED (live)** / **NOT-AVAILABLE (history)** |
| R2 | **State: awaiting instruction** | event `agent_settled`; `get_state.isStreaming==false` | — | **VERIFIED (live)** / **NOT-AVAILABLE (history)** |
| R3 | **Has queued work (steer/follow-up)** | event `queue_update {steering,followUp}`; `get_state.pendingMessageCount` | — | **VERIFIED (live only)** |
| R4 | **Elapsed / runtime duration** | derive from `agent_start`→`agent_settled` wall-clock (client-side timer) | entry-level ISO `timestamp` deltas (first↔last, or per-turn) | **VERIFIED** (computed, not a single field) |
| R5 | **Token usage (session totals)** | `get_session_stats.tokens {input,output,cacheRead,cacheWrite,total}` | sum `assistant.message.usage.*` | **VERIFIED** |
| R6 | **Context-window occupancy / %** | `get_session_stats.contextUsage {tokens,contextWindow,percent}` | last assistant `usage.totalTokens` vs `model.contextWindow` (estimate) | **VERIFIED (live exact)** / **VERIFIED-ESTIMATE (history)** |
| R7 | **Cost** | `get_session_stats.cost` | sum `assistant.message.usage.cost.total` | **VERIFIED but plan-dependent** — `0` on claude-max (R-note §2.4.1) |
| R8 | **Live streamed output** | events `message_update` (`text_delta`/`thinking_delta`), `tool_execution_update.partialResult` | tail file for new lines (coarse; whole messages only, no sub-message deltas) | **VERIFIED (live streaming)** / **VERIFIED-COARSE (history tail)** |
| R9 | **Send prompt** | command `prompt {message,images?}` | — | **VERIFIED (live only)** |
| R10 | **Steer** | command `steer {message}` or `prompt {streamingBehavior:"steer"}` | — | **VERIFIED (live only)** |
| R11 | **Follow-up** | command `follow_up {message}` or `prompt {streamingBehavior:"followUp"}` | — | **VERIFIED (live only)** |
| R12 | **Abort** | command `abort` | — | **VERIFIED (live only)** |
| R13 | **Model in use** | `get_state.model` (full `Model`) | latest `model_change`, else `message.model`/`provider` | **VERIFIED** |
| R14 | **Session identity / name** | `get_state.{sessionId,sessionName,sessionFile}` | header `id`/`cwd`; latest `session_info.name` | **VERIFIED** |
| R15 | **Conversation history / transcript** | `get_entries`(+`since` cursor) / `get_messages` / `get_tree` | parse all entries | **VERIFIED** |
| R16 | **Session discovery (list all)** | (per-process only) | walk `sessions/*/*.jsonl`, read header line 1 | **VERIFIED** |

### Load-bearing conclusions

- **Control (R9–R12) and live state (R1–R3) require an RPC process.** JSONL is
  read-only history — it cannot tell you a session is *currently* running, and
  it cannot send input. A dashboard row for a session we merely found on disk is
  **inherently "detached/unknown-state"** until we attach or `switch_session`
  onto it in an RPC process.
- **Two token concepts, keep them separate:** cumulative *spend* (sum) vs
  current *context occupancy* (last / `contextUsage`). RPC gives both directly;
  JSONL gives spend exactly and occupancy as an estimate.
- **Elapsed time is always computed, never read** — there is no duration field;
  derive from event wall-clock (live) or ISO `timestamp` deltas (history).
- **Cost is unreliable as a headline metric** on subscription plans (`0`). Lead
  with tokens; show cost only when `>0` / provider is metered.

---

## 4. Beacon note — can extensions push live status? (partial verification)

**Verified from rpc.md:** In RPC mode, pi extensions can already *push*
status outward via the fire-and-forget extension-UI messages — `setStatus`
(`{statusKey, statusText}`) and `notify` (`{message, notifyType}`) arrive on
stdout with no reply required. Extension commands drive their own LLM
interaction via `pi.sendMessage()`. So a client attached to an RPC process
gets extension-authored status for free on the existing stream.

**Verified from DECISIONS kickoff:** the pi SDK/extensions are **TypeScript,
in-process** with the agent, and the SDK exposes lifecycle events
(`agent_start`/`agent_end`, `turn_start`/`turn_end`, `tool_execution_*`,
`message_update`, `queue_update`). An in-process extension can therefore
subscribe to those and, being ordinary Node/TS, perform network I/O.

**Implication for the pi-remote beacon (D11, "hosts that aren't the workspace
backend"):** a small pi extension that subscribes to agent lifecycle events and
`POST`s `{host, sessionId, state, tokens, elapsed}` to an Umber registry
endpoint is the clean way to register non-workspace agents — no SSH channel
required, and it reuses events pi already emits. This keeps us on "documented
surfaces" (PLAN risk row).

> **Not re-verified this pass (be honest):** the exact extension event-hook
> *registration API* (method names, handler signatures) in `docs/extensions.md`
> was not cleanly re-read here due to a tooling hiccup. Before building the
> beacon, read `docs/extensions.md` end-to-end and confirm the event-subscription
> API and whether extensions may open outbound sockets under pi's permission
> model. The beacon design above is sound in shape but its API details are
> **pending verification**, not established fact.

---

## 5. Recommended `AgentBackend` trait shape

Driven by the matrix: a backend is **two capabilities** — *observe* (works for
history and live) and *control* (live only). Modeling them separately keeps a
detached, file-only session representable without faking control it can't do.

```rust
/// One discovered agent session (live or historical).
pub struct SessionSummary {
    pub id: String,               // header uuid / get_state.sessionId
    pub name: Option<String>,     // session_info.name / get_state.sessionName
    pub cwd: PathBuf,             // header.cwd
    pub file: Option<PathBuf>,    // sessionFile (None for --no-session)
    pub model: Option<ModelInfo>, // latest model_change / get_state.model
    pub state: AgentState,
    pub started_at: OffsetDateTime, // header/first-entry timestamp
    pub last_activity: OffsetDateTime,
}

pub enum AgentState {
    Running,             // agent_start seen / isStreaming
    AwaitingInstruction, // agent_settled / !isStreaming
    AwaitingWithQueue,   // pendingMessageCount>0 / queue_update non-empty
    Detached,            // found on disk, no RPC process attached — state unknown
    Errored(String),
    Exited,
}

pub struct TokenStats {           // maps get_session_stats
    pub input: u64, pub output: u64,
    pub cache_read: u64, pub cache_write: u64,
    pub total: u64,
    pub cost_usd: Option<f64>,    // None when unmetered (claude-max => 0)
    pub context_tokens: Option<u64>,     // contextUsage.tokens
    pub context_window: Option<u64>,     // contextUsage.contextWindow
    pub context_percent: Option<f32>,    // contextUsage.percent
}

/// Streamed to the dashboard panel. Superset of the pi events we surface.
pub enum AgentEvent {
    Started, Settled,
    TurnStarted, TurnEnded,
    OutputDelta { kind: DeltaKind, text: String }, // text/thinking/tool deltas
    ToolStarted { call_id: String, tool: String },
    ToolOutput  { call_id: String, cumulative: String },
    ToolEnded   { call_id: String, is_error: bool },
    QueueChanged { steering: Vec<String>, follow_up: Vec<String> },
    Tokens(TokenStats),
    Status { key: String, text: Option<String> }, // extension setStatus/notify
    Error(String),
}

pub enum DeltaKind { Text, Thinking, ToolCall }

/// Observe-only: satisfiable by JSONL history OR a live process.
#[async_trait]
pub trait AgentObserve {
    async fn list_sessions(&self) -> Result<Vec<SessionSummary>>;   // walk sessions/*/*.jsonl or per-process
    async fn stats(&self, id: &str) -> Result<TokenStats>;          // get_session_stats or JSONL sum
    async fn transcript(&self, id: &str, since: Option<&str>)       // get_entries(since) or file tail
        -> Result<(Vec<Entry>, /*leaf*/ Option<String>)>;
}

/// Control: live RPC process only. A detached session offers None of this
/// until `attach`/`switch` puts it under an RPC process.
#[async_trait]
pub trait AgentControl: AgentObserve {
    async fn prompt(&self, id: &str, msg: Prompt) -> Result<()>;
    async fn steer(&self, id: &str, msg: Prompt) -> Result<()>;
    async fn follow_up(&self, id: &str, msg: Prompt) -> Result<()>;
    async fn abort(&self, id: &str) -> Result<()>;
    async fn set_model(&self, id: &str, provider: &str, model_id: &str) -> Result<()>;
    /// Live event tail; the panel consumes this for R8 streaming.
    fn subscribe(&self, id: &str) -> BoxStream<'static, AgentEvent>;
}

/// What P4 wires up. Two impls behind it:
///   PiLocalBackend  — spawns/attaches `pi --mode rpc` on the local host,
///                     lists history from ~/.pi/agent/sessions/*/*.jsonl.
///   PiRemoteBackend — same commands executed on the workspace backend
///                     (umberd over the existing SSH channel), plus a
///                     beacon-registered read view for non-workspace hosts.
pub trait AgentBackend: AgentControl + Send + Sync {
    fn id(&self) -> &str;         // "pi-local" | "pi-remote:<host>"
    fn is_live(&self, id: &str) -> bool; // true => AgentControl usable
}
```

### Why this shape (traceable to the matrix)

- **`AgentObserve` vs `AgentControl` split** = the R1–R3/R9–R12 "live only" vs
  R5/R15/R16 "works from history" divide. A `Detached` session is a valid
  `SessionSummary` exposing only `AgentObserve`; it does not pretend to accept
  prompts.
- **`TokenStats.cost_usd: Option<f64>`** encodes §2.4.1 — `None`/unmetered vs a
  real metered cost, so the UI never mislabels a `0` as "free".
- **`context_*` separate from `input/output/...`** encodes the spend-vs-occupancy
  distinction (R5 vs R6).
- **`AgentEvent::Status`** carries extension `setStatus`/`notify` (§1.6) so the
  same channel that feeds live output also feeds the beacon/status line.
- **`subscribe` returns a stream** so the panel (tui + plain surfaces, PLAN D5)
  can render `OutputDelta`/`ToolOutput` incrementally, replacing on cumulative
  tool output as the RPC doc requires.
- **`is_live`** lets the panel gray-out control affordances for detached rows —
  the honest UI for "history only, can't drive it."

### Backends (P4 exit criteria mapping)

- **`PiLocalBackend`** — owns a `pi --mode rpc` child per driven session
  (stdin/stdout JSONL, LF framing per §1.1); discovers history by walking
  `~/.pi/agent/sessions/*/*.jsonl`. Attaching to an existing on-disk session =
  spawn RPC + `switch_session {sessionPath}`.
- **`PiRemoteBackend`** — identical command set executed by `umberd` on moo,
  riding the existing P3 SSH channel (PLAN "remote agents ride the existing SSH
  channel"); the beacon (§4) feeds a read-only registry view for hosts that are
  not the workspace backend, satisfying D11's "non-workspace hosts register."

---

## 6. Open items before P4 build (honest gaps)

1. **`docs/extensions.md` not fully re-read this pass** — confirm the exact
   event-subscription API and the extension network/permission model before
   committing to the beacon (§4).
2. **Multi-session per RPC process:** the RPC surface is single-session
   (`switch_session` swaps the *one* active session). Driving N live sessions
   ⇒ N `pi --mode rpc` processes. Confirm this is acceptable vs. a pool.
3. **`contextWindow` for history estimates** (R6) needs a model→window table;
   live sessions get it from `get_session_stats`. Source a static map or read it
   from `get_available_models` once at attach.
4. **Cost display policy** — decide the UI rule for `cost==0` / unmetered plans
   (§2.4.1) so subscription usage isn't shown as free.
