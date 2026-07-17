//! Live pi agent control over `pi --mode rpc` (P4 slice 2).
//!
//! pi's SDK is TypeScript/in-process only; the sanctioned language-agnostic
//! surface is `pi --mode rpc`: newline-delimited JSON, commands to stdin,
//! responses + events on stdout (docs/RESEARCH-pi.md §1). This module owns a
//! child `pi` process, a stdout reader thread, and the classification of that
//! stream into dashboard state.
//!
//! The framing + classification core is pure and unit-tested; the process
//! plumbing is a thin shell around it so tests never spawn `pi`.
//!
//! Framing rules (verified, §1.1): split on the `\n` BYTE only (never
//! `U+2028`/`U+2029`, which are valid inside JSON strings), strip a trailing
//! `\r`. Responses carry the echoed command `id`; events never carry `id` —
//! that is how they are told apart on one stream.
//!
//! State-machine rule (verified, §1.3): `agent_start` -> Running,
//! `agent_settled` -> AwaitingInstruction. `agent_end` is NOT idle (a retry,
//! compaction, or queued continuation may follow).

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use serde_json::{json, Value};

/// Live run-state of an attached agent (docs/RESEARCH-pi.md §1.3/§1.4).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AgentRunState {
    /// Between spawn and the first `agent_start`.
    Starting,
    /// `agent_start` seen, not yet settled — the agent is working.
    Running,
    /// `agent_settled` — fully idle, awaiting instruction.
    AwaitingInstruction,
    /// Pending steering/follow-up work queued (`queue_update` non-empty).
    Queued,
    /// The child exited.
    Exited,
}

/// One classified line from pi's stdout. `Other` covers lines we parse but do
/// not act on (kept so the reader never silently drops malformed-looking but
/// valid frames).
#[derive(Clone, Debug, PartialEq)]
pub enum RpcInbound {
    /// A `type:"response"` line, with the echoed command id (if any).
    Response { id: Option<u64>, success: bool },
    /// A streaming assistant text delta (`message_update` text_delta).
    TextDelta(String),
    /// A run-state transition derived from an event.
    State(AgentRunState),
    /// A tool started (name for the status line).
    ToolStart(String),
    /// A parseable line we don't surface.
    Other,
}

/// Classify one already-parsed JSON line into an [`RpcInbound`]. Pure — the
/// heart of the protocol, fully unit-tested. A line is a *response* iff its
/// `type` is `"response"`; everything else is an event (§1.1).
pub fn classify(v: &Value) -> RpcInbound {
    let ty = v.get("type").and_then(Value::as_str).unwrap_or("");
    if ty == "response" {
        return RpcInbound::Response {
            id: v.get("id").and_then(Value::as_u64),
            // Absent `success` is treated as success (some responses carry only
            // `data`); an explicit `false` is the failure signal.
            success: v.get("success").and_then(Value::as_bool).unwrap_or(true),
        };
    }
    match ty {
        "agent_start" => RpcInbound::State(AgentRunState::Running),
        "agent_settled" => RpcInbound::State(AgentRunState::AwaitingInstruction),
        "queue_update" => {
            let steering = v
                .get("steering")
                .and_then(Value::as_array)
                .map(|a| a.len())
                .unwrap_or(0);
            let follow = v
                .get("followUp")
                .and_then(Value::as_array)
                .map(|a| a.len())
                .unwrap_or(0);
            if steering + follow > 0 {
                RpcInbound::State(AgentRunState::Queued)
            } else {
                RpcInbound::Other
            }
        }
        "message_update" => {
            let ev = v.get("assistantMessageEvent");
            let is_text = ev
                .and_then(|e| e.get("type"))
                .and_then(Value::as_str)
                .map(|t| t == "text_delta")
                .unwrap_or(false);
            if is_text {
                let delta = ev
                    .and_then(|e| e.get("delta"))
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                RpcInbound::TextDelta(delta)
            } else {
                RpcInbound::Other
            }
        }
        "tool_execution_start" => {
            let name = v
                .get("toolName")
                .and_then(Value::as_str)
                .unwrap_or("tool")
                .to_string();
            RpcInbound::ToolStart(name)
        }
        _ => RpcInbound::Other,
    }
}

/// Parse + classify one raw stdout line. Splitting on `\n` is the caller's job
/// (§1.1); here we strip a trailing `\r` and ignore blank/unparseable lines.
pub fn classify_line(line: &str) -> Option<RpcInbound> {
    let line = line.strip_suffix('\r').unwrap_or(line);
    if line.trim().is_empty() {
        return None;
    }
    let v: Value = serde_json::from_str(line).ok()?;
    Some(classify(v_ref(&v)))
}

fn v_ref(v: &Value) -> &Value {
    v
}

/// Serialize a `prompt` command. During streaming pi requires an explicit
/// `streamingBehavior` or it errors (§1.2); callers pass `Some("steer")` /
/// `Some("followUp")` while a run is active.
pub fn prompt_command(id: u64, text: &str, streaming_behavior: Option<&str>) -> String {
    let mut cmd = json!({"type": "prompt", "id": id, "text": text});
    if let Some(b) = streaming_behavior {
        cmd["streamingBehavior"] = json!(b);
    }
    cmd.to_string()
}

/// Serialize a bare command carrying only a type + id (`abort`, `get_state`,
/// `get_session_stats`, ...).
pub fn simple_command(id: u64, ty: &str) -> String {
    json!({"type": ty, "id": id}).to_string()
}

/// Shared live view of an attached agent, updated by the reader thread and
/// read by the UI thread.
#[derive(Default)]
pub struct AgentLiveState {
    pub run_state: Mutex<Option<AgentRunState>>,
    /// Rolling tail of streamed assistant text (bounded).
    pub output_tail: Mutex<String>,
    pub last_tool: Mutex<Option<String>>,
}

/// Max bytes retained in the streamed-output tail (older text rolls off).
const OUTPUT_TAIL_CAP: usize = 8192;

impl AgentLiveState {
    fn apply(&self, inbound: &RpcInbound) {
        match inbound {
            RpcInbound::State(s) => *self.run_state.lock().unwrap() = Some(*s),
            RpcInbound::TextDelta(d) => {
                let mut tail = self.output_tail.lock().unwrap();
                tail.push_str(d);
                if tail.len() > OUTPUT_TAIL_CAP {
                    let cut = tail.len() - OUTPUT_TAIL_CAP;
                    // Trim on a char boundary so the String stays valid.
                    let cut = (cut..tail.len())
                        .find(|&i| tail.is_char_boundary(i))
                        .unwrap_or(tail.len());
                    *tail = tail.split_off(cut);
                }
            }
            RpcInbound::ToolStart(name) => *self.last_tool.lock().unwrap() = Some(name.clone()),
            _ => {}
        }
    }

    pub fn run_state(&self) -> Option<AgentRunState> {
        *self.run_state.lock().unwrap()
    }
    pub fn output_tail(&self) -> String {
        self.output_tail.lock().unwrap().clone()
    }
    pub fn last_tool(&self) -> Option<String> {
        self.last_tool.lock().unwrap().clone()
    }
}

/// Woken by the reader thread when new inbound data updated the live state.
pub trait AgentNotifier: Clone + Send + 'static {
    fn agent_updated(&self);
}

/// A spawned, attached `pi --mode rpc` process.
pub struct AgentProcess {
    child: Child,
    stdin: ChildStdin,
    next_id: AtomicU64,
    pub state: Arc<AgentLiveState>,
    reader: Option<std::thread::JoinHandle<()>>,
}

impl AgentProcess {
    /// Spawn `pi --mode rpc` in `cwd` and start the reader thread. `program`
    /// is normally `"pi"`; tests inject a stub that speaks the same JSONL.
    pub fn spawn<N: AgentNotifier>(
        program: &str,
        cwd: &std::path::Path,
        notifier: N,
    ) -> std::io::Result<Self> {
        let mut child = Command::new(program)
            .arg("--mode")
            .arg("rpc")
            .current_dir(cwd)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()?;
        let stdin = child.stdin.take().expect("piped stdin");
        let stdout = child.stdout.take().expect("piped stdout");
        let state = Arc::new(AgentLiveState::default());
        *state.run_state.lock().unwrap() = Some(AgentRunState::Starting);

        let reader_state = state.clone();
        let reader = std::thread::spawn(move || {
            // Split on the `\n` byte only (§1.1): BufRead::lines does exactly
            // this and never breaks on U+2028/U+2029.
            let buf = BufReader::new(stdout);
            for line in buf.lines() {
                let Ok(line) = line else { break };
                if let Some(inbound) = classify_line(&line) {
                    reader_state.apply(&inbound);
                    notifier.agent_updated();
                }
            }
            *reader_state.run_state.lock().unwrap() = Some(AgentRunState::Exited);
            notifier.agent_updated();
        });

        Ok(Self {
            child,
            stdin,
            next_id: AtomicU64::new(1),
            state,
            reader: Some(reader),
        })
    }

    fn next_id(&self) -> u64 {
        self.next_id.fetch_add(1, Ordering::Relaxed)
    }

    fn send_line(&mut self, line: &str) -> std::io::Result<()> {
        self.stdin.write_all(line.as_bytes())?;
        self.stdin.write_all(b"\n")?;
        self.stdin.flush()
    }

    /// Send a prompt. While the agent is Running, pass a streaming behavior
    /// (`"steer"` or `"followUp"`); when idle, `None`.
    pub fn prompt(&mut self, text: &str, streaming_behavior: Option<&str>) -> std::io::Result<()> {
        let id = self.next_id();
        let line = prompt_command(id, text, streaming_behavior);
        self.send_line(&line)
    }

    /// Abort the current operation.
    pub fn abort(&mut self) -> std::io::Result<()> {
        let id = self.next_id();
        let line = simple_command(id, "abort");
        self.send_line(&line)
    }

    /// Stop the process: close stdin (pi exits on EOF), reap the child, join
    /// the reader. Bounded — never blocks the UI thread indefinitely.
    pub fn shutdown(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        if let Some(reader) = self.reader.take() {
            let _ = reader.join();
        }
    }
}

impl Drop for AgentProcess {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn responses_are_distinguished_from_events_by_type() {
        let r = classify_line(r#"{"type":"response","id":7,"success":true}"#).unwrap();
        assert_eq!(
            r,
            RpcInbound::Response {
                id: Some(7),
                success: true
            }
        );
        // An event on the same stream, no id.
        let e = classify_line(r#"{"type":"agent_start"}"#).unwrap();
        assert_eq!(e, RpcInbound::State(AgentRunState::Running));
    }

    #[test]
    fn settled_is_idle_but_end_is_not() {
        assert_eq!(
            classify_line(r#"{"type":"agent_settled"}"#).unwrap(),
            RpcInbound::State(AgentRunState::AwaitingInstruction)
        );
        // agent_end must NOT be classified as idle (verified §1.3).
        assert_eq!(
            classify_line(r#"{"type":"agent_end"}"#).unwrap(),
            RpcInbound::Other
        );
    }

    #[test]
    fn text_deltas_extracted_other_updates_ignored() {
        let t = classify_line(
            r#"{"type":"message_update","assistantMessageEvent":{"type":"text_delta","delta":"hello"}}"#,
        )
        .unwrap();
        assert_eq!(t, RpcInbound::TextDelta("hello".to_string()));
        let th = classify_line(
            r#"{"type":"message_update","assistantMessageEvent":{"type":"thinking_delta","delta":"x"}}"#,
        )
        .unwrap();
        assert_eq!(th, RpcInbound::Other);
    }

    #[test]
    fn queue_update_state_depends_on_contents() {
        assert_eq!(
            classify_line(r#"{"type":"queue_update","steering":["m"],"followUp":[]}"#).unwrap(),
            RpcInbound::State(AgentRunState::Queued)
        );
        assert_eq!(
            classify_line(r#"{"type":"queue_update","steering":[],"followUp":[]}"#).unwrap(),
            RpcInbound::Other
        );
    }

    #[test]
    fn framing_tolerates_cr_blank_and_garbage() {
        assert!(classify_line("").is_none());
        assert!(classify_line("   ").is_none());
        assert!(classify_line("not json").is_none());
        // Trailing \r stripped (CRLF-ish producers).
        assert_eq!(
            classify_line("{\"type\":\"agent_start\"}\r").unwrap(),
            RpcInbound::State(AgentRunState::Running)
        );
    }

    #[test]
    fn commands_serialize_to_expected_json() {
        assert_eq!(
            prompt_command(3, "hi", None),
            r#"{"id":3,"text":"hi","type":"prompt"}"#
        );
        assert_eq!(
            prompt_command(4, "stop", Some("steer")),
            r#"{"id":4,"streamingBehavior":"steer","text":"stop","type":"prompt"}"#
        );
        assert_eq!(simple_command(9, "abort"), r#"{"id":9,"type":"abort"}"#);
    }

    #[test]
    fn output_tail_is_bounded_and_char_safe() {
        let state = AgentLiveState::default();
        for _ in 0..5000 {
            state.apply(&RpcInbound::TextDelta("\u{1f600}".to_string())); // 4-byte
        }
        let tail = state.output_tail();
        assert!(tail.len() <= OUTPUT_TAIL_CAP);
        // Still valid UTF-8 (no split emoji) — String guarantees it, but assert
        // the char count is whole.
        assert!(tail.chars().all(|c| c == '\u{1f600}'));
    }
}
