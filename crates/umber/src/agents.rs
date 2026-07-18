//! pi agent session observability (P4 slice 1: read-only history).
//!
//! Parses pi's session JSONL trees from `~/.pi/agent/sessions/<encoded-cwd>/`
//! per the verified anatomy in docs/RESEARCH-pi.md §2:
//!
//! - line 1 is a `{"type":"session",...}` header (cwd, timestamp, version)
//! - remaining lines are tree entries with `id`/`parentId`; the ACTIVE branch
//!   is the parent chain of the last appended entry (in-place branching means
//!   earlier siblings may be abandoned — they must not count)
//! - token usage lives per `assistant` message in `message.usage`
//!   (`totalTokens` per message; SUM over the branch = spend, the LAST
//!   assistant message's `totalTokens` ≈ current context size)
//! - model/provider in force = latest `model_change` entry on the branch,
//!   else the assistant messages' own `model`/`provider`
//!
//! Live state + control (running/awaiting, steer, prompt) require an RPC
//! process (`pi --mode rpc`) — that is slice 2; disk-only sessions here are
//! inherently `Detached`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde_json::Value;

/// A parsed, detached (history-only) pi session.
#[derive(Clone, Debug)]
pub struct SessionSummary {
    pub path: PathBuf,
    pub cwd: String,
    pub started: String,
    pub last_active: String,
    /// `message`-type entries on the active branch.
    pub messages: usize,
    pub provider: String,
    pub model: String,
    /// Sum of `usage.totalTokens` over assistant messages on the active
    /// branch (the session's running token spend).
    pub tokens_total: u64,
    /// The LAST assistant message's `totalTokens` (~ current context size).
    pub context_tokens: u64,
    /// Seconds since the session file was last written (activity signal: a
    /// recently-written JSONL means an agent is (or just was) running it).
    pub age_secs: u64,
}

/// `~/.pi/agent/sessions`, or `None` when `$HOME` is unset.
pub fn sessions_root() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    Some(
        PathBuf::from(home)
            .join(".pi")
            .join("agent")
            .join("sessions"),
    )
}

/// Discover and parse up to `limit` sessions under `root`, newest (by file
/// mtime) first. Sessions nest one level deep (`sessions/<cwd-dir>/*.jsonl`
/// — a flat glob finds nothing, RESEARCH-pi.md §2.1). Unreadable or
/// unparseable files are skipped, never fatal.
pub fn discover_sessions(root: &Path, limit: usize) -> Vec<SessionSummary> {
    let mut files: Vec<(std::time::SystemTime, PathBuf)> = Vec::new();
    let Ok(dirs) = std::fs::read_dir(root) else {
        return Vec::new();
    };
    for dir in dirs.flatten() {
        let Ok(entries) = std::fs::read_dir(dir.path()) else {
            continue;
        };
        for f in entries.flatten() {
            let p = f.path();
            if p.extension().map(|e| e == "jsonl").unwrap_or(false) {
                let mtime = f
                    .metadata()
                    .and_then(|m| m.modified())
                    .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
                files.push((mtime, p));
            }
        }
    }
    files.sort_by(|a, b| b.0.cmp(&a.0));
    let now = std::time::SystemTime::now();
    files
        .into_iter()
        .take(limit)
        .filter_map(|(mtime, p)| {
            let text = std::fs::read_to_string(&p).ok()?;
            let mut s = parse_session(&text, &p)?;
            s.age_secs = now
                .duration_since(mtime)
                .map(|d| d.as_secs())
                .unwrap_or(u64::MAX);
            Some(s)
        })
        .collect()
}

/// Parse one session JSONL. Pure (testable headlessly). Returns `None` for
/// files without a valid session header. Unknown keys/entry types are
/// ignored (the format grows — RESEARCH-pi.md §2.4).
pub fn parse_session(text: &str, path: &Path) -> Option<SessionSummary> {
    let mut lines = text.lines();
    let header: Value = serde_json::from_str(lines.next()?).ok()?;
    if header.get("type").and_then(Value::as_str) != Some("session") {
        return None;
    }
    let cwd = header
        .get("cwd")
        .and_then(Value::as_str)
        .unwrap_or("?")
        .to_string();
    let started = header
        .get("timestamp")
        .and_then(Value::as_str)
        .unwrap_or("?")
        .to_string();

    // Minimal per-entry record for the branch walk.
    struct Entry {
        parent: Option<String>,
        timestamp: String,
        kind: EntryKind,
    }
    enum EntryKind {
        Message {
            assistant_total: Option<u64>,
            provider: Option<String>,
            model: Option<String>,
        },
        ModelChange {
            provider: String,
            model: String,
        },
        Other,
    }

    let mut entries: HashMap<String, Entry> = HashMap::new();
    let mut last_id: Option<String> = None;
    for line in lines {
        let Ok(v) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        let Some(id) = v.get("id").and_then(Value::as_str) else {
            continue;
        };
        let parent = v
            .get("parentId")
            .and_then(Value::as_str)
            .map(str::to_string);
        let timestamp = v
            .get("timestamp")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let kind = match v.get("type").and_then(Value::as_str) {
            Some("message") => {
                let msg = v.get("message");
                let is_assistant = msg
                    .and_then(|m| m.get("role"))
                    .and_then(Value::as_str)
                    .map(|r| r == "assistant")
                    .unwrap_or(false);
                let assistant_total = if is_assistant {
                    msg.and_then(|m| m.get("usage"))
                        .and_then(|u| u.get("totalTokens"))
                        .and_then(Value::as_u64)
                        .or(Some(0))
                } else {
                    None
                };
                EntryKind::Message {
                    assistant_total,
                    provider: msg
                        .and_then(|m| m.get("provider"))
                        .and_then(Value::as_str)
                        .map(str::to_string),
                    model: msg
                        .and_then(|m| m.get("model"))
                        .and_then(Value::as_str)
                        .map(str::to_string),
                }
            }
            Some("model_change") => EntryKind::ModelChange {
                provider: v
                    .get("provider")
                    .and_then(Value::as_str)
                    .unwrap_or("?")
                    .to_string(),
                model: v
                    .get("modelId")
                    .and_then(Value::as_str)
                    .unwrap_or("?")
                    .to_string(),
            },
            _ => EntryKind::Other,
        };
        entries.insert(
            id.to_string(),
            Entry {
                parent,
                timestamp,
                kind,
            },
        );
        last_id = Some(id.to_string());
    }

    // Walk the active branch: parent chain from the last appended entry.
    let mut messages = 0usize;
    let mut tokens_total = 0u64;
    let mut context_tokens = 0u64;
    let mut provider = String::new();
    let mut model = String::new();
    let mut last_active = started.clone();
    let mut cursor = last_id;
    let mut hops = 0usize;
    while let Some(id) = cursor {
        let Some(e) = entries.get(&id) else { break };
        // Corrupt parent cycles must not hang the UI.
        hops += 1;
        if hops > entries.len() + 1 {
            break;
        }
        if last_active == started && !e.timestamp.is_empty() {
            last_active = e.timestamp.clone();
        }
        match &e.kind {
            EntryKind::Message {
                assistant_total,
                provider: p,
                model: m,
            } => {
                messages += 1;
                if let Some(t) = assistant_total {
                    tokens_total += t;
                    // First assistant hit on the walk = newest = context size.
                    if context_tokens == 0 {
                        context_tokens = *t;
                    }
                }
                if model.is_empty() {
                    if let (Some(p), Some(m)) = (p, m) {
                        provider = p.clone();
                        model = m.clone();
                    }
                }
            }
            // Walking leaf->root, the FIRST model_change seen is the latest.
            EntryKind::ModelChange {
                provider: p,
                model: m,
            } => {
                if model.is_empty() {
                    provider = p.clone();
                    model = m.clone();
                }
            }
            EntryKind::Other => {}
        }
        cursor = e.parent.clone();
    }
    if model.is_empty() {
        model = "?".to_string();
        provider = "?".to_string();
    }

    Some(SessionSummary {
        path: path.to_path_buf(),
        cwd,
        started,
        last_active,
        messages,
        provider,
        model,
        tokens_total,
        context_tokens,
        age_secs: u64::MAX,
    })
}

/// Compact age for dashboard rows: `95` -> `"1m ago"`.
pub fn fmt_age(secs: u64) -> String {
    if secs == u64::MAX {
        "?".to_string()
    } else if secs < 60 {
        format!("{secs}s ago")
    } else if secs < 3600 {
        format!("{}m ago", secs / 60)
    } else if secs < 86400 {
        format!("{}h ago", secs / 3600)
    } else {
        format!("{}d ago", secs / 86400)
    }
}

/// Compact token count for dashboard rows: `143586` -> `"143.6k"`.
pub fn fmt_tokens(n: u64) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}k", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn assistant(id: &str, parent: &str, total: u64) -> String {
        format!(
            r#"{{"type":"message","id":"{id}","parentId":"{parent}","timestamp":"2026-07-15T22:33:12.865Z","message":{{"role":"assistant","content":[],"provider":"claude-max","model":"claude-fable-5","usage":{{"input":2,"output":5,"cacheRead":0,"cacheWrite":0,"totalTokens":{total},"cost":{{"input":0,"output":0,"cacheRead":0,"cacheWrite":0,"total":0}},"cacheWrite1h":0,"reasoning":1}}}}}}"#
        )
    }

    fn user(id: &str, parent: Option<&str>) -> String {
        let parent = match parent {
            Some(p) => format!(r#""{p}""#),
            None => "null".to_string(),
        };
        format!(
            r#"{{"type":"message","id":"{id}","parentId":{parent},"timestamp":"2026-07-15T22:32:40.000Z","message":{{"role":"user","content":"hi"}}}}"#
        )
    }

    const HEADER: &str = r#"{"type":"session","version":3,"id":"019f","timestamp":"2026-07-15T22:32:31.197Z","cwd":"/home/asmartcow"}"#;

    #[test]
    fn sums_usage_and_reads_model_on_active_branch() {
        let text = [
            HEADER.to_string(),
            r#"{"type":"model_change","id":"aaaaaaaa","parentId":null,"timestamp":"2026-07-15T22:32:32.140Z","provider":"claude-max","modelId":"claude-fable-5"}"#.to_string(),
            user("bbbbbbbb", Some("aaaaaaaa")),
            assistant("cccccccc", "bbbbbbbb", 1000),
            assistant("dddddddd", "cccccccc", 2500),
        ]
        .join("\n");
        let s = parse_session(&text, Path::new("/tmp/x.jsonl")).unwrap();
        assert_eq!(s.cwd, "/home/asmartcow");
        assert_eq!(s.messages, 3);
        assert_eq!(s.tokens_total, 3500);
        assert_eq!(s.context_tokens, 2500); // last assistant, not the sum
        assert_eq!(s.provider, "claude-max");
        assert_eq!(s.model, "claude-fable-5");
    }

    #[test]
    fn abandoned_branch_does_not_count() {
        // b has two children: c1 (abandoned) and c2 (active, appended later).
        let text = [
            HEADER.to_string(),
            user("bbbbbbbb", None),
            assistant("c1c1c1c1", "bbbbbbbb", 9999),
            assistant("c2c2c2c2", "bbbbbbbb", 100),
        ]
        .join("\n");
        let s = parse_session(&text, Path::new("/tmp/x.jsonl")).unwrap();
        assert_eq!(s.tokens_total, 100, "abandoned sibling must not count");
        assert_eq!(s.messages, 2);
    }

    #[test]
    fn rejects_non_session_files_and_survives_garbage() {
        assert!(parse_session("not json", Path::new("/tmp/x")).is_none());
        assert!(parse_session(r#"{"type":"other"}"#, Path::new("/tmp/x")).is_none());
        let with_garbage = format!("{HEADER}\nnot-json-line\n{}", user("bbbbbbbb", None));
        let s = parse_session(&with_garbage, Path::new("/tmp/x")).unwrap();
        assert_eq!(s.messages, 1);
    }

    #[test]
    fn parent_cycle_terminates() {
        // a -> b -> a: corrupt, but must not hang.
        let text = [
            HEADER.to_string(),
            r#"{"type":"message","id":"aaaaaaaa","parentId":"bbbbbbbb","timestamp":"","message":{"role":"user","content":"x"}}"#.to_string(),
            r#"{"type":"message","id":"bbbbbbbb","parentId":"aaaaaaaa","timestamp":"","message":{"role":"user","content":"y"}}"#.to_string(),
        ]
        .join("\n");
        assert!(parse_session(&text, Path::new("/tmp/x")).is_some());
    }

    #[test]
    fn token_formatting() {
        assert_eq!(fmt_tokens(950), "950");
        assert_eq!(fmt_tokens(143_586), "143.6k");
        assert_eq!(fmt_tokens(2_400_000), "2.4M");
    }
}
