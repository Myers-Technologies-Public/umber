//! Language Server Protocol client (P5c).
//!
//! LSP is JSON-RPC over stdio with `Content-Length` framing (NOT the JSONL that
//! pi-rpc and umber-proto use — LSP predates that convention). This module owns
//! the framing, the message builders, diagnostic parsing, and a spawned server
//! process with a reader thread. Framing + builders + parsing are pure and
//! unit-tested; only the process wrapper needs a real server.
//!
//! Slice scope: initialize handshake, `didOpen`/`didChange`, and
//! `publishDiagnostics` (the squiggles). Completion/hover/definition are
//! follow-ups built on the same transport.

use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Arc, Mutex};

use serde_json::{json, Value};

/// Diagnostic severity (LSP numeric values 1..=4).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Severity {
    Error,
    Warning,
    Information,
    Hint,
}

impl Severity {
    fn from_lsp(n: u64) -> Severity {
        match n {
            1 => Severity::Error,
            2 => Severity::Warning,
            3 => Severity::Information,
            _ => Severity::Hint,
        }
    }
}

/// One diagnostic, positions 0-based (LSP native) — line and character.
#[derive(Clone, Debug, PartialEq)]
pub struct Diagnostic {
    pub line: usize,
    pub col: usize,
    pub end_line: usize,
    pub end_col: usize,
    pub severity: Severity,
    pub message: String,
}

// --- framing ---------------------------------------------------------------

/// Encode a JSON value as an LSP message: `Content-Length: N\r\n\r\n` + body.
pub fn encode_message(value: &Value) -> Vec<u8> {
    let body = serde_json::to_vec(value).expect("serialize json");
    let mut out = format!("Content-Length: {}\r\n\r\n", body.len()).into_bytes();
    out.extend_from_slice(&body);
    out
}

/// Read one LSP message from `reader`, parsing the `Content-Length` header(s)
/// then exactly that many body bytes. `Ok(None)` on clean EOF.
pub fn read_message<R: BufRead>(reader: &mut R) -> std::io::Result<Option<Value>> {
    let mut content_len: Option<usize> = None;
    loop {
        let mut line = String::new();
        let n = reader.read_line(&mut line)?;
        if n == 0 {
            return Ok(None); // EOF
        }
        let trimmed = line.trim_end_matches(['\r', '\n']);
        if trimmed.is_empty() {
            break; // end of headers
        }
        if let Some(v) = trimmed.strip_prefix("Content-Length:") {
            content_len = v.trim().parse().ok();
        }
    }
    let len = match content_len {
        Some(l) => l,
        None => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "LSP message without Content-Length",
            ))
        }
    };
    let mut body = vec![0u8; len];
    reader.read_exact(&mut body)?;
    let value = serde_json::from_slice(&body)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    Ok(Some(value))
}

// --- message builders ------------------------------------------------------

/// `initialize` request. `root_uri` is a `file://` URI.
pub fn initialize_request(id: i64, root_uri: &str) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "method": "initialize",
        "params": {
            "processId": std::process::id(),
            "rootUri": root_uri,
            "capabilities": {
                "textDocument": {
                    "publishDiagnostics": { "relatedInformation": false },
                    "synchronization": { "didSave": true, "dynamicRegistration": false }
                }
            }
        }
    })
}

pub fn initialized_notification() -> Value {
    json!({"jsonrpc": "2.0", "method": "initialized", "params": {}})
}

pub fn did_open(uri: &str, language_id: &str, version: i64, text: &str) -> Value {
    json!({
        "jsonrpc": "2.0",
        "method": "textDocument/didOpen",
        "params": {
            "textDocument": {
                "uri": uri, "languageId": language_id,
                "version": version, "text": text
            }
        }
    })
}

pub fn did_change(uri: &str, version: i64, text: &str) -> Value {
    // Full-document sync (simplest correct sync kind).
    json!({
        "jsonrpc": "2.0",
        "method": "textDocument/didChange",
        "params": {
            "textDocument": { "uri": uri, "version": version },
            "contentChanges": [ { "text": text } ]
        }
    })
}

/// Parse a `textDocument/publishDiagnostics` notification into `(uri,
/// diagnostics)`. Returns `None` for any other message.
pub fn parse_publish_diagnostics(msg: &Value) -> Option<(String, Vec<Diagnostic>)> {
    if msg.get("method").and_then(Value::as_str) != Some("textDocument/publishDiagnostics") {
        return None;
    }
    let params = msg.get("params")?;
    let uri = params.get("uri").and_then(Value::as_str)?.to_string();
    let mut out = Vec::new();
    for d in params.get("diagnostics").and_then(Value::as_array)?.iter() {
        let range = d.get("range")?;
        let start = range.get("start")?;
        let end = range.get("end")?;
        let sev = d
            .get("severity")
            .and_then(Value::as_u64)
            .map(Severity::from_lsp)
            .unwrap_or(Severity::Information);
        out.push(Diagnostic {
            line: start.get("line").and_then(Value::as_u64).unwrap_or(0) as usize,
            col: start.get("character").and_then(Value::as_u64).unwrap_or(0) as usize,
            end_line: end.get("line").and_then(Value::as_u64).unwrap_or(0) as usize,
            end_col: end.get("character").and_then(Value::as_u64).unwrap_or(0) as usize,
            severity: sev,
            message: d
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string(),
        });
    }
    Some((uri, out))
}

/// Convert a filesystem path to a `file://` URI (minimal; assumes an absolute
/// path with no percent-encoding needs beyond spaces).
pub fn path_to_uri(path: &std::path::Path) -> String {
    let s = path.to_string_lossy().replace(' ', "%20");
    if s.starts_with('/') {
        format!("file://{s}")
    } else {
        format!("file:///{s}")
    }
}

// --- process client --------------------------------------------------------

/// Woken by the reader thread when diagnostics changed.
pub trait LspNotifier: Clone + Send + 'static {
    fn lsp_updated(&self);
}

/// Shared diagnostics store, keyed by document URI.
#[derive(Default)]
pub struct LspState {
    pub diagnostics: Mutex<HashMap<String, Vec<Diagnostic>>>,
}

impl LspState {
    pub fn for_uri(&self, uri: &str) -> Vec<Diagnostic> {
        self.diagnostics
            .lock()
            .unwrap()
            .get(uri)
            .cloned()
            .unwrap_or_default()
    }
}

/// A spawned language server.
pub struct LspClient {
    child: Child,
    stdin: ChildStdin,
    next_id: AtomicI64,
    version: AtomicI64,
    pub state: Arc<LspState>,
    reader: Option<std::thread::JoinHandle<()>>,
}

impl LspClient {
    /// Spawn `program args...`, run the initialize handshake for `root`, and
    /// start the diagnostics reader thread.
    pub fn spawn<N: LspNotifier>(
        program: &str,
        args: &[&str],
        root: &std::path::Path,
        notifier: N,
    ) -> std::io::Result<Self> {
        let mut child = Command::new(program)
            .args(args)
            .current_dir(root)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()?;
        let mut stdin = child.stdin.take().expect("piped stdin");
        let stdout = child.stdout.take().expect("piped stdout");
        let state = Arc::new(LspState::default());

        let reader_state = state.clone();
        let reader = std::thread::spawn(move || {
            let mut buf = std::io::BufReader::new(stdout);
            while let Ok(Some(msg)) = read_message(&mut buf) {
                if let Some((uri, diags)) = parse_publish_diagnostics(&msg) {
                    reader_state.diagnostics.lock().unwrap().insert(uri, diags);
                    notifier.lsp_updated();
                }
            }
        });

        // Handshake.
        let root_uri = path_to_uri(root);
        stdin.write_all(&encode_message(&initialize_request(1, &root_uri)))?;
        stdin.write_all(&encode_message(&initialized_notification()))?;
        stdin.flush()?;

        Ok(Self {
            child,
            stdin,
            next_id: AtomicI64::new(2),
            version: AtomicI64::new(1),
            state,
            reader: Some(reader),
        })
    }

    fn send(&mut self, msg: &Value) -> std::io::Result<()> {
        self.stdin.write_all(&encode_message(msg))?;
        self.stdin.flush()
    }

    pub fn open_document(
        &mut self,
        uri: &str,
        language_id: &str,
        text: &str,
    ) -> std::io::Result<()> {
        let v = self.version.fetch_add(1, Ordering::Relaxed);
        self.send(&did_open(uri, language_id, v, text))
    }

    pub fn change_document(&mut self, uri: &str, text: &str) -> std::io::Result<()> {
        let v = self.version.fetch_add(1, Ordering::Relaxed);
        self.send(&did_change(uri, v, text))
    }

    #[allow(dead_code)]
    fn next_id(&self) -> i64 {
        self.next_id.fetch_add(1, Ordering::Relaxed)
    }

    pub fn shutdown(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        if let Some(r) = self.reader.take() {
            let _ = r.join();
        }
    }
}

impl Drop for LspClient {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn framing_round_trips() {
        let msg = json!({"jsonrpc":"2.0","method":"initialized","params":{}});
        let bytes = encode_message(&msg);
        // Header present and body length correct.
        let text = String::from_utf8_lossy(&bytes);
        assert!(text.starts_with("Content-Length: "));
        assert!(text.contains("\r\n\r\n"));
        let mut reader = &bytes[..];
        let got = read_message(&mut reader).unwrap().unwrap();
        assert_eq!(got, msg);
    }

    #[test]
    fn two_framed_messages_read_in_sequence() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&encode_message(&json!({"a":1})));
        buf.extend_from_slice(&encode_message(&json!({"b":2})));
        let mut reader = &buf[..];
        assert_eq!(read_message(&mut reader).unwrap().unwrap(), json!({"a":1}));
        assert_eq!(read_message(&mut reader).unwrap().unwrap(), json!({"b":2}));
        assert!(read_message(&mut reader).unwrap().is_none()); // EOF
    }

    #[test]
    fn parses_publish_diagnostics() {
        let msg = json!({
            "jsonrpc": "2.0",
            "method": "textDocument/publishDiagnostics",
            "params": {
                "uri": "file:///tmp/x.rs",
                "diagnostics": [{
                    "range": {"start": {"line": 4, "character": 8},
                               "end": {"line": 4, "character": 15}},
                    "severity": 1,
                    "message": "cannot find value `foo`"
                }]
            }
        });
        let (uri, diags) = parse_publish_diagnostics(&msg).unwrap();
        assert_eq!(uri, "file:///tmp/x.rs");
        assert_eq!(diags.len(), 1);
        assert_eq!(diags[0].line, 4);
        assert_eq!(diags[0].col, 8);
        assert_eq!(diags[0].severity, Severity::Error);
        assert_eq!(diags[0].message, "cannot find value `foo`");
    }

    #[test]
    fn non_diagnostic_messages_ignored() {
        assert!(parse_publish_diagnostics(&json!({"method":"other"})).is_none());
        assert!(parse_publish_diagnostics(&json!({"result":{}})).is_none());
    }

    #[test]
    fn path_uri_encoding() {
        assert_eq!(
            path_to_uri(std::path::Path::new("/home/a/x.rs")),
            "file:///home/a/x.rs"
        );
        assert_eq!(
            path_to_uri(std::path::Path::new("/a b/c.rs")),
            "file:///a%20b/c.rs"
        );
    }

    #[test]
    fn builders_have_required_jsonrpc_shape() {
        let init = initialize_request(1, "file:///r");
        assert_eq!(init["method"], "initialize");
        assert_eq!(init["id"], 1);
        assert_eq!(init["params"]["rootUri"], "file:///r");
        let open = did_open("file:///x", "rust", 1, "fn main(){}");
        assert_eq!(open["method"], "textDocument/didOpen");
        assert_eq!(open["params"]["textDocument"]["languageId"], "rust");
    }
}
