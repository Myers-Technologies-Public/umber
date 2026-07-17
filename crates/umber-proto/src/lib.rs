//! umber-proto — workspace protocol types (core <-> backend).
//!
//! The editor core never touches the filesystem directly (Rule 1 / D7): it
//! speaks this protocol to a workspace backend — in-process for local
//! projects, `umberd` over SSH for remote ones. Same request/response types
//! on both sides, so remote development is a transport swap, not a rewrite.
//!
//! Framing is JSONL: one JSON object per line, LF-delimited, matching pi's RPC
//! discipline (split on `\n` only). [`read_message`] / [`write_message`] are
//! the wire helpers used by both `umberd` and the editor's client.

use std::io::{BufRead, Write};

use serde::{Deserialize, Serialize};

/// Protocol version, bumped on any wire-incompatible change. The client sends
/// it in [`Request::Hello`]; the daemon rejects a mismatch.
pub const PROTOCOL_VERSION: u32 = 1;

/// A request from the editor to the workspace backend.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Request {
    /// Handshake: client protocol version. Must be first.
    Hello { version: u32 },
    /// Liveness probe.
    Ping,
    /// List a directory (non-recursive).
    ListDir { path: String },
    /// Read a UTF-8 file's contents.
    ReadFile { path: String },
    /// Write (create/overwrite) a UTF-8 file.
    WriteFile { path: String, contents: String },
    /// Metadata for a path.
    Stat { path: String },
    /// Graceful shutdown of the daemon.
    Shutdown,
}

/// A response from the workspace backend.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "ok", rename_all = "snake_case")]
pub enum Response {
    /// Handshake accepted; daemon reports its version + cwd.
    Hello {
        version: u32,
        cwd: String,
    },
    Pong,
    Dir {
        entries: Vec<DirEntry>,
    },
    File {
        contents: String,
    },
    Written {
        bytes: usize,
    },
    Stat(StatInfo),
    /// Operation failed (message is human-readable; `kind` is machine-usable).
    Error {
        kind: ErrorKind,
        message: String,
    },
}

/// One directory entry.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DirEntry {
    pub name: String,
    pub is_dir: bool,
    pub size: u64,
}

/// Path metadata.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StatInfo {
    pub is_dir: bool,
    pub size: u64,
    /// Whether the path exists at all (false = the rest is meaningless).
    pub exists: bool,
}

/// Coarse error classification the client can branch on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorKind {
    NotFound,
    PermissionDenied,
    NotUtf8,
    VersionMismatch,
    Protocol,
    Io,
}

/// Read one LF-delimited JSON message from `reader`. `Ok(None)` on clean EOF.
/// Splits on `\n` only (via `read_until(b'\n')`); a trailing `\r` is trimmed
/// by serde's whitespace tolerance.
pub fn read_message<R: BufRead, T: for<'de> Deserialize<'de>>(
    reader: &mut R,
) -> std::io::Result<Option<T>> {
    let mut buf = Vec::new();
    let n = reader.read_until(b'\n', &mut buf)?;
    if n == 0 {
        return Ok(None);
    }
    let value = serde_json::from_slice(&buf)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    Ok(Some(value))
}

/// Write one message as a single LF-terminated JSON line and flush.
pub fn write_message<W: Write, T: Serialize>(writer: &mut W, msg: &T) -> std::io::Result<()> {
    let line = serde_json::to_string(msg)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    writer.write_all(line.as_bytes())?;
    writer.write_all(b"\n")?;
    writer.flush()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_round_trips_through_jsonl() {
        let mut buf: Vec<u8> = Vec::new();
        let req = Request::WriteFile {
            path: "/tmp/x".to_string(),
            contents: "line1\nline2".to_string(),
        };
        write_message(&mut buf, &req).unwrap();
        // Exactly one line (embedded newline is JSON-escaped, not a framing
        // break).
        assert_eq!(buf.iter().filter(|&&b| b == b'\n').count(), 1);
        let mut reader = &buf[..];
        let got: Request = read_message(&mut reader).unwrap().unwrap();
        assert_eq!(got, req);
    }

    #[test]
    fn response_tag_shape_is_stable() {
        let r = Response::Error {
            kind: ErrorKind::NotFound,
            message: "nope".to_string(),
        };
        let s = serde_json::to_string(&r).unwrap();
        assert!(s.contains(r#""ok":"error""#));
        assert!(s.contains(r#""kind":"not_found""#));
    }

    #[test]
    fn clean_eof_returns_none() {
        let empty: &[u8] = b"";
        let mut reader = empty;
        let got: std::io::Result<Option<Request>> = read_message(&mut reader);
        assert!(got.unwrap().is_none());
    }

    #[test]
    fn two_messages_read_in_sequence() {
        let mut buf: Vec<u8> = Vec::new();
        write_message(&mut buf, &Request::Ping).unwrap();
        write_message(
            &mut buf,
            &Request::ReadFile {
                path: "a".to_string(),
            },
        )
        .unwrap();
        let mut reader = &buf[..];
        let a: Request = read_message(&mut reader).unwrap().unwrap();
        let b: Request = read_message(&mut reader).unwrap().unwrap();
        assert_eq!(a, Request::Ping);
        assert_eq!(
            b,
            Request::ReadFile {
                path: "a".to_string()
            }
        );
    }
}
