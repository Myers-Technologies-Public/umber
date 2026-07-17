//! Remote workspace client (P3b): the editor's end of the umber-proto
//! boundary. Spawns a transport subprocess — `ssh <host> umberd` for remote
//! work, or the `umberd` binary directly for local/testing — and speaks the
//! same request/response protocol the local backend uses (Rule 1 / D7).
//!
//! Synchronous request/response over the child's stdio: the editor sends one
//! `Request` and blocks for its `Response`. That is fine for open/save/list at
//! P3b; a streaming/multiplexed transport is a later concern.

use std::io::{BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};

use umber_proto::{
    read_message, write_message, DirEntry, Request, Response, StatInfo, PROTOCOL_VERSION,
};

/// A connected remote (or local-subprocess) workspace.
pub struct RemoteWorkspace {
    child: Child,
    stdin: ChildStdin,
    reader: BufReader<std::process::ChildStdout>,
    /// The daemon's reported root, for display.
    pub cwd: String,
    /// Human label for the connection (e.g. the ssh host).
    pub label: String,
}

impl RemoteWorkspace {
    /// Connect over `ssh <host> umberd`. `umberd` must be on the remote PATH.
    pub fn connect_ssh(host: &str) -> Result<Self, String> {
        Self::connect_command(
            Command::new("ssh").arg(host).arg("umberd"),
            format!("ssh:{host}"),
        )
    }

    /// Connect to a local `umberd` at `bin` rooted at `root` (used by tests and
    /// a local-backend fallback).
    pub fn connect_local(bin: &std::path::Path, root: &std::path::Path) -> Result<Self, String> {
        Self::connect_command(
            Command::new(bin).env("UMBERD_ROOT", root),
            format!("local:{}", root.display()),
        )
    }

    fn connect_command(cmd: &mut Command, label: String) -> Result<Self, String> {
        let mut child = cmd
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| format!("spawn transport: {e}"))?;
        let stdin = child.stdin.take().expect("piped stdin");
        let stdout = child.stdout.take().expect("piped stdout");
        let mut ws = Self {
            child,
            stdin,
            reader: BufReader::new(stdout),
            cwd: String::new(),
            label,
        };
        // Handshake: version must match or the daemon errors.
        match ws.request(&Request::Hello {
            version: PROTOCOL_VERSION,
        })? {
            Response::Hello { cwd, .. } => {
                ws.cwd = cwd;
                Ok(ws)
            }
            Response::Error { message, .. } => Err(format!("handshake rejected: {message}")),
            other => Err(format!("unexpected handshake response: {other:?}")),
        }
    }

    /// Send one request, block for its response.
    fn request(&mut self, req: &Request) -> Result<Response, String> {
        write_message(&mut self.stdin, req).map_err(|e| format!("send: {e}"))?;
        self.stdin.flush().map_err(|e| format!("flush: {e}"))?;
        match read_message(&mut self.reader).map_err(|e| format!("recv: {e}"))? {
            Some(resp) => Ok(resp),
            None => Err("daemon closed the connection".to_string()),
        }
    }

    /// Read a remote UTF-8 file.
    pub fn read_file(&mut self, path: &str) -> Result<String, String> {
        match self.request(&Request::ReadFile {
            path: path.to_string(),
        })? {
            Response::File { contents } => Ok(contents),
            Response::Error { message, .. } => Err(message),
            other => Err(format!("unexpected: {other:?}")),
        }
    }

    /// Write a remote UTF-8 file.
    pub fn write_file(&mut self, path: &str, contents: &str) -> Result<usize, String> {
        match self.request(&Request::WriteFile {
            path: path.to_string(),
            contents: contents.to_string(),
        })? {
            Response::Written { bytes } => Ok(bytes),
            Response::Error { message, .. } => Err(message),
            other => Err(format!("unexpected: {other:?}")),
        }
    }

    /// List a remote directory.
    pub fn list_dir(&mut self, path: &str) -> Result<Vec<DirEntry>, String> {
        match self.request(&Request::ListDir {
            path: path.to_string(),
        })? {
            Response::Dir { entries } => Ok(entries),
            Response::Error { message, .. } => Err(message),
            other => Err(format!("unexpected: {other:?}")),
        }
    }

    /// Stat a remote path.
    pub fn stat(&mut self, path: &str) -> Result<StatInfo, String> {
        match self.request(&Request::Stat {
            path: path.to_string(),
        })? {
            Response::Stat(info) => Ok(info),
            Response::Error { message, .. } => Err(message),
            other => Err(format!("unexpected: {other:?}")),
        }
    }

    /// Ask the daemon to shut down, then reap the transport.
    pub fn shutdown(&mut self) {
        let _ = write_message(&mut self.stdin, &Request::Shutdown);
        let _ = self.stdin.flush();
        let _ = self.child.wait();
    }
}

impl Drop for RemoteWorkspace {
    fn drop(&mut self) {
        self.shutdown();
    }
}
