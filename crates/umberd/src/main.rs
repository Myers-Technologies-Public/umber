//! umberd — headless workspace daemon (P3b): serves the umber-proto workspace
//! protocol over stdio, so the editor edits remote files by running
//! `ssh <host> umberd` and speaking the same protocol it uses locally
//! (Rule 1 / D7 — remote dev is a transport swap, not a rewrite).
//!
//! Transport: JSONL requests on stdin, JSONL responses on stdout, one response
//! per request (`umber_proto::read_message`/`write_message`). The daemon is
//! single-threaded and synchronous: a remote editor session is one PTY-free
//! request/response loop. PTYs/search/LSP are later slices.
//!
//! Path confinement: every path is resolved against a root (the process cwd,
//! or `$UMBERD_ROOT`) and rejected if it escapes — a remote daemon must not
//! serve arbitrary host paths just because the client asked.

use std::io::{self, BufReader};
use std::path::{Component, Path, PathBuf};

use umber_proto::{
    read_message, write_message, DirEntry, ErrorKind, Request, Response, StatInfo, PROTOCOL_VERSION,
};

fn main() {
    let root = std::env::var_os("UMBERD_ROOT")
        .map(PathBuf::from)
        .or_else(|| std::env::current_dir().ok())
        .unwrap_or_else(|| PathBuf::from("."));
    let root = root.canonicalize().unwrap_or(root);

    let stdin = io::stdin();
    let mut reader = BufReader::new(stdin.lock());
    let stdout = io::stdout();
    let mut writer = stdout.lock();

    loop {
        let req: Request = match read_message(&mut reader) {
            Ok(Some(r)) => r,
            Ok(None) => break, // clean EOF: client closed the pipe
            Err(err) => {
                let _ = write_message(
                    &mut writer,
                    &Response::Error {
                        kind: ErrorKind::Protocol,
                        message: format!("malformed request: {err}"),
                    },
                );
                continue;
            }
        };
        if matches!(req, Request::Shutdown) {
            break;
        }
        let resp = handle(&root, req);
        if write_message(&mut writer, &resp).is_err() {
            break; // client gone
        }
    }
}

fn handle(root: &Path, req: Request) -> Response {
    match req {
        Request::Hello { version } => {
            if version != PROTOCOL_VERSION {
                Response::Error {
                    kind: ErrorKind::VersionMismatch,
                    message: format!("client protocol v{version}, daemon v{PROTOCOL_VERSION}"),
                }
            } else {
                Response::Hello {
                    version: PROTOCOL_VERSION,
                    cwd: root.display().to_string(),
                }
            }
        }
        Request::Ping => Response::Pong,
        Request::Shutdown => Response::Pong, // handled in the loop; unreachable
        Request::ListDir { path } => match confine(root, &path) {
            Ok(abs) => list_dir(&abs),
            Err(e) => e,
        },
        Request::ReadFile { path } => match confine(root, &path) {
            Ok(abs) => read_file(&abs),
            Err(e) => e,
        },
        Request::WriteFile { path, contents } => match confine(root, &path) {
            Ok(abs) => write_file(&abs, &contents),
            Err(e) => e,
        },
        Request::Stat { path } => match confine(root, &path) {
            Ok(abs) => stat(&abs),
            Err(e) => e,
        },
    }
}

/// Resolve `path` under `root` and reject any escape (`..` climbing out,
/// absolute paths outside root). Returns the confined absolute path or an
/// `Error` response. Purely lexical so it works for not-yet-existing files
/// (WriteFile) — we never trust a client path.
fn confine(root: &Path, path: &str) -> Result<PathBuf, Response> {
    let requested = Path::new(path);
    let joined = if requested.is_absolute() {
        requested.to_path_buf()
    } else {
        root.join(requested)
    };
    // Lexically normalize (no fs access; handles nonexistent targets).
    let mut normalized = PathBuf::new();
    for comp in joined.components() {
        match comp {
            Component::ParentDir => {
                if !normalized.pop() {
                    return Err(escape_error(path));
                }
            }
            Component::CurDir => {}
            other => normalized.push(other.as_os_str()),
        }
    }
    if normalized.starts_with(root) {
        Ok(normalized)
    } else {
        Err(escape_error(path))
    }
}

fn escape_error(path: &str) -> Response {
    Response::Error {
        kind: ErrorKind::PermissionDenied,
        message: format!("path escapes the workspace root: {path}"),
    }
}

fn io_error(err: &io::Error) -> Response {
    let kind = match err.kind() {
        io::ErrorKind::NotFound => ErrorKind::NotFound,
        io::ErrorKind::PermissionDenied => ErrorKind::PermissionDenied,
        _ => ErrorKind::Io,
    };
    Response::Error {
        kind,
        message: err.to_string(),
    }
}

fn list_dir(abs: &Path) -> Response {
    let iter = match std::fs::read_dir(abs) {
        Ok(i) => i,
        Err(e) => return io_error(&e),
    };
    let mut entries = Vec::new();
    for e in iter.flatten() {
        let meta = e.metadata();
        let (is_dir, size) = meta.map(|m| (m.is_dir(), m.len())).unwrap_or((false, 0));
        entries.push(DirEntry {
            name: e.file_name().to_string_lossy().into_owned(),
            is_dir,
            size,
        });
    }
    entries.sort_by(|a, b| b.is_dir.cmp(&a.is_dir).then(a.name.cmp(&b.name)));
    Response::Dir { entries }
}

fn read_file(abs: &Path) -> Response {
    match std::fs::read(abs) {
        Ok(bytes) => match String::from_utf8(bytes) {
            Ok(contents) => Response::File { contents },
            Err(_) => Response::Error {
                kind: ErrorKind::NotUtf8,
                message: "file is not valid UTF-8".to_string(),
            },
        },
        Err(e) => io_error(&e),
    }
}

fn write_file(abs: &Path, contents: &str) -> Response {
    if let Some(parent) = abs.parent() {
        if let Err(e) = std::fs::create_dir_all(parent) {
            return io_error(&e);
        }
    }
    match std::fs::write(abs, contents.as_bytes()) {
        Ok(()) => Response::Written {
            bytes: contents.len(),
        },
        Err(e) => io_error(&e),
    }
}

fn stat(abs: &Path) -> Response {
    match std::fs::metadata(abs) {
        Ok(m) => Response::Stat(StatInfo {
            is_dir: m.is_dir(),
            size: m.len(),
            exists: true,
        }),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Response::Stat(StatInfo {
            is_dir: false,
            size: 0,
            exists: false,
        }),
        Err(e) => io_error(&e),
    }
}
