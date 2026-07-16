//! `umber.toml` module manifest (docs/PLAN.md "Module manifest" sketch).
//!
//! Parsed by hand with a small section-aware reader, matching umber-kernel's
//! dependency-free flat-TOML philosophy (the kernel's [`Config`] parser is
//! scalar-and-flat; a manifest additionally needs `[section]` awareness and
//! single-line string arrays, so this is a focused extension of the same
//! approach — still no `toml` crate).
//!
//! Supported shape:
//!
//! ```toml
//! [module]
//! name = "agent-dashboard"
//! version = "0.1.0"
//! kind = "wasm"            # wasm | lua
//! entry = "agent_dashboard.wasm"
//! default_on = true
//!
//! [permissions]           # v1: DENY everything; requested perms are stored
//! fs  = ["read:workspace"] # and surfaced, but the broker grants nothing.
//! net = ["localhost"]
//! exec = ["pi"]
//!
//! [ui]
//! panels   = ["agents"]
//! commands = ["agents.dashboard.open"]
//! surfaces = ["tui", "plain"]
//!
//! [config]                # module-declared config keys the ABI can read back
//! greeting = "hello"
//! ```

use std::collections::BTreeMap;
use std::fmt;
use std::path::Path;

/// Author tier / execution backend for a module (D9). The manifest `kind`
/// field selects which host backend loads `entry`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModuleKind {
    /// Rust/TS/AssemblyScript compiled to a core WebAssembly module.
    Wasm,
    /// A Lua script run through mlua.
    Lua,
}

impl ModuleKind {
    /// The manifest spelling (`"wasm"` / `"lua"`).
    pub fn as_str(self) -> &'static str {
        match self {
            ModuleKind::Wasm => "wasm",
            ModuleKind::Lua => "lua",
        }
    }
}

/// Requested capabilities (cfx-style, deny-by-default). v1 stores these for
/// display + the [`crate::PermissionBroker`], but grants nothing — the ABI
/// exposes no I/O, so every value here is a promise the broker refuses to keep
/// yet.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Permissions {
    pub fs: Vec<String>,
    pub net: Vec<String>,
    pub exec: Vec<String>,
}

impl Permissions {
    /// A one-line human summary for the modules page (e.g.
    /// `fs:1 net:1 exec:1`), or `none` when nothing was requested.
    pub fn summary(&self) -> String {
        if self.fs.is_empty() && self.net.is_empty() && self.exec.is_empty() {
            return "no perms".to_string();
        }
        let mut parts = Vec::new();
        if !self.fs.is_empty() {
            parts.push(format!("fs:{}", self.fs.len()));
        }
        if !self.net.is_empty() {
            parts.push(format!("net:{}", self.net.len()));
        }
        if !self.exec.is_empty() {
            parts.push(format!("exec:{}", self.exec.len()));
        }
        parts.join(" ")
    }
}

/// A parsed `umber.toml`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Manifest {
    pub name: String,
    pub version: String,
    pub kind: ModuleKind,
    pub entry: String,
    pub default_on: bool,
    pub permissions: Permissions,
    pub ui_panels: Vec<String>,
    pub ui_commands: Vec<String>,
    pub ui_surfaces: Vec<String>,
    /// Module-declared config keys, readable from the guest via the ABI
    /// `config_get` import (its "own manifest-declared config keys").
    pub config: BTreeMap<String, String>,
}

/// Why a manifest failed to parse or validate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ManifestError {
    /// A required `[module]` field was absent.
    MissingField(&'static str),
    /// `kind` was present but not `wasm`/`lua`.
    UnknownKind(String),
    /// The file could not be read.
    Io(String),
}

impl fmt::Display for ManifestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ManifestError::MissingField(k) => write!(f, "missing required field `{k}`"),
            ManifestError::UnknownKind(k) => {
                write!(f, "unknown module kind `{k}` (expected `wasm` or `lua`)")
            }
            ManifestError::Io(e) => write!(f, "cannot read manifest: {e}"),
        }
    }
}

impl std::error::Error for ManifestError {}

impl Manifest {
    /// Read + parse `umber.toml` at `path`.
    pub fn from_path(path: &Path) -> Result<Self, ManifestError> {
        let text = std::fs::read_to_string(path).map_err(|e| ManifestError::Io(e.to_string()))?;
        Self::parse(&text)
    }

    /// Parse manifest text. Sections are tracked so `name` under `[module]` and
    /// `commands` under `[ui]` never collide. Values are either scalars
    /// (`key = "v"` / `key = true`) or single-line string arrays
    /// (`key = ["a", "b"]`); multi-line arrays are intentionally unsupported in
    /// v1 (the manifest sketch uses only single-line lists).
    pub fn parse(text: &str) -> Result<Self, ManifestError> {
        // (section, key) -> raw value text (right-hand side, trimmed).
        let mut scalars: BTreeMap<(String, String), String> = BTreeMap::new();
        let mut arrays: BTreeMap<(String, String), Vec<String>> = BTreeMap::new();
        let mut section = String::new();

        for raw in text.lines() {
            let line = strip_comment(raw).trim();
            if line.is_empty() {
                continue;
            }
            if let Some(name) = line.strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
                section = name.trim().to_string();
                continue;
            }
            let Some((key, value)) = line.split_once('=') else {
                continue;
            };
            let key = key.trim().to_string();
            let value = value.trim();
            let sect = section.clone();
            if value.starts_with('[') {
                arrays.insert((sect, key), parse_array(value));
            } else {
                scalars.insert((sect, key), unquote(value).to_string());
            }
        }

        let scalar = |sect: &str, key: &str| scalars.get(&(sect.to_string(), key.to_string()));
        let array = |sect: &str, key: &str| {
            arrays
                .get(&(sect.to_string(), key.to_string()))
                .cloned()
                .unwrap_or_default()
        };

        let name = scalar("module", "name")
            .ok_or(ManifestError::MissingField("module.name"))?
            .clone();
        let version = scalar("module", "version")
            .ok_or(ManifestError::MissingField("module.version"))?
            .clone();
        let kind_raw =
            scalar("module", "kind").ok_or(ManifestError::MissingField("module.kind"))?;
        let kind = match kind_raw.as_str() {
            "wasm" => ModuleKind::Wasm,
            "lua" => ModuleKind::Lua,
            other => return Err(ManifestError::UnknownKind(other.to_string())),
        };
        let entry = scalar("module", "entry")
            .ok_or(ManifestError::MissingField("module.entry"))?
            .clone();
        let default_on = scalar("module", "default_on")
            .map(|v| v == "true")
            .unwrap_or(false);

        let permissions = Permissions {
            fs: array("permissions", "fs"),
            net: array("permissions", "net"),
            exec: array("permissions", "exec"),
        };

        // The `[config]` section is a flat scalar table; collect every key.
        let mut config = BTreeMap::new();
        for ((sect, key), value) in &scalars {
            if sect == "config" {
                config.insert(key.clone(), value.clone());
            }
        }

        Ok(Manifest {
            name,
            version,
            kind,
            entry,
            default_on,
            permissions,
            ui_panels: array("ui", "panels"),
            ui_commands: array("ui", "commands"),
            ui_surfaces: array("ui", "surfaces"),
            config,
        })
    }
}

/// Drop an inline `# comment`. Scalars here are unquoted bools/strings and the
/// arrays are bracketed, so a `#` outside of quotes can only start a comment.
/// A `#` inside a double-quoted string is preserved.
fn strip_comment(line: &str) -> &str {
    let mut in_quotes = false;
    for (i, c) in line.char_indices() {
        match c {
            '"' => in_quotes = !in_quotes,
            '#' if !in_quotes => return &line[..i],
            _ => {}
        }
    }
    line
}

/// Strip surrounding whitespace and one layer of double quotes.
fn unquote(value: &str) -> &str {
    let v = value.trim();
    v.strip_prefix('"')
        .and_then(|s| s.strip_suffix('"'))
        .unwrap_or(v)
}

/// Parse a single-line string array `["a", "b"]` into its elements. Tolerant of
/// missing brackets and trailing commas; each element is unquoted + trimmed.
fn parse_array(value: &str) -> Vec<String> {
    let inner = value
        .trim()
        .trim_start_matches('[')
        .trim_end_matches(']')
        .trim();
    if inner.is_empty() {
        return Vec::new();
    }
    inner
        .split(',')
        .map(|s| unquote(s.trim()).to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const GOOD: &str = r#"
        [module]
        name = "hello-panel"
        version = "0.1.0"
        kind = "wasm"
        entry = "hello.wasm"
        default_on = true

        [permissions]
        fs = ["read:workspace", "write:tmp"]
        net = ["localhost"]
        exec = ["pi"]

        [ui]
        panels = ["hello"]
        commands = ["hello.open", "hello.close"]
        surfaces = ["tui", "plain"]

        [config]
        greeting = "hi there"
    "#;

    #[test]
    fn parses_full_manifest() {
        let m = Manifest::parse(GOOD).unwrap();
        assert_eq!(m.name, "hello-panel");
        assert_eq!(m.version, "0.1.0");
        assert_eq!(m.kind, ModuleKind::Wasm);
        assert_eq!(m.entry, "hello.wasm");
        assert!(m.default_on);
        assert_eq!(m.permissions.fs, vec!["read:workspace", "write:tmp"]);
        assert_eq!(m.permissions.net, vec!["localhost"]);
        assert_eq!(m.permissions.exec, vec!["pi"]);
        assert_eq!(m.ui_commands, vec!["hello.open", "hello.close"]);
        assert_eq!(m.ui_surfaces, vec!["tui", "plain"]);
        assert_eq!(
            m.config.get("greeting").map(String::as_str),
            Some("hi there")
        );
    }

    #[test]
    fn missing_required_field_is_error() {
        let text = "[module]\nname = \"x\"\nkind = \"lua\"\nentry = \"x.lua\"\n";
        assert_eq!(
            Manifest::parse(text),
            Err(ManifestError::MissingField("module.version"))
        );
    }

    #[test]
    fn unknown_kind_is_error() {
        let text = "[module]\nname=\"x\"\nversion=\"1\"\nkind=\"python\"\nentry=\"x.py\"\n";
        assert_eq!(
            Manifest::parse(text),
            Err(ManifestError::UnknownKind("python".to_string()))
        );
    }

    #[test]
    fn empty_permission_lists_default_empty() {
        let text = "[module]\nname=\"x\"\nversion=\"1\"\nkind=\"lua\"\nentry=\"x.lua\"\n";
        let m = Manifest::parse(text).unwrap();
        assert!(m.permissions.fs.is_empty());
        assert_eq!(m.permissions.summary(), "no perms");
        assert!(!m.default_on);
    }

    #[test]
    fn comment_inside_quotes_is_kept() {
        let text =
            "[module]\nname=\"a#b\"\nversion=\"1\"\nkind=\"lua\"\nentry=\"x.lua\" # trailing\n";
        let m = Manifest::parse(text).unwrap();
        assert_eq!(m.name, "a#b");
        assert_eq!(m.entry, "x.lua");
    }
}
