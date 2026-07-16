//! umber-kernel — the unremovable kernel boundary (D10): command registry,
//! command palette matching, TOML config (D13), and the feature/module
//! registry (D10 proto-modules).
//!
//! P-phase: **P1**. This crate is UI-free and filesystem-light: it owns the
//! *data* (commands, config values, feature flags) and the pure logic
//! (fuzzy matching, TOML load/save, range clamping). The `umber` bin drives
//! the actual palette/settings/modules UI through umber-ui's render
//! primitives and applies config changes live.
//!
//! Nothing here pulls an external dependency: the config format is a flat TOML
//! table (scalars only), parsed and written by hand so the kernel stays a
//! zero-dep leaf crate.

use std::path::PathBuf;

// ===========================================================================
// Commands (D6 command registry + palette source)
// ===========================================================================

/// A single invokable action. `id` is the stable machine name the bin
/// dispatches on (e.g. `"file.save"`); `title` is the human label shown in the
/// palette (e.g. `"File: Save"`); `keybinding` is a display-only chord label
/// (e.g. `"Ctrl+S"`), empty when the action has no default chord.
#[derive(Clone, Copy, Debug)]
pub struct Command {
    pub id: &'static str,
    pub title: &'static str,
    pub keybinding: &'static str,
}

/// The command registry (D6). The bin registers every action at startup; the
/// palette filters `title`s with a subsequence fuzzy match.
#[derive(Default)]
pub struct CommandRegistry {
    commands: Vec<Command>,
}

impl CommandRegistry {
    pub fn new() -> Self {
        Self {
            commands: Vec::new(),
        }
    }

    /// Register one command (order is preserved and used as the tie-break for
    /// equal fuzzy scores, and as the full listing when the query is empty).
    pub fn register(&mut self, command: Command) {
        self.commands.push(command);
    }

    /// All registered commands in registration order.
    pub fn commands(&self) -> &[Command] {
        &self.commands
    }

    /// Look a command up by `id`.
    pub fn find(&self, id: &str) -> Option<Command> {
        self.commands.iter().copied().find(|c| c.id == id)
    }

    /// Filter+rank commands against `query`. Returns indices into
    /// [`CommandRegistry::commands`], best match first. An empty/whitespace
    /// query returns every command in registration order. A non-empty query
    /// keeps only commands whose `title` contains the query as a
    /// case-insensitive character subsequence, ranked by [`fuzzy_score`] (ties
    /// broken by registration order for stability).
    pub fn filter(&self, query: &str) -> Vec<usize> {
        if query.trim().is_empty() {
            return (0..self.commands.len()).collect();
        }
        let mut scored: Vec<(usize, i32)> = self
            .commands
            .iter()
            .enumerate()
            .filter_map(|(i, c)| fuzzy_score(c.title, query).map(|s| (i, s)))
            .collect();
        scored.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        scored.into_iter().map(|(i, _)| i).collect()
    }
}

/// Score how well `needle` fuzzy-matches `haystack` as a case-insensitive
/// character subsequence, or `None` when it is not a subsequence at all.
/// Higher is better. Bonuses: matching the first char, matching just after a
/// word boundary (space, `:`, `.`, `_`, `-`), and each additional char in a
/// contiguous run. This is a deliberately small, dependency-free scorer — the
/// palette is the flagship interaction (D6), so it must be instant, not
/// clever.
pub fn fuzzy_score(haystack: &str, needle: &str) -> Option<i32> {
    let need: Vec<char> = needle
        .chars()
        .flat_map(|c| c.to_lowercase())
        .filter(|c| !c.is_whitespace())
        .collect();
    if need.is_empty() {
        return Some(0);
    }
    let hay: Vec<char> = haystack.chars().flat_map(|c| c.to_lowercase()).collect();

    let mut hi = 0usize;
    let mut score = 0i32;
    let mut prev_matched = false;
    let mut run = 0i32;
    for &nc in &need {
        let mut found = false;
        while hi < hay.len() {
            let hc = hay[hi];
            hi += 1;
            if hc == nc {
                score += 1;
                if hi == 1 {
                    score += 6; // matched the very first char
                } else {
                    let prev = hay[hi - 2];
                    if matches!(prev, ' ' | ':' | '.' | '_' | '-' | '/') {
                        score += 4; // matched at a word boundary
                    }
                }
                if prev_matched {
                    run += 1;
                    score += run * 2; // contiguous run bonus
                } else {
                    run = 0;
                }
                prev_matched = true;
                found = true;
                break;
            } else {
                prev_matched = false;
            }
        }
        if !found {
            return None;
        }
    }
    Some(score)
}

// ===========================================================================
// Config (D13 TOML)
// ===========================================================================

/// Font size bounds (logical px) for the settings page numeric control.
pub const FONT_MIN: f32 = 6.0;
pub const FONT_MAX: f32 = 32.0;
/// Line-height bounds (logical px).
pub const LINE_MIN: f32 = 10.0;
pub const LINE_MAX: f32 = 48.0;
/// Scrollbar auto-hide linger bounds (ms).
pub const LINGER_MIN: u64 = 100;
pub const LINGER_MAX: u64 = 5000;

/// Persisted user configuration (D13). Loaded once at startup; a missing file
/// yields [`Config::default`]. Every field is live-appliable by the bin.
#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    /// Body-text font size in logical px (feeds the renderer's metrics).
    pub font_size: f32,
    /// Line height in logical px.
    pub line_height: f32,
    /// Overlay scrollbar auto-hide linger, milliseconds.
    pub scrollbar_linger_ms: u64,
    /// Draw the line-number gutter.
    pub gutter: bool,
    /// Enable the overlay scrollbar at all.
    pub scrollbar: bool,
    /// Show the keystroke->present latency segment in the banner.
    pub latency_hud: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            font_size: 14.0,
            line_height: 20.0,
            scrollbar_linger_ms: 800,
            gutter: true,
            scrollbar: true,
            latency_hud: true,
        }
    }
}

impl Config {
    /// `$XDG_CONFIG_HOME/umber/config.toml`, falling back to
    /// `$HOME/.config/umber/config.toml`. `None` only if neither var is set.
    pub fn path() -> Option<PathBuf> {
        if let Some(xdg) = std::env::var_os("XDG_CONFIG_HOME") {
            if !xdg.is_empty() {
                return Some(PathBuf::from(xdg).join("umber").join("config.toml"));
            }
        }
        let home = std::env::var_os("HOME")?;
        Some(
            PathBuf::from(home)
                .join(".config")
                .join("umber")
                .join("config.toml"),
        )
    }

    /// Load config from [`Config::path`]. A missing/unreadable file returns
    /// defaults; recognised keys override defaults; unknown keys are ignored.
    /// The result is range-clamped so a hand-edited file can never push the
    /// renderer out of a sane state.
    pub fn load() -> Self {
        let mut cfg = Self::default();
        if let Some(path) = Self::path() {
            if let Ok(text) = std::fs::read_to_string(&path) {
                cfg.apply_toml(&text);
            }
        }
        cfg.clamp();
        cfg
    }

    /// Parse a flat TOML table (`key = value` lines, `#` comments, `[section]`
    /// headers ignored). Only scalar bool/number values are understood.
    fn apply_toml(&mut self, text: &str) {
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') || line.starts_with('[') {
                continue;
            }
            let Some((key, value)) = line.split_once('=') else {
                continue;
            };
            let key = key.trim();
            // Strip an inline `# comment` before unquoting: values here are
            // scalar bools/numbers, so a `#` can only begin a comment.
            let value = match value.find('#') {
                Some(i) => &value[..i],
                None => value,
            };
            let value = value.trim().trim_matches('"');
            match key {
                "font_size" => {
                    if let Ok(v) = value.parse() {
                        self.font_size = v;
                    }
                }
                "line_height" => {
                    if let Ok(v) = value.parse() {
                        self.line_height = v;
                    }
                }
                "scrollbar_linger_ms" => {
                    if let Ok(v) = value.parse() {
                        self.scrollbar_linger_ms = v;
                    }
                }
                "gutter" => {
                    if let Some(v) = parse_bool(value) {
                        self.gutter = v;
                    }
                }
                "scrollbar" => {
                    if let Some(v) = parse_bool(value) {
                        self.scrollbar = v;
                    }
                }
                "latency_hud" => {
                    if let Some(v) = parse_bool(value) {
                        self.latency_hud = v;
                    }
                }
                _ => {}
            }
        }
    }

    /// Serialise to the flat TOML table [`Config::apply_toml`] reads back.
    pub fn to_toml(&self) -> String {
        format!(
            "# Umber configuration (D13). Edited live via the settings page\n\
             # (Ctrl+, or \"Preferences: Open Settings\").\n\
             font_size = {}\n\
             line_height = {}\n\
             scrollbar_linger_ms = {}\n\
             gutter = {}\n\
             scrollbar = {}\n\
             latency_hud = {}\n",
            self.font_size,
            self.line_height,
            self.scrollbar_linger_ms,
            self.gutter,
            self.scrollbar,
            self.latency_hud,
        )
    }

    /// Write the config to [`Config::path`], creating the parent directory.
    pub fn save(&self) -> std::io::Result<()> {
        let path = Self::path().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "no config directory ($XDG_CONFIG_HOME/$HOME unset)",
            )
        })?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&path, self.to_toml())
    }

    /// Clamp every field into its documented range.
    pub fn clamp(&mut self) {
        self.font_size = self.font_size.clamp(FONT_MIN, FONT_MAX);
        self.line_height = self.line_height.clamp(LINE_MIN, LINE_MAX);
        self.scrollbar_linger_ms = self.scrollbar_linger_ms.clamp(LINGER_MIN, LINGER_MAX);
    }
}

fn parse_bool(s: &str) -> Option<bool> {
    match s {
        "true" => Some(true),
        "false" => Some(false),
        _ => None,
    }
}

// ===========================================================================
// Features (D10 proto-modules)
// ===========================================================================

/// One entry in the module/feature registry (D10). `removable == false` marks a
/// kernel entry (render loop, config, keybind engine, command palette) that the
/// modules page refuses to disable.
#[derive(Clone, Debug)]
pub struct Feature {
    pub id: &'static str,
    pub name: &'static str,
    pub description: &'static str,
    pub default_on: bool,
    pub enabled: bool,
    pub removable: bool,
}

/// The feature registry (D10). Toggleable features (gutter, scrollbar,
/// latency-hud) mirror [`Config`] booleans; kernel entries are always on.
pub struct FeatureRegistry {
    features: Vec<Feature>,
}

impl FeatureRegistry {
    /// Build the registry, seeding each toggleable feature's `enabled` state
    /// from `cfg`. Kernel entries (command-palette, keybinds) are fixed on and
    /// `removable = false` per D10.
    pub fn from_config(cfg: &Config) -> Self {
        Self {
            features: vec![
                Feature {
                    id: "gutter",
                    name: "Line-number gutter",
                    description: "Line numbers down the left edge",
                    default_on: true,
                    enabled: cfg.gutter,
                    removable: true,
                },
                Feature {
                    id: "scrollbar",
                    name: "Overlay scrollbar",
                    description: "Ghostty-style auto-hiding scrollbar",
                    default_on: true,
                    enabled: cfg.scrollbar,
                    removable: true,
                },
                Feature {
                    id: "latency-hud",
                    name: "Latency HUD",
                    description: "Keystroke->present p50/p99 in the banner",
                    default_on: true,
                    enabled: cfg.latency_hud,
                    removable: true,
                },
                Feature {
                    id: "command-palette",
                    name: "Command palette",
                    description: "Ctrl+Shift+P fuzzy command launcher",
                    default_on: true,
                    enabled: true,
                    removable: false,
                },
                Feature {
                    id: "keybinds",
                    name: "Keybind engine",
                    description: "Chord keymap dispatch (D6)",
                    default_on: true,
                    enabled: true,
                    removable: false,
                },
            ],
        }
    }

    pub fn features(&self) -> &[Feature] {
        &self.features
    }

    /// Index of the feature with `id`, if present.
    pub fn index_of(&self, id: &str) -> Option<usize> {
        self.features.iter().position(|f| f.id == id)
    }

    /// Whether the feature with `id` is currently enabled (absent = `false`).
    pub fn is_enabled(&self, id: &str) -> bool {
        self.features
            .iter()
            .find(|f| f.id == id)
            .map(|f| f.enabled)
            .unwrap_or(false)
    }

    /// Toggle the feature at `idx`. Returns the new `enabled` state, or an
    /// `Err` hint when the entry is a non-removable kernel feature (D10).
    pub fn toggle(&mut self, idx: usize) -> Result<bool, &'static str> {
        let f = self.features.get_mut(idx).ok_or("no such feature")?;
        if !f.removable {
            return Err("kernel module \u{2014} cannot be disabled (D10)");
        }
        f.enabled = !f.enabled;
        Ok(f.enabled)
    }

    /// Push the toggleable features' `enabled` state back into `cfg` so a save
    /// persists them.
    pub fn apply_to_config(&self, cfg: &mut Config) {
        cfg.gutter = self.is_enabled("gutter");
        cfg.scrollbar = self.is_enabled("scrollbar");
        cfg.latency_hud = self.is_enabled("latency-hud");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fuzzy_matches_subsequence_and_ranks_prefix_first() {
        assert!(fuzzy_score("File: Save", "fs").is_some());
        assert!(fuzzy_score("File: Save", "save").is_some());
        assert!(fuzzy_score("File: Save", "zzz").is_none());
        // A prefix/word-boundary match should outscore a scattered one.
        let prefix = fuzzy_score("Save File", "save").unwrap();
        let scattered = fuzzy_score("Some Awful Verbose Example", "save").unwrap();
        assert!(prefix > scattered);
    }

    #[test]
    fn empty_query_lists_all_in_order() {
        let mut reg = CommandRegistry::new();
        reg.register(Command {
            id: "a",
            title: "Alpha",
            keybinding: "",
        });
        reg.register(Command {
            id: "b",
            title: "Beta",
            keybinding: "",
        });
        assert_eq!(reg.filter(""), vec![0, 1]);
        assert_eq!(reg.filter("  "), vec![0, 1]);
    }

    #[test]
    fn config_toml_roundtrips() {
        let mut cfg = Config::default();
        cfg.font_size = 18.0;
        cfg.gutter = false;
        cfg.scrollbar_linger_ms = 1200;
        let mut back = Config::default();
        back.apply_toml(&cfg.to_toml());
        back.clamp();
        assert_eq!(cfg, back);
    }

    #[test]
    fn config_clamps_out_of_range() {
        let mut cfg = Config::default();
        cfg.font_size = 999.0;
        cfg.scrollbar_linger_ms = 10;
        cfg.clamp();
        assert_eq!(cfg.font_size, FONT_MAX);
        assert_eq!(cfg.scrollbar_linger_ms, LINGER_MIN);
    }

    #[test]
    fn kernel_features_refuse_toggle() {
        let mut reg = FeatureRegistry::from_config(&Config::default());
        let palette = reg.index_of("command-palette").unwrap();
        assert!(reg.toggle(palette).is_err());
        let gutter = reg.index_of("gutter").unwrap();
        assert_eq!(reg.toggle(gutter), Ok(false));
        assert!(!reg.is_enabled("gutter"));
    }
}
