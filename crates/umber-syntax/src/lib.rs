//! umber-syntax — tree-sitter incremental parse + highlight (P1 crate,
//! now real). Produces flat, non-overlapping byte-range spans with a small
//! style enum; umber-ui maps styles to theme colors and feeds cosmic-text
//! rich spans.
//!
//! Scope: the editor highlights the *visible window* text per change (the
//! same window it shapes), so parses are a few hundred lines — sub-ms. Whole-
//! file incremental parsing is a later upgrade; block constructs that start
//! above the window may mis-highlight until scrolled (known trade-off).

use std::collections::HashMap;

use tree_sitter_highlight::{HighlightConfiguration, HighlightEvent, Highlighter};

/// Languages with bundled grammars.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Lang {
    Rust,
    Toml,
    Json,
}

/// Map a file extension (lowercase, no dot) to a bundled language.
/// `lock` maps to TOML (Cargo.lock).
pub fn lang_for_ext(ext: &str) -> Option<Lang> {
    match ext {
        "rs" => Some(Lang::Rust),
        "toml" | "lock" => Some(Lang::Toml),
        "json" => Some(Lang::Json),
        _ => None,
    }
}

/// Highlight style, mapped to theme colors by the renderer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Style {
    Keyword,
    Function,
    Type,
    String,
    Number,
    Comment,
    Property,
    Punct,
}

/// One highlighted byte range (non-overlapping, ascending).
#[derive(Clone, Debug, PartialEq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
    pub style: Style,
}

/// Capture names we recognize, in tree-sitter-highlight's prefix-matching
/// scheme. Order defines the index -> style mapping below.
const CAPTURES: &[&str] = &[
    "keyword",
    "function",
    "constructor",
    "type",
    "attribute",
    "string",
    "escape",
    "number",
    "constant",
    "comment",
    "property",
    "punctuation",
    "operator",
    "variable.builtin",
];

fn style_for_capture(idx: usize) -> Style {
    match CAPTURES.get(idx).copied().unwrap_or("") {
        "keyword" | "variable.builtin" => Style::Keyword,
        "function" | "constructor" => Style::Function,
        "type" | "attribute" => Style::Type,
        "string" | "escape" => Style::String,
        "number" | "constant" => Style::Number,
        "comment" => Style::Comment,
        "property" => Style::Property,
        _ => Style::Punct,
    }
}

/// Owns the per-language configurations + a reusable highlighter.
pub struct SyntaxSet {
    highlighter: Highlighter,
    configs: HashMap<Lang, HighlightConfiguration>,
}

impl Default for SyntaxSet {
    fn default() -> Self {
        Self::new()
    }
}

impl SyntaxSet {
    /// Build the bundled language set. Grammar/query failures are impossible
    /// with vendored grammars barring version skew, but degrade to an empty
    /// config map (no highlighting) rather than panicking.
    pub fn new() -> Self {
        let mut configs = HashMap::new();
        let mut add = |lang: Lang, ts: tree_sitter::Language, name: &str, query: &str| {
            if let Ok(mut cfg) = HighlightConfiguration::new(ts, name, query, "", "") {
                cfg.configure(CAPTURES);
                configs.insert(lang, cfg);
            }
        };
        add(
            Lang::Rust,
            tree_sitter_rust::LANGUAGE.into(),
            "rust",
            tree_sitter_rust::HIGHLIGHTS_QUERY,
        );
        add(
            Lang::Toml,
            tree_sitter_toml_ng::LANGUAGE.into(),
            "toml",
            tree_sitter_toml_ng::HIGHLIGHTS_QUERY,
        );
        add(
            Lang::Json,
            tree_sitter_json::LANGUAGE.into(),
            "json",
            tree_sitter_json::HIGHLIGHTS_QUERY,
        );
        Self {
            highlighter: Highlighter::new(),
            configs,
        }
    }

    /// Highlight `text`, returning flat non-overlapping spans in ascending
    /// byte order. Unhighlighted gaps are simply absent. Errors (or an
    /// unconfigured language) yield an empty vec — plain text.
    pub fn highlight(&mut self, lang: Lang, text: &str) -> Vec<Span> {
        let Some(cfg) = self.configs.get(&lang) else {
            return Vec::new();
        };
        let Ok(events) = self
            .highlighter
            .highlight(cfg, text.as_bytes(), None, |_| None)
        else {
            return Vec::new();
        };
        let mut spans = Vec::new();
        let mut stack: Vec<Style> = Vec::new();
        for event in events {
            let Ok(event) = event else { break };
            match event {
                HighlightEvent::HighlightStart(h) => stack.push(style_for_capture(h.0)),
                HighlightEvent::HighlightEnd => {
                    stack.pop();
                }
                HighlightEvent::Source { start, end } => {
                    if let Some(&style) = stack.last() {
                        // Merge adjacent same-style ranges.
                        if let Some(last) = spans.last_mut() {
                            let last: &mut Span = last;
                            if last.end == start && last.style == style {
                                last.end = end;
                                continue;
                            }
                        }
                        spans.push(Span { start, end, style });
                    }
                }
            }
        }
        spans
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rust_keywords_and_strings_highlight() {
        let mut set = SyntaxSet::new();
        let src = "fn main() { let s = \"hi\"; // comment\n}";
        let spans = set.highlight(Lang::Rust, src);
        assert!(!spans.is_empty(), "expected spans for rust source");
        let styled = |style: Style| spans.iter().any(|s| s.style == style);
        assert!(
            styled(Style::Keyword),
            "fn/let should be keywords: {spans:?}"
        );
        assert!(styled(Style::String), "\"hi\" should be a string");
        assert!(styled(Style::Comment), "// comment should be a comment");
        // Spans are ascending and non-overlapping.
        for w in spans.windows(2) {
            assert!(w[0].end <= w[1].start, "overlap: {w:?}");
        }
    }

    #[test]
    fn toml_properties_highlight() {
        let mut set = SyntaxSet::new();
        let spans = set.highlight(Lang::Toml, "[package]\nname = \"umber\"\n");
        assert!(!spans.is_empty());
        assert!(spans.iter().any(|s| s.style == Style::String));
    }

    #[test]
    fn unknown_ext_maps_to_none() {
        assert_eq!(lang_for_ext("rs"), Some(Lang::Rust));
        assert_eq!(lang_for_ext("lock"), Some(Lang::Toml));
        assert_eq!(lang_for_ext("xyz"), None);
    }
}
