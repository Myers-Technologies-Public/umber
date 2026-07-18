//! Project-wide text search (P5): a dependency-light recursive grep used by
//! the search overlay (Ctrl+Shift+F). Pure and headless-testable; the UI layer
//! only renders the [`Match`] list and jumps to a chosen hit.
//!
//! Scope discipline: skips well-known noise dirs (`.git`, `target`,
//! `node_modules`, hidden dirs), skips files that look binary (a NUL byte in
//! the first chunk) or exceed a size cap, and bounds the total match count so a
//! huge tree can't hang the UI thread. Matching is a case-insensitive
//! substring — literal, not regex, at this slice.

use std::path::{Path, PathBuf};

/// One search hit: file, 1-based line, 0-based column (char offset), and the
/// (trimmed) line text for display.
#[derive(Clone, Debug, PartialEq)]
pub struct Match {
    pub path: PathBuf,
    pub line: usize,
    pub col: usize,
    pub text: String,
}

/// Directory names never descended into.
const SKIP_DIRS: &[&str] = &[".git", "target", "node_modules", ".cargo", "dist", "build"];
/// Files larger than this are skipped (bytes).
const MAX_FILE: u64 = 2 * 1024 * 1024;
/// Bytes sniffed for a NUL to classify a file as binary.
const SNIFF: usize = 8000;

/// Search `root` recursively for `query` (case-insensitive substring), up to
/// `limit` matches. Empty query yields nothing.
pub fn search_dir(root: &Path, query: &str, limit: usize) -> Vec<Match> {
    let mut out = Vec::new();
    if query.is_empty() {
        return out;
    }
    let needle = query.to_lowercase();
    walk(root, &needle, limit, &mut out);
    out
}

/// List workspace files (same skip rules as the content walk), filtered by
/// a case-insensitive subsequence match on the root-relative path — the
/// fuzzy-finder behind the file picker. Empty filter = first `limit` files.
pub fn list_files(root: &Path, filter: &str, limit: usize) -> Vec<std::path::PathBuf> {
    let mut out = Vec::new();
    let needle = filter.to_lowercase();
    walk_list(root, root, &needle, limit, &mut out);
    out
}

fn walk_list(
    root: &Path,
    dir: &Path,
    needle: &str,
    limit: usize,
    out: &mut Vec<std::path::PathBuf>,
) {
    if out.len() >= limit {
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut items: Vec<_> = entries.flatten().collect();
    items.sort_by_key(|e| e.file_name());
    for entry in items {
        if out.len() >= limit {
            return;
        }
        let name = entry.file_name();
        let name = name.to_string_lossy();
        let path = entry.path();
        let is_dir = entry.file_type().map(|t| t.is_dir()).unwrap_or(false);
        if is_dir {
            if name.starts_with('.') || SKIP_DIRS.contains(&name.as_ref()) {
                continue;
            }
            walk_list(root, &path, needle, limit, out);
        } else {
            if name.starts_with('.') {
                continue;
            }
            let rel = path
                .strip_prefix(root)
                .unwrap_or(&path)
                .to_string_lossy()
                .to_lowercase();
            if needle.is_empty() || subseq_match(&rel, needle) {
                out.push(path);
            }
        }
    }
}

/// `needle`'s chars appear in order (not necessarily adjacent) in `hay`.
fn subseq_match(hay: &str, needle: &str) -> bool {
    let mut hay_chars = hay.chars();
    'outer: for nc in needle.chars() {
        for hc in hay_chars.by_ref() {
            if hc == nc {
                continue 'outer;
            }
        }
        return false;
    }
    true
}

fn walk(dir: &Path, needle: &str, limit: usize, out: &mut Vec<Match>) {
    if out.len() >= limit {
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    // Deterministic order: dirs and files sorted by name.
    let mut items: Vec<_> = entries.flatten().collect();
    items.sort_by_key(|e| e.file_name());
    for entry in items {
        if out.len() >= limit {
            return;
        }
        let name = entry.file_name();
        let name = name.to_string_lossy();
        let path = entry.path();
        let is_dir = entry.file_type().map(|t| t.is_dir()).unwrap_or(false);
        if is_dir {
            if name.starts_with('.') || SKIP_DIRS.contains(&name.as_ref()) {
                continue;
            }
            walk(&path, needle, limit, out);
        } else {
            if name.starts_with('.') {
                continue;
            }
            search_file(&path, needle, limit, out);
        }
    }
}

fn search_file(path: &Path, needle: &str, limit: usize, out: &mut Vec<Match>) {
    let meta = match std::fs::metadata(path) {
        Ok(m) => m,
        Err(_) => return,
    };
    if meta.len() > MAX_FILE {
        return;
    }
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(_) => return,
    };
    // Binary sniff: a NUL in the first chunk => skip.
    if bytes.iter().take(SNIFF).any(|&b| b == 0) {
        return;
    }
    let Ok(text) = String::from_utf8(bytes) else {
        return;
    };
    for (i, line) in text.lines().enumerate() {
        if out.len() >= limit {
            return;
        }
        let lower = line.to_lowercase();
        if let Some(byte_idx) = lower.find(needle) {
            // Column as a char offset (the editor cursor is char-indexed).
            let col = line[..byte_idx].chars().count();
            let trimmed = line.trim_start();
            let trimmed_lead = line.len() - trimmed.len();
            out.push(Match {
                path: path.to_path_buf(),
                line: i + 1,
                col,
                text: trimmed.chars().take(120).collect::<String>()
                    + if trimmed.len() > 120 { "\u{2026}" } else { "" },
            });
            let _ = trimmed_lead;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn tmp(name: &str) -> PathBuf {
        let d = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/test-tmp/search")
            .join(name);
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn finds_case_insensitive_matches_with_line_and_col() {
        let d = tmp("basic");
        fs::write(d.join("a.txt"), "hello World\nno match here\nWORLD again\n").unwrap();
        let hits = search_dir(&d, "world", 100);
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].line, 1);
        assert_eq!(hits[0].col, 6); // char offset of "World"
        assert_eq!(hits[1].line, 3);
        assert_eq!(hits[1].col, 0);
    }

    #[test]
    fn skips_binary_and_noise_dirs() {
        let d = tmp("skip");
        fs::write(d.join("code.rs"), "let target = 1;\n").unwrap();
        fs::create_dir_all(d.join("target")).unwrap();
        fs::write(d.join("target/gen.rs"), "target in build output\n").unwrap();
        fs::write(d.join("bin.dat"), [0u8, b'x', b't', b'a', b'r', b'g', 0u8]).unwrap();
        let hits = search_dir(&d, "target", 100);
        // Only the real source line; the target/ dir and binary file are skipped.
        assert_eq!(hits.len(), 1);
        assert!(hits[0].path.ends_with("code.rs"));
    }

    #[test]
    fn respects_the_match_limit() {
        let d = tmp("limit");
        let many = "x\n".repeat(50);
        fs::write(d.join("many.txt"), many).unwrap();
        let hits = search_dir(&d, "x", 10);
        assert_eq!(hits.len(), 10);
    }

    #[test]
    fn empty_query_returns_nothing() {
        let d = tmp("empty");
        fs::write(d.join("a.txt"), "content\n").unwrap();
        assert!(search_dir(&d, "", 100).is_empty());
    }

    #[test]
    fn subseq_match_is_ordered_fuzzy() {
        assert!(subseq_match("crates/umber/src/main.rs", "umain"));
        assert!(subseq_match("crates/umber/src/main.rs", "c/u/s/m.rs"));
        assert!(!subseq_match("crates/umber/src/main.rs", "mainz"));
        assert!(subseq_match("anything", ""));
    }

    #[test]
    fn list_files_filters_and_skips() {
        let d = tmp("list");
        fs::write(d.join("alpha.rs"), "").unwrap();
        fs::write(d.join("beta.txt"), "").unwrap();
        fs::create_dir_all(d.join("target")).unwrap();
        fs::write(d.join("target/skip.rs"), "").unwrap();
        let all = list_files(&d, "", 100);
        assert_eq!(all.len(), 2);
        let rs = list_files(&d, "alph.rs", 100);
        assert_eq!(rs.len(), 1);
        assert!(rs[0].ends_with("alpha.rs"));
    }
}
