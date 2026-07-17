//! Git line-status for the gutter (P5): shells out to `git` (no libgit2
//! dependency) and parses `git diff -U0` hunk headers into per-line change
//! markers the renderer paints in the gutter.
//!
//! The parser (`parse_diff`) is pure and unit-tested; the process call
//! (`file_line_status`) is a thin wrapper. A file with no repo / no changes /
//! any git error yields an empty map — the gutter simply shows nothing.

use std::collections::HashMap;
use std::path::Path;
use std::process::Command;

/// The change state of a single line, for the gutter marker color.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LineChange {
    Added,
    Modified,
    /// Lines were deleted immediately above this line (shown as a marker on
    /// the line that now sits at the deletion point).
    Deleted,
}

/// Parse `git diff -U0` output into `line -> LineChange` (1-based new-file
/// line numbers). `-U0` gives zero context so every hunk is exactly the
/// changed region.
///
/// Hunk header: `@@ -oldStart,oldCount +newStart,newCount @@`. Counts default
/// to 1 when omitted. Classification:
/// - `newCount == 0` -> pure deletion -> mark `Deleted` at `newStart` (the
///   surviving line just after the removed block; `newStart` is the line
///   before it in git's convention, so we mark `newStart` clamped to >= 1)
/// - `oldCount == 0` -> pure addition -> `Added` for the new lines
/// - both > 0 -> modification -> `Modified` for the new lines
pub fn parse_diff(diff: &str) -> HashMap<usize, LineChange> {
    let mut out = HashMap::new();
    for line in diff.lines() {
        if !line.starts_with("@@") {
            continue;
        }
        // @@ -a,b +c,d @@
        let Some(plus) = line.split('+').nth(1) else {
            continue;
        };
        let spec = plus.split('@').next().unwrap_or("").trim();
        let mut it = spec.split(',');
        let Some(new_start) = it.next().and_then(|s| s.trim().parse::<usize>().ok()) else {
            continue;
        };
        let new_count: usize = it.next().and_then(|s| s.trim().parse().ok()).unwrap_or(1);

        // Old count, for add-vs-modify: `@@ -a,b +c,d @@`.
        let old_count: usize = line
            .split('-')
            .nth(1)
            .and_then(|s| s.split_whitespace().next())
            .and_then(|s| s.split(',').nth(1))
            .and_then(|s| s.parse().ok())
            .unwrap_or(1);

        if new_count == 0 {
            // Pure deletion: mark the line at/after the removal point.
            let at = new_start.max(1);
            out.entry(at).or_insert(LineChange::Deleted);
        } else {
            let kind = if old_count == 0 {
                LineChange::Added
            } else {
                LineChange::Modified
            };
            for l in new_start..new_start + new_count {
                out.insert(l.max(1), kind);
            }
        }
    }
    out
}

/// Line status for `path` from `git diff -U0`. Empty on any error (not a repo,
/// git missing, file untracked/unchanged).
pub fn file_line_status(path: &Path) -> HashMap<usize, LineChange> {
    let Some(dir) = path.parent() else {
        return HashMap::new();
    };
    let Some(name) = path.file_name() else {
        return HashMap::new();
    };
    let output = Command::new("git")
        .arg("diff")
        .arg("--no-color")
        .arg("-U0")
        .arg("--")
        .arg(name)
        .current_dir(dir)
        .output();
    match output {
        Ok(o) if o.status.success() => parse_diff(&String::from_utf8_lossy(&o.stdout)),
        _ => HashMap::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_addition_modification_deletion() {
        // 3 lines added at new-line 2; 1 line modified at 10; deletion after 20.
        let diff = "\
diff --git a/f b/f
--- a/f
+++ b/f
@@ -1,0 +2,3 @@
+new1
+new2
+new3
@@ -10,1 +13,1 @@
-old10
+mod10
@@ -20,2 +22,0 @@
-gone1
-gone2
";
        let m = parse_diff(diff);
        assert_eq!(m.get(&2), Some(&LineChange::Added));
        assert_eq!(m.get(&3), Some(&LineChange::Added));
        assert_eq!(m.get(&4), Some(&LineChange::Added));
        assert_eq!(m.get(&13), Some(&LineChange::Modified));
        assert_eq!(m.get(&22), Some(&LineChange::Deleted));
    }

    #[test]
    fn empty_diff_yields_nothing() {
        assert!(parse_diff("").is_empty());
        assert!(parse_diff("diff --git a/f b/f\n").is_empty());
    }

    #[test]
    fn end_to_end_against_a_real_repo() {
        // Skip gracefully if git isn't available in the environment.
        if Command::new("git").arg("--version").output().is_err() {
            return;
        }
        let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/test-tmp/git_e2e");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let run = |args: &[&str]| {
            Command::new("git")
                .args(args)
                .current_dir(&dir)
                .output()
                .unwrap();
        };
        run(&["init", "-q"]);
        run(&["config", "user.email", "t@t"]);
        run(&["config", "user.name", "t"]);
        std::fs::write(dir.join("f.txt"), "a\nb\nc\n").unwrap();
        run(&["add", "."]);
        run(&["commit", "-qm", "init"]);
        // Modify line 2.
        std::fs::write(dir.join("f.txt"), "a\nB-changed\nc\n").unwrap();
        let m = file_line_status(&dir.join("f.txt"));
        assert_eq!(m.get(&2), Some(&LineChange::Modified), "map: {m:?}");
    }
}
