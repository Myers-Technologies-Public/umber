//! umber-text — buffer model on ropey.
//!
//! P-phase: **P1** owns the full model (edits, undo tree, multi-cursor data
//! model, marks). This P0 slice is read-only: load a file into a [`Rope`] and
//! expose the line access the render spike needs to draw visible lines.

use std::fs::File;
use std::io::{self, BufReader};
use std::path::{Path, PathBuf};

use ropey::Rope;

/// An in-memory text buffer backed by a ropey [`Rope`].
///
/// P0 scope is deliberately read-only. Edits, the undo tree, and marks land in
/// P1 on top of this same rope.
pub struct TextBuffer {
    rope: Rope,
    path: Option<PathBuf>,
}

impl TextBuffer {
    /// An empty scratch buffer (no backing file).
    pub fn empty() -> Self {
        Self {
            rope: Rope::new(),
            path: None,
        }
    }

    /// Load a file into a rope, streaming it through a [`BufReader`] so a large
    /// file (the P0 100 MB exit criterion) never lands in one contiguous
    /// allocation.
    pub fn from_path(path: impl AsRef<Path>) -> io::Result<Self> {
        let path = path.as_ref();
        let rope = Rope::from_reader(BufReader::new(File::open(path)?))?;
        Ok(Self {
            rope,
            path: Some(path.to_path_buf()),
        })
    }

    /// The backing file path, if this buffer was loaded from one.
    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    /// Total number of lines in the buffer.
    pub fn len_lines(&self) -> usize {
        self.rope.len_lines()
    }

    /// Total number of bytes in the buffer.
    pub fn len_bytes(&self) -> usize {
        self.rope.len_bytes()
    }

    /// Collect up to `count` lines starting at `start` (0-based) into a single
    /// string for shaping. This is the render spike's window into the rope; P1
    /// replaces it with damage-tracked, per-line shaping in umber-ui.
    pub fn visible_text(&self, start: usize, count: usize) -> String {
        let last = self.rope.len_lines();
        let start = start.min(last);
        let end = start.saturating_add(count).min(last);
        let mut out = String::new();
        for line in start..end {
            out.push_str(&self.rope.line(line).to_string());
        }
        out
    }
}

impl Default for TextBuffer {
    fn default() -> Self {
        Self::empty()
    }
}
