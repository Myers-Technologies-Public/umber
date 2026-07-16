//! umber-text — buffer model on ropey.
//!
//! P-phase: **P1** owns the full model (edits, undo tree, multi-cursor data
//! model, marks). This P0 slice adds a single-cursor edit surface (insert /
//! remove at a char index) on top of the read-only line access the render
//! spike needs to draw visible lines. Multi-cursor and the undo tree are P1.

use std::fs::File;
use std::io::{self, BufReader};
use std::path::{Path, PathBuf};

use ropey::Rope;

/// An in-memory text buffer backed by a ropey [`Rope`].
///
/// P0 scope is single-cursor: char-indexed insert/remove plus the line/char
/// conversions a cursor needs. The undo tree, marks, and multi-cursor land in
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

    /// Total number of `char`s in the buffer (the unit the cursor indexes in).
    pub fn len_chars(&self) -> usize {
        self.rope.len_chars()
    }

    /// Insert a single `char` at `char_idx` (clamped to the buffer length).
    pub fn insert_char(&mut self, char_idx: usize, ch: char) {
        let idx = char_idx.min(self.rope.len_chars());
        self.rope.insert_char(idx, ch);
    }

    /// Insert `text` at `char_idx` (clamped to the buffer length).
    pub fn insert_str(&mut self, char_idx: usize, text: &str) {
        let idx = char_idx.min(self.rope.len_chars());
        self.rope.insert(idx, text);
    }

    /// Remove the half-open char range `start..end` (both clamped, no-op if
    /// empty). Backspace/Delete route through here.
    pub fn remove_char_range(&mut self, start: usize, end: usize) {
        let len = self.rope.len_chars();
        let start = start.min(len);
        let end = end.min(len);
        if start < end {
            self.rope.remove(start..end);
        }
    }

    /// Line index (0-based) containing `char_idx` (clamped).
    pub fn char_to_line(&self, char_idx: usize) -> usize {
        self.rope.char_to_line(char_idx.min(self.rope.len_chars()))
    }

    /// Char index of the first char of `line` (clamped).
    pub fn line_to_char(&self, line: usize) -> usize {
        self.rope.line_to_char(line.min(self.rope.len_lines()))
    }

    /// Char length of `line` excluding a trailing `\n` (or `\r\n`). This is the
    /// visual column count a cursor can occupy on that line.
    pub fn visual_line_len_chars(&self, line: usize) -> usize {
        if line >= self.rope.len_lines() {
            return 0;
        }
        let l = self.rope.line(line);
        let mut n = l.len_chars();
        if n > 0 && l.chars_at(n - 1).next() == Some('\n') {
            n -= 1;
            if n > 0 && l.chars_at(n - 1).next() == Some('\r') {
                n -= 1;
            }
        }
        n
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
