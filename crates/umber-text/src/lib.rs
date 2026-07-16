//! umber-text — buffer model on ropey.
//!
//! P-phase: **P1** owns the full model (edits, undo tree, multi-cursor data
//! model, marks). This slice adds a single-cursor edit surface (insert /
//! remove at a char index) on top of the read-only line access the render
//! spike needs to draw visible lines, plus a linear **undo/redo** stack, a
//! **dirty** flag for the save indicator, and char-range slicing for the
//! clipboard. Multi-cursor and the full undo *tree* are still P1.

use std::fs::File;
use std::io::{self, BufReader, BufWriter};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use ropey::Rope;

/// Consecutive single-char typing coalesces into one undo group only while
/// keystrokes stay within this gap; a longer pause starts a fresh group so a
/// single undo doesn't wipe a whole paragraph typed over minutes.
const COALESCE_GAP: Duration = Duration::from_secs(1);

/// One primitive edit: text inserted or removed at a char index. Undo applies
/// the inverse; redo re-applies the forward op. `text` is the exact run of
/// chars inserted (for `Insert`) or removed (for `Remove`) so the inverse is
/// self-describing — no cursor bookkeeping needed to reconstruct it.
enum Kind {
    Insert,
    Remove,
}

struct Edit {
    kind: Kind,
    /// Char index at which the run begins.
    pos: usize,
    /// The inserted/removed run.
    text: String,
}

/// An atomic unit of undo: one or more [`Edit`]s applied together. Plain typing
/// coalesces single-char inserts into one group's single edit; a selection
/// replacement (delete + insert) is a two-edit group. The `id` is a monotonic
/// stamp used to detect the dirty state relative to the last save without
/// tracking content hashes (see [`TextBuffer::is_dirty`]).
struct Group {
    id: u64,
    edits: Vec<Edit>,
}

/// An in-memory text buffer backed by a ropey [`Rope`], with a linear
/// undo/redo history.
///
/// Editing routes through [`TextBuffer::insert_char`],
/// [`TextBuffer::insert_str`], and [`TextBuffer::remove_char_range`], which
/// record inverse ops. Multi-step edits (selection replace, cut, paste over a
/// selection) wrap in [`TextBuffer::begin_transaction`] /
/// [`TextBuffer::end_transaction`] so they undo atomically.
pub struct TextBuffer {
    rope: Rope,
    path: Option<PathBuf>,

    undo: Vec<Group>,
    redo: Vec<Group>,
    /// Next group id to hand out (monotonic; never reused).
    next_group_id: u64,
    /// Id of the undo-stack top at the last save (or `None` if the stack was
    /// empty then). `is_dirty` compares the current top against this.
    saved_id: Option<u64>,
    /// Whether the top undo group is still open to absorb the next single-char
    /// typing keystroke. Cleared by navigation, whitespace, saves, undo/redo,
    /// and transactions.
    open_coalesce: bool,
    /// Inside a `begin_transaction`/`end_transaction` span, every recorded edit
    /// appends to one group regardless of coalescing.
    in_transaction: bool,
    /// Receipt time of the last recorded edit, for the coalescing gap check.
    last_edit: Option<Instant>,
}

impl TextBuffer {
    /// An empty scratch buffer (no backing file).
    pub fn empty() -> Self {
        Self::wrap(Rope::new(), None)
    }

    /// Load a file into a rope, streaming it through a [`BufReader`] so a large
    /// file (the P0 100 MB exit criterion) never lands in one contiguous
    /// allocation.
    pub fn from_path(path: impl AsRef<Path>) -> io::Result<Self> {
        let path = path.as_ref();
        let rope = Rope::from_reader(BufReader::new(File::open(path)?))?;
        Ok(Self::wrap(rope, Some(path.to_path_buf())))
    }

    fn wrap(rope: Rope, path: Option<PathBuf>) -> Self {
        Self {
            rope,
            path,
            undo: Vec::new(),
            redo: Vec::new(),
            next_group_id: 0,
            saved_id: None,
            open_coalesce: false,
            in_transaction: false,
            last_edit: None,
        }
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
    /// Recorded as an undoable op; consecutive single chars coalesce (see the
    /// module docs).
    pub fn insert_char(&mut self, char_idx: usize, ch: char) {
        let idx = char_idx.min(self.rope.len_chars());
        self.rope.insert_char(idx, ch);
        self.push_edit(
            Edit {
                kind: Kind::Insert,
                pos: idx,
                text: ch.to_string(),
            },
            true,
        );
    }

    /// Insert `text` at `char_idx` (clamped to the buffer length). Recorded as
    /// a single non-coalescing op (paste / programmatic insert).
    pub fn insert_str(&mut self, char_idx: usize, text: &str) {
        if text.is_empty() {
            return;
        }
        let idx = char_idx.min(self.rope.len_chars());
        self.rope.insert(idx, text);
        self.push_edit(
            Edit {
                kind: Kind::Insert,
                pos: idx,
                text: text.to_string(),
            },
            false,
        );
    }

    /// Remove the half-open char range `start..end` (both clamped, no-op if
    /// empty). Backspace/Delete and selection deletion route through here.
    pub fn remove_char_range(&mut self, start: usize, end: usize) {
        let len = self.rope.len_chars();
        let start = start.min(len);
        let end = end.min(len);
        if start < end {
            let removed = self.rope.slice(start..end).to_string();
            self.rope.remove(start..end);
            self.push_edit(
                Edit {
                    kind: Kind::Remove,
                    pos: start,
                    text: removed,
                },
                false,
            );
        }
    }

    /// Extract the half-open char range `start..end` as an owned string (for
    /// the clipboard). Clamped; empty range yields an empty string.
    pub fn slice_chars(&self, start: usize, end: usize) -> String {
        let len = self.rope.len_chars();
        let start = start.min(len);
        let end = end.min(len);
        if start >= end {
            return String::new();
        }
        self.rope.slice(start..end).to_string()
    }

    // --- undo / redo -------------------------------------------------------

    /// Record an edit into the history, coalescing single-char typing when
    /// `allow_coalesce` and the run is unbroken (contiguous, non-whitespace,
    /// within [`COALESCE_GAP`]). Any new edit invalidates the redo stack.
    fn push_edit(&mut self, edit: Edit, allow_coalesce: bool) {
        self.redo.clear();
        let now = Instant::now();

        let single_nonws_insert = matches!(edit.kind, Kind::Insert)
            && edit.text.chars().count() == 1
            && !edit.text.chars().next().unwrap().is_whitespace();

        if self.in_transaction {
            if let Some(g) = self.undo.last_mut() {
                g.edits.push(edit);
            }
            self.open_coalesce = false;
            self.last_edit = Some(now);
            return;
        }

        let coalesced = allow_coalesce && self.try_coalesce(&edit, now);
        if !coalesced {
            let id = self.next_group_id;
            self.next_group_id += 1;
            self.undo.push(Group {
                id,
                edits: vec![edit],
            });
        }
        // Only an unbroken non-whitespace single-char insert leaves the group
        // open for the next keystroke; whitespace ends the run (word-granular
        // undo), and every other op is its own group.
        self.open_coalesce = allow_coalesce && single_nonws_insert;
        self.last_edit = Some(now);
    }

    /// Try to fold a single-char insert into the current run. Returns `true`
    /// (and mutates the top group) on success.
    fn try_coalesce(&mut self, edit: &Edit, now: Instant) -> bool {
        if !self.open_coalesce {
            return false;
        }
        if !matches!(edit.kind, Kind::Insert) || edit.text.chars().count() != 1 {
            return false;
        }
        if edit.text.chars().next().unwrap().is_whitespace() {
            return false;
        }
        if let Some(prev) = self.last_edit {
            if now.duration_since(prev) > COALESCE_GAP {
                return false;
            }
        }
        if let Some(g) = self.undo.last_mut() {
            if let Some(last) = g.edits.last_mut() {
                if matches!(last.kind, Kind::Insert)
                    && last.pos + last.text.chars().count() == edit.pos
                {
                    last.text.push_str(&edit.text);
                    return true;
                }
            }
        }
        false
    }

    /// End the current typing run so the next keystroke starts a fresh undo
    /// group. Called by the bin on any cursor navigation.
    pub fn break_coalescing(&mut self) {
        self.open_coalesce = false;
    }

    /// Open an atomic edit group: every edit until [`end_transaction`] undoes
    /// as one unit (used for selection replace / cut / paste-over-selection).
    ///
    /// [`end_transaction`]: TextBuffer::end_transaction
    pub fn begin_transaction(&mut self) {
        self.redo.clear();
        let id = self.next_group_id;
        self.next_group_id += 1;
        self.undo.push(Group {
            id,
            edits: Vec::new(),
        });
        self.in_transaction = true;
        self.open_coalesce = false;
    }

    /// Close the current atomic group. An empty group (no edits landed) is
    /// discarded so it doesn't leave a no-op undo step.
    pub fn end_transaction(&mut self) {
        self.in_transaction = false;
        if let Some(g) = self.undo.last() {
            if g.edits.is_empty() {
                self.undo.pop();
            }
        }
    }

    /// Undo the most recent group, applying each edit's inverse in reverse
    /// order. Returns the char index the cursor should move to (the op site),
    /// or `None` if there is nothing to undo.
    pub fn undo(&mut self) -> Option<usize> {
        let group = self.undo.pop()?;
        let mut cursor = 0;
        for edit in group.edits.iter().rev() {
            match edit.kind {
                Kind::Insert => {
                    let n = edit.text.chars().count();
                    self.rope.remove(edit.pos..edit.pos + n);
                    cursor = edit.pos;
                }
                Kind::Remove => {
                    self.rope.insert(edit.pos, &edit.text);
                    cursor = edit.pos + edit.text.chars().count();
                }
            }
        }
        self.redo.push(group);
        self.open_coalesce = false;
        Some(cursor)
    }

    /// Redo the most recently undone group, re-applying each edit forward.
    /// Returns the char index the cursor should move to, or `None` if the redo
    /// stack is empty.
    pub fn redo(&mut self) -> Option<usize> {
        let group = self.redo.pop()?;
        let mut cursor = 0;
        for edit in group.edits.iter() {
            match edit.kind {
                Kind::Insert => {
                    self.rope.insert(edit.pos, &edit.text);
                    cursor = edit.pos + edit.text.chars().count();
                }
                Kind::Remove => {
                    let n = edit.text.chars().count();
                    self.rope.remove(edit.pos..edit.pos + n);
                    cursor = edit.pos;
                }
            }
        }
        self.undo.push(group);
        self.open_coalesce = false;
        Some(cursor)
    }

    // --- save / dirty ------------------------------------------------------

    /// Whether the buffer has unsaved edits. Compares the undo-stack top's id
    /// against the id captured at the last save; coalescing mutates a group in
    /// place but keeps its id, and saves/typing break coalescing so the top id
    /// reliably changes when new content is entered.
    pub fn is_dirty(&self) -> bool {
        self.saved_id != self.undo.last().map(|g| g.id)
    }

    /// Write the buffer back to its backing path. Returns `Ok(true)` on a
    /// successful write, `Ok(false)` when the buffer has no path (scratch), and
    /// the io error on failure. A successful save clears the dirty state.
    pub fn save(&mut self) -> io::Result<bool> {
        let path = match &self.path {
            Some(p) => p.clone(),
            None => return Ok(false),
        };
        let file = File::create(&path)?;
        self.rope.write_to(BufWriter::new(file))?;
        self.saved_id = self.undo.last().map(|g| g.id);
        self.open_coalesce = false;
        Ok(true)
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
