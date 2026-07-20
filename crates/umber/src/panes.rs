//! Ghostty-style tiling: a binary split tree over the content area.
//!
//! Leaves host content (the editor, or one terminal session by id);
//! internal nodes split horizontally or vertically at a draggable ratio.
//! The tree is pure layout logic — no renderer types. `layout()` walks it
//! into normalized `[0,1]` rects that the renderer scales to pixels, and
//! divider rects are emitted alongside for hit-testing drags.
//!
//! Invariants:
//! - Exactly one `Editor` leaf exists (documents/tabs live inside it).
//! - Pane ids are stable across splits/closes (monotonic counter).
//! - Ratios clamp to [MIN_RATIO, 1-MIN_RATIO] so no pane collapses.

/// Minimum share a pane keeps on either side of a divider.
pub const MIN_RATIO: f32 = 0.15;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplitDir {
    /// Side-by-side (divider is vertical).
    Horizontal,
    /// Stacked (divider is horizontal).
    Vertical,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaneContent {
    /// The live editing surface (exactly one exists).
    Editor,
    Terminal(u64),
    /// A read-view tile of the sidebar document at this index; clicking it
    /// swaps it with the live editor.
    Doc(usize),
}

#[derive(Debug, Clone)]
pub enum PaneNode {
    Leaf {
        id: u64,
        content: PaneContent,
    },
    Split {
        dir: SplitDir,
        ratio: f32,
        a: Box<PaneNode>,
        b: Box<PaneNode>,
    },
}

/// Normalized rect in the content area, all fields 0..=1.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Frac {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

/// One laid-out pane.
#[derive(Debug, Clone, Copy)]
pub struct PaneRect {
    pub id: u64,
    pub content: PaneContent,
    pub rect: Frac,
}

/// One divider between two subtrees, for drag hit-testing. `rect` is the
/// divider line's normalized position: for `Horizontal` splits it is a
/// vertical line (w == 0), for `Vertical` a horizontal line (h == 0).
#[derive(Debug, Clone, Copy)]
pub struct DividerRect {
    /// Path to the split node that owns this divider (see [`PaneTree::drag`]).
    pub path: u32,
    pub dir: SplitDir,
    pub rect: Frac,
}

#[derive(Debug)]
pub struct PaneTree {
    root: PaneNode,
    next_id: u64,
    pub focused: u64,
}

impl PaneTree {
    /// A fresh tree: just the editor pane, focused.
    pub fn new() -> Self {
        Self {
            root: PaneNode::Leaf {
                id: 0,
                content: PaneContent::Editor,
            },
            next_id: 1,
            focused: 0,
        }
    }

    /// True when the tree is a single leaf (structural). The leaf may hold
    /// the editor, a terminal, or a doc — so this is *not* a reliable test
    /// for "no tiling / plain editor". Use [`is_plain_editor`] for that.
    pub fn is_single(&self) -> bool {
        matches!(self.root, PaneNode::Leaf { .. })
    }

    /// The "plain editor" state: the whole content area is the single editor
    /// leaf. This — not [`is_single`] — is the correct gate for legacy
    /// fullscreen editor / terminal-tab behaviour, because a lone terminal
    /// or doc leaf is *also* a single leaf yet must be driven by the pane
    /// path (its own focus, input routing, and render mode).
    pub fn is_plain_editor(&self) -> bool {
        matches!(
            self.root,
            PaneNode::Leaf {
                content: PaneContent::Editor,
                ..
            }
        )
    }

    /// True whenever the pane system owns the content area: a split exists,
    /// or the single leaf is a terminal/doc rather than the editor. The
    /// inverse of [`is_plain_editor`].
    pub fn tiling_active(&self) -> bool {
        !self.is_plain_editor()
    }

    /// The id of the leaf currently holding the editor surface, if any. The
    /// editor tile is not always id 0 — once closed and reopened it takes a
    /// fresh id — so callers that want to focus it must look it up here.
    pub fn editor_leaf(&self) -> Option<u64> {
        fn walk(n: &PaneNode) -> Option<u64> {
            match n {
                PaneNode::Leaf {
                    id,
                    content: PaneContent::Editor,
                } => Some(*id),
                PaneNode::Leaf { .. } => None,
                PaneNode::Split { a, b, .. } => walk(a).or_else(|| walk(b)),
            }
        }
        walk(&self.root)
    }

    /// True when any leaf still holds the editor surface. False once the
    /// editor pane has been closed (the editor is hidden until reopened).
    pub fn has_editor(&self) -> bool {
        fn walk(n: &PaneNode) -> bool {
            match n {
                PaneNode::Leaf { content, .. } => matches!(content, PaneContent::Editor),
                PaneNode::Split { a, b, .. } => walk(a) || walk(b),
            }
        }
        walk(&self.root)
    }

    /// Split the focused pane in `dir`, placing `content` in the new half.
    /// `before` = the new pane takes the left/top half (split left/up);
    /// otherwise the right/bottom half. Returns the new pane's id and
    /// focuses it.
    pub fn split(&mut self, dir: SplitDir, content: PaneContent, before: bool) -> u64 {
        let new_id = self.next_id;
        self.next_id += 1;
        let focused = self.focused;
        Self::split_node(&mut self.root, focused, dir, content, new_id, before);
        self.focused = new_id;
        new_id
    }

    fn split_node(
        node: &mut PaneNode,
        target: u64,
        dir: SplitDir,
        content: PaneContent,
        new_id: u64,
        before: bool,
    ) -> bool {
        match node {
            PaneNode::Leaf { id, .. } if *id == target => {
                let old = std::mem::replace(
                    node,
                    PaneNode::Leaf {
                        id: 0,
                        content: PaneContent::Editor,
                    },
                );
                let new_leaf = PaneNode::Leaf {
                    id: new_id,
                    content,
                };
                let (a, b) = if before {
                    (new_leaf, old)
                } else {
                    (old, new_leaf)
                };
                *node = PaneNode::Split {
                    dir,
                    ratio: 0.5,
                    a: Box::new(a),
                    b: Box::new(b),
                };
                true
            }
            PaneNode::Leaf { .. } => false,
            PaneNode::Split { a, b, .. } => {
                Self::split_node(a, target, dir, content, new_id, before)
                    || Self::split_node(b, target, dir, content, new_id, before)
            }
        }
    }

    /// Close pane `id`: its sibling subtree replaces the parent split.
    /// The editor pane cannot be closed; returns the closed pane's content
    /// (so the app can shut its terminal down) or `None` if nothing closed.
    pub fn close(&mut self, id: u64) -> Option<PaneContent> {
        if matches!(Self::find(&self.root, id), None | Some(PaneContent::Editor)) {
            return None;
        }
        let closed = Self::close_node(&mut self.root, id)?;
        if !self.contains(self.focused) {
            // Focus falls back to the first leaf (leftmost/topmost).
            self.focused = Self::first_leaf(&self.root);
        }
        Some(closed)
    }

    /// Close pane `id` even when it is the editor tile — the caller must
    /// guarantee another editor surface exists (e.g. a promoted document
    /// view). Refuses only if `id` is absent or the tree is a single leaf.
    pub fn force_close(&mut self, id: u64) -> Option<PaneContent> {
        if self.is_single() || Self::find(&self.root, id).is_none() {
            return None;
        }
        let closed = Self::close_node(&mut self.root, id)?;
        if !self.contains(self.focused) {
            self.focused = Self::first_leaf(&self.root);
        }
        Some(closed)
    }

    fn close_node(node: &mut PaneNode, target: u64) -> Option<PaneContent> {
        if !matches!(node, PaneNode::Split { .. }) {
            return None;
        }
        // Take the split by value: no overlapping borrows while we decide
        // whether a child leaf is the victim and its sibling replaces us.
        let taken = std::mem::replace(
            node,
            PaneNode::Leaf {
                id: u64::MAX,
                content: PaneContent::Editor,
            },
        );
        let PaneNode::Split { dir, ratio, a, b } = taken else {
            unreachable!("checked above");
        };
        let leaf_match = |n: &PaneNode| match n {
            PaneNode::Leaf { id, content } if *id == target => Some(*content),
            _ => None,
        };
        if let Some(c) = leaf_match(&a) {
            *node = *b;
            return Some(c);
        }
        if let Some(c) = leaf_match(&b) {
            *node = *a;
            return Some(c);
        }
        // Neither child is the victim leaf: recurse, then reassemble.
        let (mut a, mut b) = (a, b);
        let found = Self::close_node(&mut a, target).or_else(|| Self::close_node(&mut b, target));
        *node = PaneNode::Split { dir, ratio, a, b };
        found
    }

    /// Replace the content of pane `id` (the live-editor swap on click).
    pub fn set_content(&mut self, id: u64, content: PaneContent) {
        Self::set_content_node(&mut self.root, id, content);
    }

    fn set_content_node(node: &mut PaneNode, id: u64, content: PaneContent) -> bool {
        match node {
            PaneNode::Leaf { id: i, content: c } if *i == id => {
                *c = content;
                true
            }
            PaneNode::Leaf { .. } => false,
            PaneNode::Split { a, b, .. } => {
                Self::set_content_node(a, id, content) || Self::set_content_node(b, id, content)
            }
        }
    }

    /// A closed sidebar document shifts every later doc index down by one;
    /// tiles bound to the closed doc are returned so the caller closes them.
    pub fn remap_docs_after_close(&mut self, closed: usize) -> Vec<u64> {
        let mut dead = Vec::new();
        Self::remap_node(&mut self.root, closed, &mut dead);
        dead
    }

    fn remap_node(node: &mut PaneNode, closed: usize, dead: &mut Vec<u64>) {
        match node {
            PaneNode::Leaf {
                id,
                content: PaneContent::Doc(i),
            } => {
                if *i == closed {
                    dead.push(*id);
                } else if *i > closed {
                    *i -= 1;
                }
            }
            PaneNode::Leaf { .. } => {}
            PaneNode::Split { a, b, .. } => {
                Self::remap_node(a, closed, dead);
                Self::remap_node(b, closed, dead);
            }
        }
    }

    pub fn contains(&self, id: u64) -> bool {
        Self::find(&self.root, id).is_some()
    }

    /// Content of pane `id`, if it is present in the tree.
    pub fn content_of(&self, id: u64) -> Option<PaneContent> {
        Self::find(&self.root, id)
    }

    /// The direct leaf child on one side of the split identified by `path`
    /// (see [`DividerRect::path`]): `first` selects the a/left/top child.
    /// Returns the child's id only when it is a leaf (not a nested split),
    /// so a divider dragged to the edge can close the single tile there.
    pub fn edge_child_leaf(&self, path: u32, first: bool) -> Option<u64> {
        let mut node = &self.root;
        // Path is `1 b_{k-1} .. b_0`; the leading 1 is the root sentinel, each
        // remaining bit (MSB->LSB) picks the a (0) or b (1) child.
        let depth = 32 - path.leading_zeros();
        for i in (0..depth.saturating_sub(1)).rev() {
            match node {
                PaneNode::Split { a, b, .. } => {
                    node = if (path >> i) & 1 == 0 { a } else { b };
                }
                PaneNode::Leaf { .. } => return None,
            }
        }
        match node {
            PaneNode::Split { a, b, .. } => {
                let child = if first { a.as_ref() } else { b.as_ref() };
                match child {
                    PaneNode::Leaf { id, .. } => Some(*id),
                    PaneNode::Split { .. } => None,
                }
            }
            PaneNode::Leaf { .. } => None,
        }
    }

    pub fn find(node: &PaneNode, id: u64) -> Option<PaneContent> {
        match node {
            PaneNode::Leaf { id: i, content } if *i == id => Some(*content),
            PaneNode::Leaf { .. } => None,
            PaneNode::Split { a, b, .. } => Self::find(a, id).or_else(|| Self::find(b, id)),
        }
    }

    fn first_leaf(node: &PaneNode) -> u64 {
        match node {
            PaneNode::Leaf { id, .. } => *id,
            PaneNode::Split { a, .. } => Self::first_leaf(a),
        }
    }

    /// The focused pane's content.
    pub fn focused_content(&self) -> PaneContent {
        Self::find(&self.root, self.focused).unwrap_or(PaneContent::Editor)
    }

    /// The pane whose rect contains normalized point `(x, y)`.
    pub fn pane_at(&self, x: f32, y: f32) -> Option<(u64, PaneContent)> {
        self.layout().into_iter().find_map(|p| {
            let r = p.rect;
            (x >= r.x && x < r.x + r.w && y >= r.y && y < r.y + r.h).then_some((p.id, p.content))
        })
    }

    /// Focus the pane whose rect contains normalized point `(x, y)`.
    /// Returns the newly focused id.
    pub fn focus_at(&mut self, x: f32, y: f32) -> u64 {
        if let Some((id, _)) = self.pane_at(x, y) {
            self.focused = id;
        }
        self.focused
    }

    /// Walk the tree into normalized pane rects + dividers.
    /// `path` encodes the route to each split node: bit-per-level, LSB first
    /// (0 = `a`, 1 = `b`), with a leading 1 sentinel — enough for any
    /// practical tiling depth (31 levels).
    pub fn layout(&self) -> Vec<PaneRect> {
        let mut out = Vec::new();
        Self::layout_node(
            &self.root,
            Frac {
                x: 0.0,
                y: 0.0,
                w: 1.0,
                h: 1.0,
            },
            &mut out,
            &mut Vec::new(),
        );
        out
    }

    pub fn dividers(&self) -> Vec<DividerRect> {
        let mut panes = Vec::new();
        let mut divs = Vec::new();
        Self::layout_node(
            &self.root,
            Frac {
                x: 0.0,
                y: 0.0,
                w: 1.0,
                h: 1.0,
            },
            &mut panes,
            &mut divs,
        );
        divs
    }

    fn layout_node(
        node: &PaneNode,
        rect: Frac,
        out: &mut Vec<PaneRect>,
        divs: &mut Vec<DividerRect>,
    ) {
        Self::layout_walk(node, rect, 1, out, divs);
    }

    fn layout_walk(
        node: &PaneNode,
        rect: Frac,
        path: u32,
        out: &mut Vec<PaneRect>,
        divs: &mut Vec<DividerRect>,
    ) {
        match node {
            PaneNode::Leaf { id, content } => out.push(PaneRect {
                id: *id,
                content: *content,
                rect,
            }),
            PaneNode::Split { dir, ratio, a, b } => {
                let (ra, rb, dv) = match dir {
                    SplitDir::Horizontal => {
                        let aw = rect.w * ratio;
                        (
                            Frac { w: aw, ..rect },
                            Frac {
                                x: rect.x + aw,
                                w: rect.w - aw,
                                ..rect
                            },
                            Frac {
                                x: rect.x + aw,
                                y: rect.y,
                                w: 0.0,
                                h: rect.h,
                            },
                        )
                    }
                    SplitDir::Vertical => {
                        let ah = rect.h * ratio;
                        (
                            Frac { h: ah, ..rect },
                            Frac {
                                y: rect.y + ah,
                                h: rect.h - ah,
                                ..rect
                            },
                            Frac {
                                x: rect.x,
                                y: rect.y + ah,
                                w: rect.w,
                                h: 0.0,
                            },
                        )
                    }
                };
                divs.push(DividerRect {
                    path,
                    dir: *dir,
                    rect: dv,
                });
                Self::layout_walk(a, ra, path << 1, out, divs);
                Self::layout_walk(b, rb, (path << 1) | 1, out, divs);
            }
        }
    }

    /// Set the ratio of the split at `path` (from [`DividerRect::path`]) so
    /// the divider lands at normalized `pos` within that split's rect span.
    pub fn drag(&mut self, path: u32, pos: f32) {
        // Recompute the target split's rect by walking the recorded path.
        fn walk(node: &mut PaneNode, rect: Frac, path: u32, target: u32, pos: f32) -> bool {
            let PaneNode::Split { dir, ratio, a, b } = node else {
                return false;
            };
            if path == target {
                let new = match dir {
                    SplitDir::Horizontal if rect.w > 0.0 => (pos - rect.x) / rect.w,
                    SplitDir::Vertical if rect.h > 0.0 => (pos - rect.y) / rect.h,
                    _ => return true,
                };
                *ratio = new.clamp(MIN_RATIO, 1.0 - MIN_RATIO);
                return true;
            }
            let (ra, rb) = match dir {
                SplitDir::Horizontal => {
                    let aw = rect.w * *ratio;
                    (
                        Frac { w: aw, ..rect },
                        Frac {
                            x: rect.x + aw,
                            w: rect.w - aw,
                            ..rect
                        },
                    )
                }
                SplitDir::Vertical => {
                    let ah = rect.h * *ratio;
                    (
                        Frac { h: ah, ..rect },
                        Frac {
                            y: rect.y + ah,
                            h: rect.h - ah,
                            ..rect
                        },
                    )
                }
            };
            walk(a, ra, path << 1, target, pos) || walk(b, rb, (path << 1) | 1, target, pos)
        }
        let root_rect = Frac {
            x: 0.0,
            y: 0.0,
            w: 1.0,
            h: 1.0,
        };
        walk(&mut self.root, root_rect, 1, path, pos);
    }
}

impl Default for PaneTree {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_editor_layout_fills() {
        let t = PaneTree::new();
        assert!(t.is_single());
        let l = t.layout();
        assert_eq!(l.len(), 1);
        assert_eq!(l[0].content, PaneContent::Editor);
        assert_eq!(l[0].rect.w, 1.0);
        assert_eq!(l[0].rect.h, 1.0);
    }

    #[test]
    fn split_right_then_down() {
        let mut t = PaneTree::new();
        let a = t.split(SplitDir::Horizontal, PaneContent::Terminal(1), false);
        assert_eq!(t.focused, a);
        let b = t.split(SplitDir::Vertical, PaneContent::Terminal(2), false);
        assert_eq!(t.focused, b);
        let l = t.layout();
        assert_eq!(l.len(), 3);
        // Editor keeps the left half.
        let ed = l.iter().find(|p| p.content == PaneContent::Editor).unwrap();
        assert!((ed.rect.w - 0.5).abs() < 1e-6);
        assert!((ed.rect.h - 1.0).abs() < 1e-6);
        // The two terminals stack in the right half.
        let t1 = l
            .iter()
            .find(|p| p.content == PaneContent::Terminal(1))
            .unwrap();
        let t2 = l
            .iter()
            .find(|p| p.content == PaneContent::Terminal(2))
            .unwrap();
        assert!((t1.rect.x - 0.5).abs() < 1e-6);
        assert!((t1.rect.h - 0.5).abs() < 1e-6);
        assert!((t2.rect.y - 0.5).abs() < 1e-6);
    }

    #[test]
    fn close_merges_sibling_back() {
        let mut t = PaneTree::new();
        let a = t.split(SplitDir::Horizontal, PaneContent::Terminal(1), false);
        let closed = t.close(a);
        assert_eq!(closed, Some(PaneContent::Terminal(1)));
        assert!(t.is_single());
        assert_eq!(t.focused_content(), PaneContent::Editor);
    }

    #[test]
    fn editor_pane_cannot_close() {
        let mut t = PaneTree::new();
        t.split(SplitDir::Horizontal, PaneContent::Terminal(1), false);
        assert_eq!(t.close(0), None);
        assert_eq!(t.layout().len(), 2);
    }

    #[test]
    fn focus_at_picks_pane_under_point() {
        let mut t = PaneTree::new();
        t.split(SplitDir::Horizontal, PaneContent::Terminal(1), false);
        assert_eq!(t.focus_at(0.25, 0.5), 0);
        assert_eq!(t.focused_content(), PaneContent::Editor);
        let id = t.focus_at(0.75, 0.5);
        assert_eq!(t.focused_content(), PaneContent::Terminal(1));
        assert_eq!(t.focused, id);
    }

    #[test]
    fn drag_moves_divider_clamped() {
        let mut t = PaneTree::new();
        t.split(SplitDir::Horizontal, PaneContent::Terminal(1), false);
        let divs = t.dividers();
        assert_eq!(divs.len(), 1);
        assert!((divs[0].rect.x - 0.5).abs() < 1e-6);
        t.drag(divs[0].path, 0.7);
        let divs = t.dividers();
        assert!((divs[0].rect.x - 0.7).abs() < 1e-6);
        // Clamp: dragging to the edge keeps MIN_RATIO.
        t.drag(divs[0].path, 0.01);
        let divs = t.dividers();
        assert!((divs[0].rect.x - MIN_RATIO).abs() < 1e-6);
    }

    #[test]
    fn nested_close_refocuses_first_leaf() {
        let mut t = PaneTree::new();
        let t1 = t.split(SplitDir::Horizontal, PaneContent::Terminal(1), false);
        t.focused = t1;
        let t2 = t.split(SplitDir::Vertical, PaneContent::Terminal(2), false);
        assert_eq!(t.layout().len(), 3);
        assert_eq!(t.close(t2), Some(PaneContent::Terminal(2)));
        assert!(t.contains(t1));
        assert_eq!(t.layout().len(), 2);
        // Focus stayed valid (t2 was focused; falls back to first leaf).
        assert!(t.contains(t.focused));
    }

    #[test]
    fn doc_tiles_swap_and_remap() {
        let mut t = PaneTree::new();
        let d = t.split(SplitDir::Horizontal, PaneContent::Doc(2), false);
        // Live-editor swap: doc tile becomes the editor and vice versa.
        t.set_content(0, PaneContent::Doc(0));
        t.set_content(d, PaneContent::Editor);
        assert_eq!(PaneTree::find(&t.root, d), Some(PaneContent::Editor));
        assert_eq!(PaneTree::find(&t.root, 0), Some(PaneContent::Doc(0)));
        // Closing sidebar doc 0: the Doc(0) tile dies, higher indices shift.
        t.set_content(0, PaneContent::Doc(3));
        let dead = t.remap_docs_after_close(1);
        assert!(dead.is_empty());
        assert_eq!(PaneTree::find(&t.root, 0), Some(PaneContent::Doc(2)));
        let dead = t.remap_docs_after_close(2);
        assert_eq!(dead, vec![0]);
    }

    #[test]
    fn split_before_places_new_pane_first() {
        let mut t = PaneTree::new();
        // Split LEFT: the new terminal takes the left half.
        t.split(SplitDir::Horizontal, PaneContent::Terminal(1), true);
        let l = t.layout();
        let term = l
            .iter()
            .find(|p| p.content == PaneContent::Terminal(1))
            .unwrap();
        let ed = l.iter().find(|p| p.content == PaneContent::Editor).unwrap();
        assert!(term.rect.x.abs() < 1e-6);
        assert!((ed.rect.x - 0.5).abs() < 1e-6);
        // Split UP from the editor: the new terminal sits above it.
        t.focused = ed.id;
        t.split(SplitDir::Vertical, PaneContent::Terminal(2), true);
        let l = t.layout();
        let t2 = l
            .iter()
            .find(|p| p.content == PaneContent::Terminal(2))
            .unwrap();
        let ed = l.iter().find(|p| p.content == PaneContent::Editor).unwrap();
        assert!(t2.rect.y.abs() < 1e-6);
        assert!((t2.rect.x - 0.5).abs() < 1e-6);
        assert!((ed.rect.y - 0.5).abs() < 1e-6);
    }

    #[test]
    fn lone_terminal_pane_is_not_plain_editor() {
        // Editor closed down to a single terminal leaf: still `is_single`,
        // but the pane path must own it — `tiling_active`, not plain editor.
        let mut t = PaneTree::new();
        let term = t.split(SplitDir::Horizontal, PaneContent::Terminal(1), false);
        // Force the editor tile out (as close_pane does when nothing promotes).
        assert_eq!(t.force_close(0), Some(PaneContent::Editor));
        assert!(t.is_single());
        assert!(!t.is_plain_editor());
        assert!(t.tiling_active());
        assert_eq!(t.editor_leaf(), None);
        // The surviving terminal keeps focus and reports as a terminal, so
        // `focused_pane_term_id` upstream stays truthful.
        assert_eq!(t.focused, term);
        assert_eq!(t.focused_content(), PaneContent::Terminal(1));
    }

    #[test]
    fn editor_leaf_tracks_reopened_id() {
        let mut t = PaneTree::new();
        assert_eq!(t.editor_leaf(), Some(0));
        t.split(SplitDir::Horizontal, PaneContent::Terminal(1), false);
        assert_eq!(t.force_close(0), Some(PaneContent::Editor));
        assert_eq!(t.editor_leaf(), None);
        // Reopen the editor beside the terminal: it takes a fresh id, not 0,
        // which is exactly why `pane_focus_editor` must look the id up.
        let reopened = t.split(SplitDir::Horizontal, PaneContent::Editor, true);
        assert_ne!(reopened, 0);
        assert_eq!(t.editor_leaf(), Some(reopened));
    }
}
