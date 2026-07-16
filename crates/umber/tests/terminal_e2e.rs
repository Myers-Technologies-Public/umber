//! Headless e2e tests for the embedded terminal session (P3): a real PTY and
//! a real `/bin/sh`, no window. These exercise the full pipeline the UI sits
//! on: spawn -> reader thread -> parser grid -> coalesced wakeups -> content
//! snapshot -> shutdown/reap.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use umber::terminal::{TermNotifier, TerminalSession};

#[derive(Clone, Default)]
struct CountingNotifier {
    wakes: Arc<AtomicUsize>,
    exits: Arc<AtomicUsize>,
}

impl TermNotifier for CountingNotifier {
    fn wake(&self) {
        self.wakes.fetch_add(1, Ordering::SeqCst);
    }
    fn child_exited(&self) {
        self.exits.fetch_add(1, Ordering::SeqCst);
    }
}

/// Generous bound so slow CI can't flake the suite; the loops exit as soon as
/// the condition holds (typically well under a second).
const DEADLINE: Duration = Duration::from_secs(5);

fn wait_for(mut pred: impl FnMut() -> bool) -> bool {
    let start = Instant::now();
    while start.elapsed() < DEADLINE {
        if pred() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    false
}

fn sh(cmd: &str) -> Option<(String, Vec<String>)> {
    Some(("/bin/sh".into(), vec!["-c".into(), cmd.into()]))
}

fn grid_contains(session: &TerminalSession<CountingNotifier>, needle: &str) -> bool {
    // take_dirty BEFORE content(): the documented lost-wakeup ordering.
    session.take_dirty();
    session.content().0.contains(needle)
}

#[test]
fn printf_output_lands_in_grid() {
    let session = TerminalSession::spawn_with_shell(
        CountingNotifier::default(),
        80,
        24,
        8,
        16,
        sh("printf 'UMBER_E2E_OK'; sleep 5"),
    )
    .expect("spawn pty");
    assert!(
        wait_for(|| grid_contains(&session, "UMBER_E2E_OK")),
        "printf output never appeared in the grid"
    );
}

#[test]
fn written_input_echoes_back() {
    // Interactive sh on a pty echoes typed input; the echo of the marker is
    // the assertion.
    let session = TerminalSession::spawn_with_shell(
        CountingNotifier::default(),
        80,
        24,
        8,
        16,
        Some(("/bin/sh".into(), Vec::new())),
    )
    .expect("spawn pty");
    session.write(b"echo UMBER_ECHO_42\n".to_vec());
    assert!(
        wait_for(|| grid_contains(&session, "UMBER_ECHO_42")),
        "typed input never echoed back through the grid"
    );
}

#[test]
fn child_exit_is_detected() {
    let notifier = CountingNotifier::default();
    let exits = notifier.exits.clone();
    let session =
        TerminalSession::spawn_with_shell(notifier, 80, 24, 8, 16, sh("exit 0")).expect("spawn");
    assert!(
        wait_for(|| session.has_exited()),
        "child exit was never detected"
    );
    assert!(
        wait_for(|| exits.load(Ordering::SeqCst) >= 1),
        "child_exited notification never fired"
    );
}

#[test]
fn ansi_color_sequences_parse_without_panic() {
    let session = TerminalSession::spawn_with_shell(
        CountingNotifier::default(),
        80,
        24,
        8,
        16,
        sh("printf '\\033[31mUMBER_RED\\033[0m \\033[1;44mBLUE_BG\\033[0m'; sleep 5"),
    )
    .expect("spawn pty");
    assert!(
        wait_for(|| grid_contains(&session, "UMBER_RED")),
        "colored text never appeared (or the parser panicked)"
    );
}

#[test]
fn wakeups_are_coalesced_not_per_byte() {
    let notifier = CountingNotifier::default();
    let wakes = notifier.wakes.clone();
    let session = TerminalSession::spawn_with_shell(
        notifier,
        80,
        24,
        8,
        16,
        // ~64KB of output in one burst.
        sh("i=0; while [ $i -lt 1000 ]; do echo 'UMBER_FLOOD_LINE_PADDING_PADDING_PADDING_PADDING'; i=$((i+1)); done; printf 'UMBER_FLOOD_DONE'; sleep 5"),
    )
    .expect("spawn pty");
    assert!(
        wait_for(|| grid_contains(&session, "UMBER_FLOOD_DONE")),
        "flood never completed"
    );
    // Coalescing bound: without take_dirty consumption there can be at most
    // one wake in flight per consume. We consumed in the poll loop (~every
    // 20ms), so wakes must be far below the ~65k bytes/1000 lines written.
    let observed = wakes.load(Ordering::SeqCst);
    assert!(
        observed < 500,
        "expected coalesced wakeups, got {observed} for a 1000-line flood"
    );
}
