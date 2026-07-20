#!/usr/bin/env bash
# Umber live dev loop: watch the source tree and, on any change, rebuild and
# relaunch the app automatically so you never rebuild by hand.
#
# Usage:
#   scripts/dev.sh [file]        # watch + run (debug build), optionally open <file>
#   UMBER_RELEASE=1 scripts/dev.sh [file]   # same, but --release (slower rebuilds)
#
# Requires cargo-watch:  cargo install cargo-watch
#
# On every save under crates/, cargo-watch kills the running umber, rebuilds
# the changed crates, and starts a fresh instance. Rust is compiled, so this
# is a rebuild-and-restart loop, not in-process hot-swap — but it's automatic.
set -euo pipefail
cd "$(dirname "$0")/.."

# cargo install drops binaries in ~/.cargo/bin, which isn't always on PATH.
export PATH="$HOME/.cargo/bin:$PATH"

if ! command -v cargo-watch >/dev/null 2>&1; then
    echo "cargo-watch not found. Install it with:" >&2
    echo "    cargo install cargo-watch" >&2
    exit 1
fi

PROFILE_ARGS=()
if [ "${UMBER_RELEASE:-0}" = "1" ]; then
    PROFILE_ARGS+=(--release)
fi

# -w crates: only watch source. -c: clear screen each run. --why: print what
# changed. -x run: rebuild + relaunch umber, forwarding any file argument.
exec cargo watch -c --why -w crates \
    -x "run ${PROFILE_ARGS[*]} -p umber -- ${*:-}"
