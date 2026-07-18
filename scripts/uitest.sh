#!/usr/bin/env bash
# Umber visual test harness: run the editor headless (Xvfb + lavapipe),
# drive it with xdotool, and capture screenshots for inspection.
#
# Usage:
#   scripts/uitest.sh start [file]     # boot Xvfb + umber (release build)
#   scripts/uitest.sh shot <name>      # screenshot -> target/uitest/<name>.png
#   scripts/uitest.sh key <keysym>     # send a key chord (xdotool syntax)
#   scripts/uitest.sh type <text>      # type literal text
#   scripts/uitest.sh click <x> <y>    # left-click at coordinates
#   scripts/uitest.sh midclick <x> <y> # middle-click at coordinates
#   scripts/uitest.sh move <x> <y>     # hover (for hover-effect shots)
#   scripts/uitest.sh stop             # tear everything down
#
# The display is :77, window 1280x800. State under target/uitest/.

set -euo pipefail
cd "$(dirname "$0")/.."

DISPLAY_NO=":77"
OUT="target/uitest"
mkdir -p "$OUT"

case "${1:-}" in
start)
    "$0" stop >/dev/null 2>&1 || true
    Xvfb "$DISPLAY_NO" -screen 0 1280x800x24 &
    echo $! > "$OUT/xvfb.pid"
    sleep 0.7
    # Force X11 (no Wayland socket) + software Vulkan (lavapipe) so the
    # harness never touches the real session or GPU.
    env -u WAYLAND_DISPLAY DISPLAY="$DISPLAY_NO" \
        VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.json \
        UMBER_DEBUG_TERM=1 \
        ./target/release/umber "${2:-Cargo.lock}" \
        > "$OUT/umber.log" 2>&1 &
    echo $! > "$OUT/umber.pid"
    sleep 2.5
    # No WM under Xvfb -> nothing assigns input focus; do it explicitly or
    # every synthetic key event lands nowhere.
    DISPLAY="$DISPLAY_NO" xdotool search --sync --name umber windowfocus --sync || true
    echo "started (display $DISPLAY_NO, log $OUT/umber.log)"
    ;;
focus)
    DISPLAY="$DISPLAY_NO" xdotool search --name umber windowfocus --sync
    ;;
shot)
    import -display "$DISPLAY_NO" -window root "$OUT/${2:?name}.png"
    echo "$OUT/${2}.png"
    ;;
key)
    DISPLAY="$DISPLAY_NO" xdotool key --clearmodifiers "${2:?keysym}"
    sleep 0.3
    ;;
type)
    DISPLAY="$DISPLAY_NO" xdotool type --delay 30 "${2:?text}"
    sleep 0.3
    ;;
click)
    DISPLAY="$DISPLAY_NO" xdotool mousemove "${2:?x}" "${3:?y}" click 1
    sleep 0.3
    ;;
midclick)
    DISPLAY="$DISPLAY_NO" xdotool mousemove "${2:?x}" "${3:?y}" click 2
    sleep 0.3
    ;;
move)
    DISPLAY="$DISPLAY_NO" xdotool mousemove "${2:?x}" "${3:?y}"
    sleep 0.3
    ;;
stop)
    [ -f "$OUT/umber.pid" ] && kill "$(cat "$OUT/umber.pid")" 2>/dev/null || true
    [ -f "$OUT/xvfb.pid" ] && kill "$(cat "$OUT/xvfb.pid")" 2>/dev/null || true
    rm -f "$OUT/umber.pid" "$OUT/xvfb.pid"
    echo "stopped"
    ;;
*)
    grep '^#' "$0" | head -16
    exit 1
    ;;
esac
