#!/bin/sh
# Runs a captured frame again (docs/replay.md) on macOS: builds VIctory for
# Windows and runs `victory replay` under CrossOver Wine with Apple's
# D3DMetal, in a Wine prefix of its own, never the one a title runs in.
#   tools/macos/replay.sh <capture folder> [replay arguments]
# Paths among the arguments (the capture, --shaders, --retranslate, --image)
# are macOS paths; they are handed to Wine as Z: paths.
# WINE defaults to the Wine that tools/macos/build-crossover-wine.sh builds,
# WINEPREFIX to target/wineprefix-replay. VICTORY_DIAGNOSTICS=1 runs the
# build with the diagnostics feature (the one a recording run uses). The
# VICTORY_* switches of the translator are passed on (for --retranslate).
# The GPU is shared: do not run this while a title runs.
set -e
cd "$(dirname "$0")/../.."
ROOT="$PWD"
export PATH="/opt/homebrew/opt/rustup/bin:$PATH"
WINE="${WINE:-$HOME/Library/Caches/victory-macos/target/wine-cx/bin/wine}"
[ -x "$WINE" ] || { echo "no Wine at $WINE: run tools/macos/build-crossover-wine.sh first" >&2; exit 1; }
[ $# -ge 1 ] || { echo "usage: tools/macos/replay.sh <capture folder> [replay arguments]" >&2; exit 1; }
if [ "${VICTORY_DIAGNOSTICS:-0}" = 1 ]; then
    cargo build --quiet --release -p victory --features diagnostics --target x86_64-pc-windows-gnullvm
else
    cargo build --quiet --release -p victory --target x86_64-pc-windows-gnullvm
fi
export WINEPREFIX="${WINEPREFIX:-$ROOT/target/wineprefix-replay}"
export WINEDEBUG="${WINEDEBUG:--all}"
export WINEDLLOVERRIDES="mscoree,mshtml=;winedbg.exe=d;d3d12,d3d12core,dxgi=b${WINEDLLOVERRIDES:+;$WINEDLLOVERRIDES}"
export ROSETTA_ADVERTISE_AVX=1
# An argument that is an absolute macOS path becomes a Z: path.
EXE="$ROOT/target/x86_64-pc-windows-gnullvm/release/victory.exe"
set -- replay "$@"
n=$#
while [ "$n" -gt 0 ]; do
    a=$1
    shift
    case "$a" in
        /*) a="Z:$(printf '%s' "$a" | tr / '\\')" ;;
    esac
    set -- "$@" "$a"
    n=$((n - 1))
done
exec "$WINE" "$EXE" "$@"
