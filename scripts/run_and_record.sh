#!/usr/bin/env bash
# Run an Arena command in the Kit GUI and screen-record the display with ffmpeg (x11grab).
#
#   scripts/run_and_record.sh out.mp4 <command...>
#   scripts/run_and_record.sh ~/videos/freezer.mp4 python ~/ur5eredodo/arena/run_arena.py --viz kit ...
#
# Records the whole screen (default :0, set DISPLAY / SCREEN_SIZE / FPS to override), so put the Kit window
# in front and keep other windows out of the way. Arena's own --record_viewport_video does not work with the
# Kit visualizer here (env.render() returns None), hence this wrapper.
set -euo pipefail
out="$1"; shift
mkdir -p "$(dirname "$out")"
size="${SCREEN_SIZE:-$(xdpyinfo | awk '/dimensions:/ {print $2}')}"
ffmpeg -y -loglevel error -f x11grab -framerate "${FPS:-30}" -video_size "$size" -i "${DISPLAY:-:0}" \
  -c:v libx264 -preset ultrafast -pix_fmt yuv420p "$out" < /dev/null &
ffmpeg_pid=$!
echo "[run_and_record] recording $size from ${DISPLAY:-:0} to $out (pid $ffmpeg_pid); stops when the command exits" >&2
trap 'echo "[run_and_record] saved $out" >&2; kill -INT "$ffmpeg_pid" 2>/dev/null; wait "$ffmpeg_pid" 2>/dev/null || true' EXIT
"$@"
