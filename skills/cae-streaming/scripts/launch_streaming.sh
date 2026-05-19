#!/usr/bin/env bash
# Launch a streaming Kit-CAE app with the bundled cae-streaming serve.py
# listener. Run from the kit-cae repo root, OR set KIT_CAE_DIR to point at it.
#
# Defaults to a headless launch (--no-window) with the title "Kit-CAE Streaming".
# Override via env:
#   CAE_STREAMING_KIT     Kit app name. Default omni.cae_vtk_streaming.kit
#                         (streaming + VTK). Set to omni.cae_streaming.kit for
#                         the slim non-VTK build, or any other custom .kit.
#   CAE_STREAMING_SCRIPT  default: skills/cae-streaming/scripts/serve.py
#   CAE_STREAMING_TITLE   default: "Kit-CAE Streaming"
#   CAE_STREAMING_WINDOW  set to 1 to disable --no-window for local debugging
#
# This wrapper deliberately does NOT pass the cae-core "clean capture" launch
# flags (asyncRendering=false etc.) — those are for offline render, not
# interactive streaming.

set -euo pipefail

KIT_CAE_DIR="${KIT_CAE_DIR:-$(pwd)}"
KIT_APP="${CAE_STREAMING_KIT:-omni.cae_vtk_streaming.kit}"
SERVE_SCRIPT="${CAE_STREAMING_SCRIPT:-skills/cae-streaming/scripts/serve.py}"
DISPLAY_TITLE="${CAE_STREAMING_TITLE:-Kit-CAE Streaming}"
WANT_WINDOW="${CAE_STREAMING_WINDOW:-0}"

cd "${KIT_CAE_DIR}"

if [[ ! -x "./repo.sh" ]]; then
    echo "error: ./repo.sh not found in ${KIT_CAE_DIR}" >&2
    echo "       cd into your kit-cae repo, or set KIT_CAE_DIR=<path>." >&2
    exit 1
fi

if [[ ! -f "${SERVE_SCRIPT}" ]]; then
    echo "error: ${SERVE_SCRIPT} not found relative to ${KIT_CAE_DIR}" >&2
    exit 1
fi

# Pass KIT_CAE_DIR through so serve.py can expand ${KIT_CAE_DIR} in scenes.yaml.
export KIT_CAE_DIR

# `--no-window` runs Kit headless — the only visible output is the streamed
# viewport. Set CAE_STREAMING_WINDOW=1 to opt out (useful for local debugging).
LAUNCH_ARGS=()
if [[ "${WANT_WINDOW}" != "1" ]]; then
    LAUNCH_ARGS+=("--no-window")
fi
LAUNCH_ARGS+=("--exec" "${SERVE_SCRIPT}")
# Display name overrides (visible in window title and AppStreamer logs).
LAUNCH_ARGS+=("--/app/name=${DISPLAY_TITLE}")
LAUNCH_ARGS+=("--/app/window/title=${DISPLAY_TITLE}")
LAUNCH_ARGS+=("--/app/titleVersion=")

exec ./repo.sh launch -n "${KIT_APP}" -- "${LAUNCH_ARGS[@]}"
