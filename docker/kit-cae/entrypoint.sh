#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

set -euo pipefail

export HOME=/home/ubuntu
export USER=ubuntu
export XDG_CACHE_HOME=/home/ubuntu/.cache
export XDG_CONFIG_HOME=/home/ubuntu/.config
export XDG_DATA_HOME=/home/ubuntu/.local/share

KIT_CAE_USER_GUIDE_DATA_URL=${KIT_CAE_USER_GUIDE_DATA_URL:-"https://d4i3qtqj3r0z5.cloudfront.net/Kit_CAE_user_guide_data%4010010.zip"}
KIT_CAE_USER_GUIDE_DATA_DIR=${KIT_CAE_USER_GUIDE_DATA_DIR:-/home/ubuntu/kit-cae-user-guide-data}
KIT_CAE_USER_GUIDE_DATA_SENTINEL=${KIT_CAE_USER_GUIDE_DATA_SENTINEL:-"${KIT_CAE_USER_GUIDE_DATA_DIR}/.Kit_CAE_user_guide_data_10010.complete"}

write_omniverse_toml() {
  local bookmark_name=$1
  local bookmark_value=$2

  mkdir --parents "${HOME}/.nvidia-omniverse/config"
  cat > /home/ubuntu/.nvidia-omniverse/config/omniverse.toml <<EOF
[bookmarks]
"${bookmark_name}" = "${bookmark_value}"
EOF
}

ensure_writable_dir() {
  local dir=$1

  mkdir --parents "${dir}"
  if [ ! -w "${dir}" ]; then
    echo "Directory is not writable by $(id -un): ${dir}" >&2
    echo "If this is a reused Docker volume, recreate it with: docker compose down -v" >&2
    exit 1
  fi
}

hydrate_user_guide_data() {
  local url=$1
  local data_dir=$2
  local sentinel=$3
  local tmp_dir
  local archive
  local unpack_dir

  ensure_writable_dir "${data_dir}"
  if [ -f "${sentinel}" ]; then
    echo "Kit-CAE user guide data already exists in ${data_dir}; skipping download."
    return
  fi

  tmp_dir=$(mktemp -d)
  trap 'rm -rf "${tmp_dir:-}"' RETURN
  archive="${tmp_dir}/Kit_CAE_user_guide_data.zip"
  unpack_dir="${tmp_dir}/unpacked"

  echo "Downloading Kit-CAE user guide data from ${url}"
  mkdir --parents "${unpack_dir}"
  curl -fL --retry 5 --retry-delay 2 --connect-timeout 20 -o "${archive}" "${url}"
  unzip -q "${archive}" -d "${unpack_dir}"
  cp -a "${unpack_dir}/." "${data_dir}/"
  touch "${sentinel}"
}

EXTRA_ARGS=()
if [ -n "${NVDA_KIT_ARGS:-}" ]; then
  read -r -a EXTRA_ARGS <<< "${NVDA_KIT_ARGS}"
fi

NUCLEUS_ARGS=()
if [ -n "${NVDA_KIT_NUCLEUS:-}" ]; then
  write_omniverse_toml \
    "${NVDA_KIT_BOOKMARK_NAME:-${NVDA_KIT_NUCLEUS}}" \
    "${NVDA_KIT_BOOKMARK_VALUE:-omniverse://${NVDA_KIT_NUCLEUS}}"
  NUCLEUS_ARGS=("--/ovc/nucleus/server=${NVDA_KIT_NUCLEUS}")
fi

APP_FILE=${KIT_APP_FILE:-omni.cae_streaming.kit}
KIT_FILE="/app/apps/${APP_FILE}"

CMD=(
  "/app/kit/kit"
  "${KIT_FILE}"
  "--no-window"
  "--enable omni.cae.delegate.vtk"
  "--enable omni.cae.importer.vtk"
  "--ext-folder"
  "/home/ubuntu/.local/share/ov/data/exts/v2"
  "${EXTRA_ARGS[@]}"
  "${NUCLEUS_ARGS[@]}"
)

if [ "${OM_KIT_VERBOSE:-0}" = "1" ]; then
  echo "==== Print out kit config ${KIT_FILE} for debugging ===="
  sed -n '1,240p' "${KIT_FILE}"
  echo "==== End of kit config ${KIT_FILE} ===="
fi

ensure_writable_dir "${XDG_CACHE_HOME}"
ensure_writable_dir "${XDG_CACHE_HOME}/ov"
ensure_writable_dir "${XDG_CACHE_HOME}/warp"
ensure_writable_dir "${XDG_DATA_HOME}"
ensure_writable_dir "${XDG_DATA_HOME}/ov"
ensure_writable_dir "${HOME}/.nvidia-omniverse/config"
hydrate_user_guide_data \
  "${KIT_CAE_USER_GUIDE_DATA_URL}" \
  "${KIT_CAE_USER_GUIDE_DATA_DIR}" \
  "${KIT_CAE_USER_GUIDE_DATA_SENTINEL}"

echo "Starting Kit with ${CMD[*]} $*"
exec "${CMD[@]}" "$@"
