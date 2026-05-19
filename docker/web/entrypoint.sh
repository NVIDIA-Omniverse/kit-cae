#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

set -xeuo pipefail

ENV=${ENV:-brev}
if [ -z "${FORCE_WSS:-}" ]; then
  FORCE_WSS="window.location.protocol === 'https:'"
fi

case "${FORCE_WSS}" in
  true)
    DEFAULT_SIGNALING_PORT=443
    ;;
  false)
    DEFAULT_SIGNALING_PORT=80
    ;;
  *)
    DEFAULT_SIGNALING_PORT="window.location.protocol === 'https:' ? 443 : 80"
    ;;
esac
SIGNALING_PORT=${SIGNALING_PORT:-${DEFAULT_SIGNALING_PORT}}

get_ip() {
  case "${ENV}" in
    instance)
      nslookup "$(hostname)" | awk '/Address/ { address=$2 } END { print address }'
      ;;
    localhost)
      echo "127.0.0.1"
      ;;
    brev)
      curl -s https://icanhazip.com
      ;;
    *)
      echo "Env ${ENV} not understood" >&2
      exit 1
      ;;
  esac
}

main() {
  local ip
  ip=$(get_ip)

  SIGNALING_SERVER=${SIGNALING_SERVER:-window.location.hostname}
  MEDIA_SERVER=${MEDIA_SERVER:-${ip}}

  echo "Configuring stream settings:"
  echo "  SIGNALING_SERVER: ${SIGNALING_SERVER}"
  echo "  SIGNALING_PORT: ${SIGNALING_PORT}"
  echo "  MEDIA_SERVER: ${MEDIA_SERVER}"
  echo "  FORCE_WSS: ${FORCE_WSS}"

  sed -i "s/signalingServer: [^,]*/signalingServer: ${SIGNALING_SERVER}/" /app/kit-cae-web-viewer/src/main.ts
  sed -i "s/signalingPort: [^,]*/signalingPort: ${SIGNALING_PORT}/" /app/kit-cae-web-viewer/src/main.ts
  sed -i "s/mediaServer: '[^']*'/mediaServer: '${MEDIA_SERVER}'/" /app/kit-cae-web-viewer/src/main.ts
  sed -i "s/forceWSS: [^,]*/forceWSS: ${FORCE_WSS}/" /app/kit-cae-web-viewer/src/main.ts

  exec npm run dev -- --host 0.0.0.0
}

main "$@"
