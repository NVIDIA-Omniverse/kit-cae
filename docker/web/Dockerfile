# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

FROM nvcr.io/nvidia/base/ubuntu:22.04_20240212

WORKDIR /app

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        dnsutils \
        git \
        jq && \
    curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends nodejs && \
    apt-get clean autoclean && \
    rm -rf /var/lib/apt/lists/*

RUN echo 'registry=https://registry.npmjs.org/' > /root/.npmrc && \
    echo '@nvidia:registry=https://edge.urm.nvidia.com/artifactory/api/npm/omniverse-client-npm-local/' >> /root/.npmrc

RUN npm install -g @nvidia/create-ov-web-sample@1.13.4 --ignore-scripts

RUN create-ov-web-sample \
      --name="kit-cae-web-viewer" \
      --sample=viewport-only \
      --external \
      --outputDir=/app

WORKDIR /app/kit-cae-web-viewer

RUN sed -i "s/export default defineConfig({/export default defineConfig({\n  base: '\/viewer\/',\n  server: {\n    allowedHosts: true\n  },/" vite.config.ts && \
    sed -i "s/signalingServer: '127.0.0.1',/signalingServer: __SIGNALING_SERVER__,\n            signalingPort: __SIGNALING_PORT__,\n            mediaServer: '__MEDIA_SERVER__',\n            forceWSS: __FORCE_WSS__,/" src/main.ts && \
    sed -i "s/<title>.*<\/title>/<title>Kit CAE Web Viewer<\/title>/" index.html

RUN npm install --ignore-scripts

COPY entrypoint.sh /entrypoint.sh

ENTRYPOINT ["bash", "/entrypoint.sh"]
