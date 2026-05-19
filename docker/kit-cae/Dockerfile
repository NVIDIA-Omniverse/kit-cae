# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

ARG OV_BASE_IMAGE=nvcr.io/nvidia/omniverse/ov-base-ubuntu-22:2025.2.0

FROM ${OV_BASE_IMAGE} AS builder

USER root
WORKDIR /workspace

RUN if [ -f /etc/apt/sources.list.canonical ]; then \
        cp /etc/apt/sources.list.canonical /etc/apt/sources.list.d/canonical.list; \
    fi && \
    rm -f /root/.pip/pip.conf && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        git \
        git-lfs \
        jq \
        python3 \
        python3-pip \
        unzip && \
    apt-get clean autoclean && \
    rm -rf /var/lib/apt/lists/*

COPY . /workspace

ARG KIT_VERSION=110.1.1
ENV KIT_VERSION=${KIT_VERSION}

RUN ./repo.sh select_kit_version --version "${KIT_VERSION}" && \
    ./repo.sh build -rx && \
    ./repo.sh pip_download && \
    ./repo.sh build -r && \
    ./repo.sh select_kit_version --auto --no-use-symlinks && \
    ./repo.sh schema -c release --pluginfo-only --no-warn && \
    ./repo.sh package -c release && \
    install -d /package && \
    package_zip="$(python3 -c 'from pathlib import Path; p=sorted(Path("_build/packages").glob("kit-cae-fat*.release*.zip")) or sorted(Path("_build/packages").glob("kit-cae-fat*.zip")); print(p[-1] if p else "")')" && \
    test -n "${package_zip}" && \
    unzip -q "${package_zip}" -d /package

FROM ${OV_BASE_IMAGE}

USER root
WORKDIR /app

RUN if [ -f /etc/apt/sources.list.canonical ]; then \
        cp /etc/apt/sources.list.canonical /etc/apt/sources.list.d/canonical.list; \
    fi && \
    rm -f /root/.pip/pip.conf && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        unzip && \
    apt-get clean autoclean && \
    rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash ubuntu || true
RUN install -d /home/ubuntu/.config && \
    chown -R ubuntu:ubuntu /home/ubuntu/.config

COPY --from=builder --chown=root:root /package/ /app/
COPY --chown=root:root docker/kit-cae/entrypoint.sh /entrypoint.sh
COPY --chown=ubuntu:ubuntu tools/containers/stream_sdk.txt /home/ubuntu/.config/regkeys.txt

RUN install -d \
        /app/logs \
        /home/ubuntu/.cache/ov \
        /home/ubuntu/.cache/warp \
        /home/ubuntu/.config \
        /home/ubuntu/kit-cae-user-guide-data \
        /home/ubuntu/.local/share/ov \
        /home/ubuntu/.nvidia-omniverse/config && \
    chmod 555 /entrypoint.sh /home/ubuntu/.config/regkeys.txt && \
    chown -R ubuntu:ubuntu /app/logs && \
    chown -R ubuntu:ubuntu \
        /home/ubuntu/.cache \
        /home/ubuntu/.config \
        /home/ubuntu/kit-cae-user-guide-data \
        /home/ubuntu/.local \
        /home/ubuntu/.nvidia-omniverse

ENV HOME=/home/ubuntu \
    USER=ubuntu \
    XDG_CACHE_HOME=/home/ubuntu/.cache \
    XDG_CONFIG_HOME=/home/ubuntu/.config \
    XDG_DATA_HOME=/home/ubuntu/.local/share

USER ubuntu

LABEL kit_app_template="omni.cae_streaming.kit"
LABEL org.opencontainers.image.title="Kit-CAE Brev Launchable"

EXPOSE 47995-48012/udp \
       47995-48012/tcp \
       49000-49007/udp \
       49000-49007/tcp \
       8011/tcp \
       8111/tcp \
       49100/tcp

ENTRYPOINT ["/entrypoint.sh"]
