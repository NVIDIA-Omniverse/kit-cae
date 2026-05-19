#!/bin/sh
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

set -eu

if [ ! -f /etc/nginx/tls/cert.pem ] || [ ! -f /etc/nginx/tls/key.pem ]; then
  openssl req \
    -x509 \
    -newkey rsa:4096 \
    -keyout /etc/nginx/tls/key.pem \
    -out /etc/nginx/tls/cert.pem \
    -sha256 \
    -days 3650 \
    -nodes \
    -subj "/C=US/ST=CA/L=Santa Clara/O=NVIDIA/OU=Kit-CAE/CN=localhost"
fi

exec nginx -g "daemon off;"
