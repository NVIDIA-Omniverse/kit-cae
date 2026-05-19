#!/usr/bin/env python3
"""Minimal warmup script for Kit-CAE shader cache compilation.

Launches Kit, waits for initialization, then exits cleanly.
Used during preflight to front-load the first-run shader compile cost.

Usage (from the kit-cae repo root):
    ./repo.sh launch -n omni.cae.kit -- --exec scripts/warmup.py --no-window
"""

import asyncio
import os

import omni.kit.app


async def main():
    app = omni.kit.app.get_app()

    # Wait enough frames for Kit to initialize and compile shaders
    for i in range(120):
        await app.next_update_async()

    print("WARMUP_COMPLETE")
    app.post_quit()
    for _ in range(10):
        await app.next_update_async()
    os._exit(0)


if __name__ == "__main__":
    asyncio.ensure_future(main())
