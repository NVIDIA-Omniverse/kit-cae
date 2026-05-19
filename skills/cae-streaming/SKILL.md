---
name: cae-streaming
description: >
  Run Kit-CAE as a WebRTC-streamed application that remote clients can view and
  control via data-channel messages. Provides a streaming-enabled .kit dep set,
  the wire-protocol reference, the messaging-API reference, a small reusable
  Python helper (`OmniverseAPI`), and a runnable reference handler
  (`load_scene` + scenes registry) that demonstrates the request/response
  pattern end-to-end. Triggers on requests involving "stream Kit-CAE", "WebRTC",
  "remote viewer", "browser viewport", "application streaming", "data-channel
  messages", or "remote control of Kit-CAE".
depends:
  - cae-core
version: "2.1.0"
metadata:
  author: "NVIDIA"
  tags:
    - kit-cae
    - cae
    - streaming
    - webrtc
    - remote
    - data-channel
    - omniverse
---

# CAE Streaming - Remote-Controlled Kit-CAE

Stand up Kit-CAE as a streaming application that a remote client (browser,
agent, dashboard, notebook) can both **watch** (live WebRTC viewport) and
**drive** (request/response messages over the data channel).

The skill ships:

- A streaming-enabled `.kit` dep set (`omni.kit.livestream.messaging` + a
  `hideUi` setting) you can point any custom `.kit` at.
- `scripts/launch_streaming.sh` — wraps `repo.sh launch` with sensible defaults.
- `scripts/omniverse_api.py` — a small `@api.request` / `@api.signal`
  decorator framework that handles sync AND async handlers.
- `scripts/serve.py` — a runnable reference handler (`load_scene` + scene
  registry) you can study or replace.
- The wire protocol (`references/streaming-protocol.md`) and the underlying
  Python API (`references/messaging-api.md`).

`load_scene` is **one example**. Real consumers will write their own handlers.

## Purpose

Run Kit-CAE as a WebRTC-streamed application that remote clients can view and control via data-channel messages. Provides the streaming-enabled `.kit` dep set, the wire-protocol and messaging-API references, a reusable Python helper (`OmniverseAPI`), and a runnable reference handler (`load_scene` + scenes registry).

## Prerequisites

`cae-core` (loaded automatically). A built Kit-CAE checkout that includes `omni.kit.livestream.messaging`. See `## Standing up the streaming server` for the launch command and `references/streaming-protocol.md` for the wire format.

## Instructions

1. **Stand up the streaming server** — follow the launch command in `## Standing up the streaming server`; `scripts/launch_streaming.sh` wraps `repo.sh launch` with the right defaults.
2. **Run the reference handler** — invoke `scripts/serve.py` with the `run_script` pattern (`./repo.sh launch -n <kit> -- --exec scripts/serve.py`) to host the bundled `load_scene` handler; see `## Bundled example: load_scene + scenes registry`.
3. **Connect a client** — open the WebRTC viewer and connect over the data channel as described in `## Connecting a client`.
4. **Write your own handlers** — apply the `@api.request` and `@api.signal` patterns in `## Writing handlers` and the wire format in `references/streaming-protocol.md`.
5. **Troubleshoot** — consult `## Troubleshooting` for common failures (port collisions, async coroutine handling, scene-id mismatches).

## Examples

Bundled scripts:

| Script | Purpose |
|--------|---------|
| `scripts/launch_streaming.sh` | Wrap `repo.sh launch` with sensible streaming defaults. |
| `scripts/serve.py` | Runnable reference handler — `load_scene` + scenes registry. |
| `scripts/omniverse_api.py` | `@api.request` / `@api.signal` decorator helper for sync and async handlers. |

End-to-end runnable example: `scripts/serve.py` paired with the scenes registry in `scenes.yaml`. See `## Writing handlers` for the request/response pattern and `references/messaging-api.md` for the full helper API.

## Limitations

Streaming uses WebRTC — clients need a browser or WebRTC client. `LivestreamMessaging` is per-app, not per-session, so multiple clients see the same scene. Time-varying playback inherits the same constraints as `cae-visualization`.

## When to use

- A user / agent needs to interact with Kit-CAE from a process that isn't Kit
  (browser, REST service, RPC client, ML pipeline, dashboard).
- You need a long-lived Kit-CAE session that responds to commands rather than
  one-shot scripts that exit.
- You're building any client UI that wants a live Kit viewport plus
  programmatic control.

For **headless capture** (clean PNG/EXR/MP4 outputs, no streaming), stay with
`cae-capture`. This skill is the live + interactive path, not the offline
render path.

## Architecture

```
  Remote client (browser, agent, …)        Kit-CAE host
  ┌──────────────────────────────┐         ┌──────────────────────────────┐
  │ AppStreamer (WebRTC client)  │ video   │ omni.kit.livestream.app      │
  │  - <video> / <audio> sinks   │ ◄────── │ omni.kit.livestream.webrtc   │
  │                              │         │                              │
  │ data-channel:                │ JSON    │ omni.kit.livestream.messaging│
  │  - send  ({type}_request)    │ ◄────── │ scripts/serve.py:            │
  │  - recv  ({type}_response)   │ ──────► │   @api.request handlers      │
  │  - recv  ({signal})          │ ◄────── │   @api.signal pushes         │
  └──────────────────────────────┘         └──────────────────────────────┘
```

## Standing up the streaming server

Minimal launch:

```bash
cd <kit-cae-dir>
bash skills/cae-streaming/scripts/launch_streaming.sh
```

What it does:

1. Launches a streaming Kit-CAE app. The launcher defaults to
   `omni.cae_vtk_streaming.kit` (streaming + VTK extensions). Override via
   `CAE_STREAMING_KIT=<other.kit>` for the slim non-VTK build
   (`omni.cae_streaming.kit`) or any other custom app.
2. Both shipping streaming apps depend on `kit_cae_streaming` (the streaming
   base template), which lists `omni.kit.livestream.messaging` and sets
   `[settings.app.window] hideUi = true`. The data-channel `observe` API and
   the chrome-free viewport are both wired in via that template — no special
   launch flags required.
3. Runs `scripts/serve.py` via `--exec`. That script registers handlers and
   loops forever, awaiting messages from connected clients.

After the launch, look for these log lines:

```
[ovapi] registered request handler: load_scene_request
[ovapi] registered request handler: list_scenes_request
[cae-streaming] viewport-only: hideUi=True via carb.settings
[omni.kit.livestream.app.plugin] Started primary stream server on signal port 49100 …
```

Now any compatible client can connect.

## Connecting a client

Any client speaking `omni.kit.livestream.*`'s WebRTC + data channel works.
Compatible clients include:

- A JavaScript browser app using `@nvidia/omniverse-webrtc-streaming-library`.
- A Python client over `aiortc`.
- A native Kit window pointed at the same signaling endpoint.

Whatever the language, the JSON wire format is the same — see
`references/streaming-protocol.md` for the envelopes, naming convention, and
correlation rules.

A minimal browser snippet (JS):

```js
import { AppStreamer, StreamType } from "@nvidia/omniverse-webrtc-streaming-library";
await AppStreamer.connect({
  streamSource: StreamType.DIRECT,
  streamConfig: {
    signalingServer: "kit-host", signalingPort: 49100,
    videoElementId: "kit-video", audioElementId: "kit-audio",
    onCustomEvent: (msg) => console.log("from kit:", msg),
  },
});
AppStreamer.sendMessage(JSON.stringify({
  event_type: "load_scene_request",
  payload: { id: 1, scene_id: "static_mixer" },
}));
```

## The wire protocol

Two message families:

- **request / response** — client asks Kit to do something; correlation by
  `id` field. Naming convention: `<type>_request` → `<type>_response`
  (underscore — see the colon-default warning in `references/messaging-api.md`).
- **signal** — one-way push from Kit to client; no `id`, no reply expected.

Full spec, including the JSON envelopes, the bundled-example operations, and
the error encoding: `references/streaming-protocol.md`.

## Writing handlers

Use `scripts/omniverse_api.py`'s `@api.request` / `@api.signal` decorators
(handles sync and async); drop down to `omni.kit.livestream.messaging`
directly via `references/messaging-api.md` when you need finer control.

### Request/response with `OmniverseAPI`

```python
from omniverse_api import OmniverseAPI, exclusive
from omni.cae.data.commands import execute_command
from omni.cae.importer.cgns import import_to_stage
from omni.cae.testing import wait_for_update

api = OmniverseAPI()

@api.request
@exclusive  # optional: drop concurrent calls if mutating shared state
async def load_scene(scene_id: str = "", **_):
    if not scene_id:
        return {"ok": False, "error": "scene_id required"}
    await import_to_stage(f"/data/{scene_id}.cgns", f"/World/{scene_id}")
    await wait_for_update()
    await execute_command("CreateCaeVizFaces",
                          dataset_path=f"/World/{scene_id}/Base/Zone",
                          prim_path=f"/World/CAE/Faces_{scene_id}")
    return {"ok": True, "scene_id": scene_id}
```

The decorator handles payload unpacking, async scheduling, response dispatch,
and exception → `{"ok": False, "error": "..."}` conversion (so Python
tracebacks never cross the wire).

### Pushing signals

```python
@api.signal
def progress(scene_id: str, percent: int):
    return {"scene_id": scene_id, "percent": percent}

# Calling progress(scene_id="static_mixer", percent=42) pushes a
# `progress_signal` event with that payload to all connected clients.

# Or fire a payload directly without a wrapper function:
api.dispatch_signal("notification", {"level": "info", "message": "Stage ready"})
```

### Two footguns to avoid

- **Long-lived `serve.py`.** Do **NOT** use the cae-core "Script shutdown
  (MANDATORY)" template (`app.post_quit()` + `os._exit(0)`). That's for
  one-shot capture; a streaming listener must run until the process is
  signaled.
- **Document new request types in `references/streaming-protocol.md`** so
  future readers inherit the contract.

## Bundled example: `load_scene` + scenes registry

The skill ships with one runnable handler so the end-to-end flow works the day
you check it out:

- `scripts/serve.py` registers `load_scene_request`, `list_scenes_request`,
  and `register_scene_request` handlers.
- A scene registry maps `scene_id` → `{ path, format, default_viz, ... }`.
  `serve.py` resolves it in this order: `scenes.json` (machine-friendly,
  no extra deps) → `scenes.yaml` (requires PyYAML) → an embedded
  `DEFAULT_SCENES` dict in `serve.py` itself (always works).

All three layers are illustrative — copy and adapt freely. The framework
itself does not require a registry; it exists because the bundled handler
reads it.

Field reference (when you do use the bundled handler):

| Key | Meaning |
|---|---|
| `path` | Filesystem path. `${KIT_CAE_DIR}` is expanded from the env var. |
| `format` | One of `cgns`, `vtk`, `npz`, `ensight`. Selects the importer. |
| `prim_path` | Where the data lands in the stage (e.g. `/World/<name>`). |
| `dataset_path` | Prim with the dataset (per-format; see `cae-core/references/formats.md`). |
| `default_viz` | Short label appended to the viz prim path. |
| `default_viz_command` | One of the `CreateCaeViz*` commands from `cae-visualization`. |
| `color_field` | Optional. Path to the field bound as the viz `colors` target. |

> **Verify dataset paths** with `cae-core/scripts/inspect_cgns.py` /
> `inspect_vtk.py` before declaring done — paths vary per file.

## Ports, hosts, firewalls

- WebRTC signaling: **49100** (configurable via
  `--/exts/omni.kit.livestream.app/primaryStream/streamPort=`).
- Media negotiation occurs over ranges 47995-48012 and 49000-49007 by default.
- Open these ports between client and Kit host (or use a TURN server if you
  cannot punch through NAT).
- HTTPS / WSS: not required for local testing. For remote deployment,
  terminate TLS in front of the signaling endpoint.

## Viewport-only streamed output

The streaming `.kit` template (`templates/kit_cae/kit_cae_streaming.kit`)
sets `[settings.app.window] hideUi = true`, which collapses Kit's editor
chrome (menubar, panels, toolbar, timeline) and leaves only the rendered
viewport. `serve.py` defensively re-applies the same setting at startup via
`carb.settings.set("/app/window/hideUi", True)` plus toggles for the in-viewport
gizmos (camera/light/HUD/grid/axis). Override at launch with
`--/app/window/hideUi=false` for local debugging.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| Client stuck "Connecting…" | Port 49100 blocked; verify with `ss -tlnp \| grep 49100`. |
| Client connects, viewport stays black | Stream is up but no scene loaded yet. Either trigger a handler or load a stage manually. |
| Request returns no reply | `serve.py` not running, handler not registered, or wire-name mismatch (`_request` vs `:request`). |
| Handler returns `unknown scene_id 'X'` | Bundled-example case: client `scene_id` not in the active registry. |
| Kit log: `omni.kit.livestream.messaging: extension not found` | Make sure the streaming `.kit` lists it explicitly under `[dependencies]`. |
| Multiple clients see the same scene | Expected — `LivestreamMessaging` is per-app, not per-session. |
| Camera frames an un-tessellated mesh | Add a 30-60 frame `app.next_update_async()` settle loop before calling `frame_prims`. |
| Python traceback never reaches client | Wrap handler body in try/except and return `{"ok": False, "error": ...}`. The `OmniverseAPI` framework does this for you. |
| `Object of type coroutine is not JSON serializable` | You're using `LivestreamMessaging.observe_and_dispatch` directly with an `async def` handler. The decorator does NOT await coroutines. Use `OmniverseAPI`'s `@api.request` decorator (handles both sync and async), or split the work into a sync receiver + manual `dispatch_event` from the async coroutine. |

## Cross-references

- `cae-core/SKILL.md` — preflight, Z-up, launch flags.
- `cae-core/references/kit-cae-api.md` — `import_to_stage`, `execute_command`
  signatures (used by handlers that touch the stage).
- `cae-core/references/formats.md` — per-format dataset prim paths.
- `cae-visualization/SKILL.md` — choosing visualization commands inside handlers.
- `cae-capture/SKILL.md` — for offline rendering, not live viewing.
- `references/streaming-protocol.md` — the wire spec your handlers must follow.
- `references/messaging-api.md` — `omni.kit.livestream.messaging` Python API
  (the lower layer `OmniverseAPI` wraps).
- `scripts/omniverse_api.py` — the `@api.request` / `@api.signal` framework.
- `scripts/serve.py` — runnable reference handler set.
