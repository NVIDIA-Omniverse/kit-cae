# Kit-CAE Streaming Wire Protocol

Browser frontends and Kit-CAE listeners exchange JSON messages over the WebRTC
data channel exposed by `@nvidia/omniverse-webrtc-streaming-library` (browser)
and `omni.kit.livestream.messaging` (Kit). This document is the source of truth
for the message shapes.

## Message families

There are two:

1. **Request / response** — browser asks Kit to do something and awaits a reply.
2. **Signal** — one-way push from Kit to browser, no reply expected.

## Naming convention

For every browser-initiated operation `foo`:

- Browser sends `event_type: "foo_request"`.
- Kit replies on `event_type: "foo_response"`.

Both ends spell the names with **underscores**, not colons. The default
`omni.kit.livestream.messaging` decorator uses `"{event_name}:response"` —
**always pass `response_name=` explicitly** to keep the underscore convention.

## Correlation id

Every request carries a numeric `id` in its payload. Kit must echo the same
`id` back in the response payload so the client can match request to reply.
The bundled `OmniverseAPI` framework (and `LivestreamMessaging.observe_and_dispatch`)
does this echo automatically.

**Use `id >= 1`.** `omni.kit.livestream.messaging-1.3.0`'s `_pack_messages`
treats falsy ids as missing and replaces them with a server-generated UUID
(`payload.pop("id") or _new_id()`). A client that sends `id: 0` will get a
UUID back and lose correlation. Start request ids at 1.

## Request envelope (browser → Kit)

```jsonc
{
  "event_type": "load_scene_request",
  "payload": {
    "id": 1,                  // assigned by the browser; opaque on Kit side
    "scene_id": "static_mixer"
    // …other request-specific fields…
  }
}
```

## Response envelope (Kit → client)

In `omni.kit.livestream.messaging-1.3.0`+ the `id` is at the **top level** of
the envelope; older 1.2.x put it inside `payload`. A robust client reads both
(`msg.id ?? msg.payload?.id`):

```jsonc
{
  "event_type": "load_scene_response",
  "id": 1,                    // top-level in 1.3.0+ (echoed from request)
  "payload": {
    "response": {             // handler return value
      "ok": true,
      "scene_id": "static_mixer",
      "prim_path": "/World/static_mixer",
      "viz_prim_path": "/World/static_mixer/CaeViz/Faces"
    }
  }
}
```

Clients pull the handler return value from `payload.response`.

Errors are encoded inside `response`, not as a separate envelope:

```jsonc
{
  "event_type": "load_scene_response",
  "payload": {
    "id": 1,
    "response": { "ok": false, "error": "unknown scene_id 'foo'" }
  }
}
```

## Signal envelope (Kit → browser, push)

Use signals for state pushes that don't correspond to a request. There is no
`id`, no `response`.

```jsonc
{
  "event_type": "notification_signal",
  "payload": { "level": "info", "message": "Scene loaded" }
}
```

## Bundled-example operations

The next four operations are implemented by the example `serve.py` shipped
with this skill — they are NOT framework requirements. Your own protocol can
omit them entirely or define wholly different operations (`set_camera`,
`query_field`, `apply_transform`, etc.).

### `load_scene` (request / response)

Load a sample scene by id, create a default visualization, frame the camera.

| Field | Direction | Type | Notes |
|---|---|---|---|
| `scene_id` | request | string | Key into `scenes.yaml`. |
| `ok` | response | bool | Always present. |
| `scene_id` | response | string | Echoed for caller convenience. |
| `prim_path` | response | string | Where the data was imported (e.g. `/World/static_mixer`). |
| `viz_prim_path` | response | string | Where the default visualization landed. |
| `error` | response | string | Present only when `ok` is `false`. |

### `notification_signal` (push)

Free-form info/warning/error message Kit can push to the browser.

| Field | Direction | Type | Notes |
|---|---|---|---|
| `level` | push | `"info" \| "warning" \| "error"` | UI tone. |
| `message` | push | string | Human-readable text. |

### `list_scenes` (request / response)

Fetch the current scene registry. Clients call this on connect to enumerate
available scenes without hardcoding scene ids.

| Field | Direction | Type | Notes |
|---|---|---|---|
| (no fields) | request | — | Empty payload. |
| `ok` | response | bool | Always present. |
| `scenes` | response | `SceneSummary[]` | Each entry: `{ id, label, description?, format? }` |
| `error` | response | string | Present only when `ok` is `false`. |

### `register_scene` (request / response)

Register (or replace) a scene at runtime. The intended use case is an agent
that programmatically generates a new visualization and POSTs the result
here so frontends pick it up. Triggers a `scenes_changed_signal` push to all
connected clients.

| Field | Direction | Type | Notes |
|---|---|---|---|
| `scene_id` | request | string | Canonical key. Replaces any existing entry with the same id. |
| `entry` | request | object | Full scene record; required keys: `path`, `format`, `prim_path`, `dataset_path`, `default_viz_command`. Optional: `label`, `description`, `default_viz`, `default_viz_kwargs`, `color_field`, `schema_type`. |
| `ok` | response | bool | Always present. |
| `scene_id` | response | string | Echoed. |
| `scenes` | response | `SceneSummary[]` | Updated full registry. |
| `error` | response | string | Present only when `ok` is `false`. |

### `scenes_changed_signal` (push)

Push notification that the scene registry was modified (typically after a
`register_scene` from another agent). Clients should treat the included
`scenes` list as the new truth.

| Field | Direction | Type | Notes |
|---|---|---|---|
| `scenes` | push | `SceneSummary[]` | Full updated registry — clients can use it directly without a separate `list_scenes` call. |


## Agent-driven registry

`register_scene` + `scenes_changed_signal` lets a non-client agent add scenes
that connected clients pick up automatically. The bundled `serve.py` mutates
an in-memory dict; persist to `scenes.json` from your handler if you need
restart durability.

## Implementation pointers

- Kit side: `scripts/serve.py` registers handlers via the bundled
  `OmniverseAPI` (`@api.request` / `@api.signal` decorators). The framework
  is in `scripts/omniverse_api.py` and wraps the lower-level
  `omni.kit.livestream.messaging` API.
- Client side: any WebRTC client speaking the same JSON envelope works. A
  minimal JS client uses `@nvidia/omniverse-webrtc-streaming-library`'s
  `AppStreamer.connect` / `sendMessage` / `onCustomEvent`. A Python client
  can use `aiortc` directly with the same JSON envelope.
