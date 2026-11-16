# Google 3D Tiles access through Cesium ion

This first stage resolves access only. It does not download root manifests or
tile content, create Blender objects, or implement general Cesium ion importing.

With **Source: Google**, **Access via** selects **Google** (the default for new
and existing scenes) or **Cesium ion**. The direct Google path continues to use
the Google 3D Tiles key. The ion path requires only the Cesium ion access token.
Credentials are in addon preferences under **3D Tiles:**; ArcGIS and Mapbox
credentials are grouped separately under **Overlays:**. **Get it!** opens the
[ion tokens page](https://ion.cesium.com/tokens).

An ion token needs `assets:read` permission for Google Photorealistic 3D Tiles,
asset **2275207**, which must be available in the account's **My Assets**. Tokens
used by desktop clients should not have web-only Allowed URLs restrictions.
See [Cesium's token guide](https://cesium.com/learn/ion/cesium-ion-access-tokens/).

## Runtime boundary

`BLOSM_OT_ImportData.import3dTiles()` dispatches ion access before constructing
the tile manager or renderer. `resolveIonAccess()` reads and validates the
credential and starts an `IonAccessResolution` job. Interactive Blender runs
this job through the existing cooperative controller, even when progressive
tile importing is disabled. Background and command-line imports use the same
job and HTTP transport with a blocking driver.

`threed_tiles/cesium_ion.py` has no Blender dependency. `resolve_asset()` yields
a single authenticated `GET /v1/assets/2275207/endpoint` request and returns an
`AssetEndpoint`. External 3D Tiles responses use `options.url`; hosted responses
use `url` and the returned asset token. The descriptor retains attribution and
keeps the ion API token separate from provider credentials. These response
shapes are described in [Cesium's endpoint explanation](https://community.cesium.com/t/cesium-api-endpoint-for-google-3d-tiles-asset-provides-an-incorrect-url/29510).

Requests use the existing download helper, verified TLS, a 30-second resource
deadline and a 1 MiB response limit. Cancellation and Blender's Allow Online
Access setting use the existing controller/transport lifecycle. Diagnostics
redact authentication values, and malformed-response errors do not echo data.
The per-operation descriptor is discarded after resolution and is never written
to preferences, scenes, files or logs.

Successful resolution reports:

> Cesium ion access resolved successfully. Tile import through Cesium ion is not implemented yet.

There is no fallback to the direct Google key when ion access fails. Later work
can consume the descriptor before constructing `BaseManager`, whose constructor
derives its server/base paths from the root URL. Actual ion importing, provider
session renewal, attribution display and additional assets remain deferred.

## Focused verification

- `tests/test_cesium_ion.py`: endpoint response validation, credential separation,
  redaction and real localhost HTTP through the shared helper.
- `tests/check_cesium_ion.py`: Blender registration, panel/preferences layout,
  defaults, credential checks, provider routing, cancellation and no scene data.
- `tests/check_cesium_ion_ui.py`: real Blender timers, responsive endpoint-only
  resolution, error reporting and cancellation with the shared HTTP helper.

Run Blender checks in a separate factory-startup process with an isolated user
profile and an external artifact directory. No real credential is required.
