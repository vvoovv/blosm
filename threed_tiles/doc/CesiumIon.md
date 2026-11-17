# Google 3D Tiles through Cesium ion

With **Source: Google**, **Access via** selects **Google** (the default for new
and existing scenes) or **Cesium ion**. The direct path uses the Google 3D Tiles
key. The ion path requires only the Cesium ion access token. Press **Import**
to resolve access and import tiles using the normal area, detail, joining,
progressive import and concurrent download settings.

Credentials are in addon preferences under **3D Tiles:**. The ion token is
visible, as is the Google key. ArcGIS and Mapbox credentials are grouped under
**Overlays:**. **Get it!** opens the [ion tokens page](https://ion.cesium.com/tokens).
The ion token needs `assets:read` permission for Google Photorealistic 3D Tiles,
asset **2275207**, which must be available in the account's **My Assets**. Tokens
used by desktop clients should not have web-only Allowed URLs restrictions.
See [Cesium's token guide](https://cesium.com/learn/ion/cesium-ion-access-tokens/).

## Import pipeline

`BLOSM_OT_ImportData.import3dTiles()` selects `CesiumIonManager` for ion access.
`BlenderApp.init3dTiles()` validates the ion token without requiring a direct
Google key. `CesiumIonManager.iterRender()` resolves access before calling the
normal traversal or preparing a renderer. Resolution failures and cancellation
during resolution create no scene data. Both progressive and blocking imports
use the shared download helper, including single-download imports.

`threed_tiles/cesium_ion.py` has no Blender dependency. `resolve_asset()` yields
one authenticated `GET /v1/assets/2275207/endpoint` request and returns an
`AssetEndpoint`. External 3D Tiles responses use `options.url`; hosted responses
use `url` and the returned asset token. The user ion API token is sent only to
the endpoint service. See [Cesium's IonResource implementation](https://github.com/CesiumGS/cesium/blob/main/packages/engine/Source/Core/IonResource.js).

The manager derives its server and base paths from the resolved URL, resolves
relative/root-relative nested content paths, and merges provider query
parameters with the session parameters supplied by Google. Provider query
parameters and returned authorization headers are inherited only by requests
to the resolved origin. Absolute content URLs on other origins retain their
own parameters and receive no inherited credentials. See
[Google's renderer guide](https://developers.google.com/maps/documentation/tile/create-renderer).

Endpoint credits (including image alt/title labels) and tile copyrights are
stored with the import's collection reference in the scene's attribution
registry. The Copyright panel combines all live imports without replacing
earlier credits; see [persistent attribution management](Attribution.md).
Provider credentials remain in
memory for the import and are cleared when traversal completes or is cancelled.
They are never stored in addon preferences or scene properties. Diagnostics
redact credentials; malformed endpoint errors do not echo the response.
Requests verify TLS certificates and respect Blender's Allow Online Access
setting. Endpoint resolution has a 30-second deadline; the endpoint parser
rejects responses larger than 1 MiB. The shared helper's normal download limits
apply to tiles and manifests.

The normal import success/error messages replace the earlier resolution-only
message. Cancelling a progressive import retains tiles already imported and
their scene origin. There is no fallback to the direct Google key. Each new
import resolves access again; automatic credential/session renewal within an
import and selection of other ion assets are outside this implementation.

## Verification

- `tests/test_cesium_ion.py`: pure endpoint validation, credential separation,
  redaction and localhost HTTP through the shared helper.
- `tests/check_cesium_ion.py`: Blender registration and preference visibility,
  direct Google regression, actual GLB imports in blocking/progressive modes,
  origin-scoped credentials, errors, cancellation and runtime credential cleanup.
- `tests/check_cesium_ion_ui.py`: real Blender timers and HTTP helper, nested GLB
  imports with key/session propagation, blocking and progressive execution,
  final joining, endpoint errors and cancellation before/after the first mesh.

Run Blender checks in a separate factory-startup process with an isolated user
profile and an external artifact directory. No real credential is required.
