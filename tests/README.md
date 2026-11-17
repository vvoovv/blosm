# Blosm release tests

Run these tests with Python 3.11+ and Blender 4.5+ from the main `blosm` checkout.
Run the examples from the repository root. All filesystem paths below are
relative to that root, except catalog entries, which are relative to their
configured fixture root. Adjust the example `../tools/` executable locations and
`../test-assets/blosm` asset directory to match your local setup.

The runner launches a separate Blender process and isolated preferences for each
suite. It never changes the normal Blender profile. GeoJSON is excluded.

## General, Base and Pro profiles

One catalog supplies three execution profiles:

| Profile | Runtime under test | Suites |
| --- | --- | --- |
| `general` (default) | Development checkout | All general headless suites, including Pro functionality |
| `base` | Installed Base candidate ZIP | Package and Base contracts, followed by every applicable common suite |
| `pro` | Installed Pro candidate ZIP | Package and Pro contracts, followed by the complete general suite |

Base excludes only `baking`, `export` and `pml`, recorded as `not_applicable` in a
full run. Imagery, Geometry Nodes/Buildify, GPX, terrain, OSM and ordinary 3D Tiles
are common features and must pass in Base. OSM and 3D Tiles retain their tests
with Single object both enabled and disabled.

Build profiles require a root-layout Blender extension ZIP containing
`blender_manifest.toml` and `__init__.py`. Use the distributables produced by the
sibling `../blosm_build` repository. The runner snapshots the candidate and records
its SHA-256, then installs that same ZIP in a fresh extension repository for each
Blender suite. Standalone Python tests extract the same payload. Runtime imports
from the development checkout are rejected; test helpers, catalogs and historical
comparison data remain in this checkout.

```powershell
& '../tools/python/python.exe' -B tests/run_release_tests.py `
  --blender '../tools/blender/blender.exe' --profile base `
  --candidate-zip ../artifacts/blosm_build/candidates/blosm-base-2.8.1.zip `
  --artifacts ../artifacts/blosm_release_tests/base_45

& '../tools/python/python.exe' -B tests/run_release_tests.py `
  --blender '../tools/blender/blender.exe' --profile pro `
  --candidate-zip ../artifacts/blosm_build/candidates/blosm-pro-2.8.1.zip `
  --assets ../test-assets/blosm `
  --artifacts ../artifacts/blosm_release_tests/pro_45
```

Adjust the candidate names to the version being qualified. Both profiles always
run the package and edition checks first, including when `--suite` selects only
one feature. A failed package/edition check blocks the remaining suites. A wrong
edition, unavailable requested suite, missing fixture or missing required Pro
asset directory is a failure, never a successful skip. Explicitly requesting a
Pro-only suite for Base is a configuration error. The `base` and `pro` dedicated
suites cannot be run under the other profile or `general`.

For development checks against the checkout:

```powershell
& '../tools/python/python.exe' -B `
  tests/run_release_tests.py `
  --blender '../tools/blender/blender.exe' `
  --assets ../test-assets/blosm `
  --osm-root ../osm_extracts `
  --fixture-root ../artifacts/blosm_release_tests/fixtures `
  --helsinki ../artifacts/direct_gltf_import/helsinki/1 `
  --artifacts ../artifacts/blosm_release_tests/my_run_45
```

Use a fresh output directory for each invocation. All logs, JSON reports,
generated inputs, exported textures, scenes and profiles go there. Do not add
those outputs to Git. Source and documentation stay in this repository.

Use repeated `--suite` arguments to select suites. Without them, the chosen
profile runs its complete headless selection; GUI suites are always explicit. Exit code
is nonzero if any selected suite fails or is blocked. `summary.json` records the
profile, full/selected scope, exclusions, commands, Blender version, source commit,
working-tree status, test-source hashes and durations.
It also hashes the tracked source tree and catalogs. Temporary directories are
isolated per suite. Windows Job Objects terminate descendant helper processes
when a suite exits, crashes or reaches its timeout.
New suites also write per-case `results.json`; existing Blender suites retain
their original JSON reports. Both exit code and report status are checked.

| Suite | Coverage |
| --- | --- |
| `preflight` | SHA-256 verification of selected OSM files, companions, real GPX and Buildify |
| `standalone` | Download/transport and extension-policy checks, Windows process-tree cleanup, profile routing and checkout isolation |
| `direct` | Existing GLB/B3DM correctness, historical/native glTF parity and no dataset caching |
| `tile_transforms` | Direct positioning versus the operator fallback: hierarchy/RTC, applied rotations, normals/UVs/colors, scale/shared-data/animation/modifier fallbacks, joining, rollback and context restoration; optional Helsinki fixtures |
| `operator` | Registered 3D Tiles operator, nested tilesets, partial/all-failed imports and reports |
| `progressive` | Stepped import, lifecycle/cancellation and error handling |
| `concurrent` | Blocking/progressive and serial/concurrent parity, **joining on/off**, retries and partial failures |
| `cesium_ion` | Google/ion access choices, visible token preferences, endpoint resolution, provider credentials/session queries, actual tile imports, failures and cancellation |
| `attribution` | Persistent per-collection credit registry, mixed access routes, duplicate removal, reference-only display, legacy text, save/reopen and partial-import behavior |
| `cesium_ion_ui` | Explicit GUI suite: real local HTTP endpoint/tile requests, responsive imports, joining, cancellation, provider-key/session propagation and attribution |
| `attribution_ui` | Explicit GUI suite: read-only panel callbacks, collection deletion, cleanup on import, Undo/Redo and retained partial-import credits |
| `osm` | All 12 core extracts, 2D/simple 3D, **Single object on/off**, world geometry, road/rail curves, disabled layers, repeated import and save/reopen |
| `osm_progressive` | Cooperative OSM sessions, real building/road parity, separate objects, per-object GN attributes, cancellation, owned bulk commits, RNG isolation; realistic forests and uncached asynchronous export with `--assets` |
| `osm_progressive_ui` | Explicit GUI suite: actual OSM operator/timers, incremental visibility, grouped Undo/Redo, user edits, pause, cancellation, target deletion, staged bake history and disable |
| `roofs` | Independent rectangular controls for 12 supported roof shapes plus unknown-shape fallback, both grouping states, explicit height and valid faces |
| `gpx` | Real track, curve/mesh, segments/edges, elevations, geographic control points, existing origin, terrain offsets/projection and curve width |
| `geometry_nodes` | Socket types, zero/false/empty values, instance isolation, attributes, discovery/errors and evaluated geometry |
| `buildify` | Supplied Buildify file on rectangle and **Karl Marx Allee**, face-level attributes, GN modifier, realized height and save/reopen |
| `terrain` | Generated big-endian HGT grids: quads/triangles, reduction, voids, elevation offset and two-axis seam topology |
| `overlay` | Four-color image stitching/orientation, packed output, decoded disk cache, temporary-image cleanup, URL expansion and 404 zoom fallback |
| `validation` | Missing files, setup script and data directory; disable/re-enable and subsequent import |
| `baking` | Actual CPU diffuse/emission bakes, separate/joined meshes, replace on/off, geometry/UVs, known pixel colors, packed save/reopen, channel selection, injected bake failure recovery and registered tile import → bake → GLB roundtrip |
| `export` | Realistic/import-for-export and experimental-mode routing, rectangle and **Karl Marx Allee**, both grouping states, real Pro assets, UV/material layout, generated textures, unchanged-cache reuse, linked-scene cleanup and GLB export/reimport |
| `pml` | Legacy parser function tests, including both definitions of the duplicate-named test |
| `package` | Candidate extension installation, metadata, actual installed-module path, distribution contents, OSM smoke import and disable/re-enable |
| `base` | Pro modules/setup scripts absent and unimportable; realistic mode rejected; Pro operators unregistered and uncallable; Pro UI controls hidden; highest Google detail rejected before download; stale Pro flags cannot enable the Pro renderer |
| `pro` | Pro modules/setup scripts present and importable, realistic mode available and current Pro operators registered; functionality uses the general suites |
| `ui` | Existing dedicated GUI process: progressive timers, Undo/Redo, cancellation, mode/scene changes and addon disable |

The opt-in `run_tiles_profile.py` launcher profiles growing 3D Tiles scenes in
an isolated GUI Blender. `benchmark_tiles_progressive.py` records per-tile phases,
UI/draw intervals and controlled diagnostic comparisons; `--prototype-check`
checks the restricted direct-transform experiment against the original renderer.
See [the profiling report](../docs/THREED_TILES_PROGRESSIVE_PROFILE.md) for
commands, measurements, restrictions and retained native failures. This slow
investigation is separate from normal release gates and changes no addon runtime.

The OSM catalog is `tests/fixtures/osm_catalog.toml`. Inputs are read directly from the
separate `osm_extracts` repository. `--extended` adds Bratislava and Moscow;
their companion files do not make the snapshots complete. The three cataloged
files currently untracked in that repository must also be provisioned on another
runner. Do not change input hashes merely to make a failed preflight pass.
`tests/fixtures/local_catalog.toml` pins the user's real GPX and Buildify binaries.
There is no synthetic GPX fixture.

`--helsinki` is optional and must name the directory directly containing the
three downloaded B3DM files. Supplying an empty directory fails. These inputs
are replayed locally; tests do not contact Helsinki or Google. The standalone,
concurrent and GUI suites require localhost socket access. New file-based suites
reject external network calls; imagery HTTP responses are controlled in process.
Live-provider scripts remain separate, explicitly invoked checks. The runner uses
Blender's `--online-mode` for the localhost concurrent/UI suites and the mocked
imagery download suite, because extension policy also governs these paths.
External imagery requests remain blocked by the suite's offline guard.

For a quick package and Base-contract check:

```powershell
& '../tools/python/python.exe' -B tests/run_release_tests.py `
  --blender '../tools/blender/blender.exe' --profile base --suite base `
  --candidate-zip '../artifacts/blosm_release_tests/candidate.zip' `
  --artifacts ../artifacts/blosm_release_tests/candidate_install
```

Use `--profile pro --suite pro` for the corresponding Pro checks. The legacy ZIP
layout with one outer addon directory is rejected. Headless imports set default
building levels explicitly because registration normally initializes them through
a UI timer. Some properties, including baking and export settings, remain in the
shared RNA definition in Base; tests check that Pro controls and implementations
are unavailable and that stale settings cannot activate Pro behavior.

Run `--suite ui` separately with desktop access, retaining `--profile` and
`--candidate-zip` when qualifying a build. It has a timeout and requires
`ui_checks.json` to say `passed`; closing Blender or exiting with zero is not
sufficient. The runner hides its dedicated window on Windows.
The HTTP fixture holds the second tile until the UI has processed five ticks
with a partial import, so incremental-import assertions do not race redraw speed.
The UI scenario yields to Blender's event loop after the blocking import and each
Undo/Redo action. This allows redraw and dependency-graph work to complete between
history operations; issuing them back-to-back inside a timer callback can crash
Blender 5.x. `UI_ACTION` console entries identify the last history operation.
The HTTP fixture retires its active request under the same lock used to send the
response. This prevents completed serial requests from inflating the concurrency
peak while retaining the assertions for genuinely overlapping requests.
The incomplete-transfer deadline check asserts the HTTP server's request count;
a retry may be counted internally immediately before its final deadline check
rejects it. A separate deterministic test verifies that retry backoff exhausts
one shared deadline without starting another request.

## Assertions and limits

Geometry comparisons use world-space coordinates with a 1 cm tolerance. They
preserve face multiplicity; 2D courtyard imports compare area and control vertices
because joined/unjoined imports can decompose a courtyard into different faces.
The Buildify check realizes instances only in an in-memory copy for measurement.
Ordinary 2D and GN courtyard decompositions are not assumed identical. Roof
controls use an explicit 12 m total height. No generated output has been approved
as a visual baseline.

Bakes use 128-pixel images and CPU Cycles, verify known linear surface colors after
decoding sRGB image pixels, and compare geometry and render settings. Export
texture tests use the real compositor/templates with CPU rendering at 25% size.
These are functional checks, not production-quality visual/performance approval.
The bake inputs include controlled surfaces and a red/green textured B3DM pair,
imported through the registered operator with joining both enabled and disabled.
GUI baking Undo/Redo remains an additional qualification case.

The larger release plan in `docs/RELEASE_TEST_PLAN.md` also calls for approved
visual baselines, performance/resource budgets, all supported platforms, more
terrain/georeferencing combinations, exhaustive asset/cache/error cases and
bounded live-provider checks. Those require further qualification beyond this
initial automated suite. A passing selected run is not a full release certificate.
Known product regressions remain failing assertions, not expected-failure markers.
