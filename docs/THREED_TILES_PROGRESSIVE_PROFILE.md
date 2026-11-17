# Progressive 3D Tiles responsiveness profiling

Profiling task: 17 November 2026. The user observes noticeable viewport slowdown
at about 2,000 tiles. The investigation measures Blender work separately from
downloads and compares possible optimizations without changing addon runtime code.

## Scope and reproducibility

- Source baseline: local `release`, `e5fe4ac8b0e3a66e56ac9019dd90f99a8b434702`.
  Development branch: `profile_tiles_responsiveness` in this chat's checkout.
  The newer `cesium_ion_access` checkout has the same measured glTF/update,
  collection, transform-operator and checkpoint paths; its newer attribution
  registry is not included in this release-baseline measurement.
  The reference advanced during profiling to `371be39` (Blosm 2.8.1); unfinished
  profiling work was committed and that reference merged into this isolated
  development branch. The prototype recognizes both attribution APIs. Final
  parity and paired current-release checks are recorded separately below.
- Blender 5.2.0 LTS, build `fbe6228777e7`, GUI processes, solid shading, moving
  view and redraw requests. Each process starts with factory settings and all
  five user-resource directories inside this chat's isolated profile. The
  installed Blender, normal preferences and existing sessions are not modified.
- The actual `BaseManager`, `TilesImport`, `Job`, `ProgressiveRenderer` and
  shared `Controller` execute. Only transport is replaced with deterministic,
  immediately available in-memory replies. There is one committed tile per
  driver turn, and ordinary checkpoints remain enabled in the control.
- A textured three-vertex GLB isolates object/data-count overhead. Three actual
  Helsinki 2024 B3DM files from the approved benchmark inputs exercise Draco,
  textures, RTC and normal import processing. The Helsinki samples contain
  35,547, 7,083 and 5,310 imported vertices respectively; repeated samples are
  placed on a grid. They are replayed fixtures, not 2,000 distinct city tiles.
- Prepared scenes contain independent mesh/material/image datablocks. Paired
  large-scene experiments load the same saved 2,100-object scene and add the
  same samples. Setup, file loading, final geometry checks and scene saving are
  excluded from import elapsed time.
- A separate timer requests view changes and records scheduling gaps. A draw
  handler records viewport draw intervals. These observations show opportunities
  for interaction; they are not measurements of mouse-to-display latency.
- Measurements include low-overhead phase wrappers. Python profiling is enabled
  only in explicitly marked runs. Operator timings include Blender's implicit
  updates around calls, not just the operator's selection/transform algorithm.

Raw reports, Python profiles, console logs, launch arguments, source hashes and
process-memory samples belong under:

`C:/dev/workspace/artifacts/tiles_responsiveness/20261117/`

## Findings

The dominant growing costs are main-scene updates around glTF creation,
temporary collection changes, positioning operators and checkpoint operations.
Rollback datablock snapshots are measurable but small in this workload.

The checkpoint's wall time must not be interpreted as pure undo serialization.
When checkpoints are omitted, much of that time reappears between Python
callbacks as Blender updates the scene in the event loop. Compare UI timer gaps
and total import time, not only the duration of `Controller.tick()`.

The paired results and per-phase values are recorded in the artifact summary.

With the original code and the three-vertex textured tile:

| Existing objects | Additional tiles | Mean import-driver call | Mean UI timer gap |
| --- | ---: | ---: | ---: |
| 0, growing to 100 | 100 | 17 ms | 28 ms |
| 500, growing to 550 | 50 | 123 ms | 138 ms |
| 1,000, growing to 1,050 | 50 | 243 ms | 260 ms |
| 2,100, growing to 2,200 | 100 | 755 ms | 786 ms |

This demonstrates substantial scaling cost even when incoming geometry is tiny.
The scene used for the paired large-scene trials was prepared by duplicating
2,000 independent textured fixtures, then importing another 100 tiles normally
and saving the result. It was not a successful continuous 2,000-tile import.

The 2,100-object synthetic control's approximate mean phase times per tile were:

| Phase | Mean |
| --- | ---: |
| glTF explicit view-layer update | 191 ms |
| final positioning, including four operators | 328 ms |
| checkpoint | 208 ms |
| rollback snapshots | 1.2 ms |

These phases are disjoint. Other import/setup work accounts for the rest of the
755 ms driver call. Individual glTF construction and parsing measurements are
nested inside broader phases and must not be added to overlapping totals.

Controlled changes to that same scene, adding 100 synthetic tiles:

| Experiment | Total import time | Mean UI timer gap |
| --- | ---: | ---: |
| Original | 79.51 s | 786 ms |
| Global Undo disabled | 74.58 s | 737 ms |
| Rollback snapshots bypassed | 78.05 s | 771 ms |
| Checkpoint every eight tiles | 76.22 s | 753 ms |
| No 3D View editor | 74.02 s | 732 ms |

These are individual observations. Small percentage differences should not be
treated as established improvements. For example, disabling Undo reduced the
Python callback from 755 to 538 ms, while the observed UI gap stayed at 737 ms:
scene work moved outside that callback. Retaining Undo is the recommended choice.
Idle timer gaps after completion were much shorter, so this experiment primarily
shows stalls caused by ongoing import, rather than a similarly slow idle viewport.

The matched Helsinki comparison replays each of the three actual tiles ten times
into the same 2,100-object scene:

| Import path | Total for 30 tiles | Mean UI timer gap |
| --- | ---: | ---: |
| Original | 17.50 s | 560 ms |
| Direct-positioning prototype | 10.98 s | 350 ms |

The prototype reduced total time by about 37% and the mean scheduling gap by
about 38%. Complete final scene geometry/UV/color/material-image fingerprints
match exactly in these two runs. The direct prototype keeps the normal
checkpoint after every tile and retains the normal rollback snapshots.

Timing varied between earlier and later processes: an early fused-rotation
prototype took 46.70 s for 100 synthetic tiles, while the final two-rotation
prototype took 32.31 s, with the explicit-update/checkpoint times also changing
from about 190/206 ms to 131/137 ms. The isolated rotation change cannot explain
all of that difference. Compare nearby paired runs and phase times; do not claim
a 59% production speedup by dividing the early 79.51 s control by the later
32.31 s result. These tests do not control other workstation activity or clocks.

Nearby final synthetic repeats (30 added tiles, same saved 2,100-object scene)
gave 17.84 s for the original and 10.13 s for the final prototype, a 43% reduction.
Mean timer gaps were 571 and 322 ms; 95th-percentile gaps were 619 and 350 ms.
A second successful prototype repeat took 10.48 s. Complete scene fingerprints
match for these successful control/prototype runs. An earlier repeated control
exited natively and is excluded from timing comparisons.

Native `--debug-depsgraph-time` output supports the update diagnosis. Across the
corresponding five-tile processes, including setup and final checks, the original
logged 16 relation builds and 42 graph evaluations; the final prototype logged
12 builds and 12 evaluations. Typical original relation builds were around
90 ms, versus roughly 15–34 ms for many individual evaluation passes. The C
output is buffered, so these totals are not assigned to individual Python
markers or reported as exact per-tile build counts. Removing operators reduces
repeated evaluations as well as some relation rebuilds; it does not eliminate
the dependency graph or all of its scaling cost.

### Verification against release 2.8.1

After synchronizing the isolated branch with reference `371be39`, the paired
Helsinki test was repeated with the newer attribution code, again starting from
the same prepared scene and adding 30 tiles:

| Import path | Total | Mean UI timer gap | 95th-percentile gap |
| --- | ---: | ---: | ---: |
| Current release | 23.61 s | 757 ms | 898 ms |
| Current release plus benchmark prototype | 15.32 s | 490 ms | 569 ms |

This is about 35% less import time and a 37% shorter 95th-percentile timer gap.
The final scene fingerprints are identical. The four-fixture parity check also
passes against this reference, with zero world-coordinate differences. The
release runner's `progressive` and `attribution` suites pass in isolated Blender
5.2 processes. Absolute timings shifted across the investigation; the matched
comparisons consistently support reducing operator-induced scene work.

The prototype remains confined to the profiling harness. No runtime optimization
has been transferred to the main checkout or to `release`.

## Priorities

1. Replace the per-tile selection/rotation operators with equivalent data API
   operations. There are two `select_all` and two `transform_apply` calls in
   `BlenderRenderer.finalize()` per progressive tile. A restricted prototype
   demonstrates the benefit while keeping one checkpoint per tile. Production
   implementation must retain context restoration, hierarchy handling,
   shared-mesh behavior, normals, transforms and error cleanup for supported
   glTF inputs, with a safe fallback for cases the direct path cannot handle.
2. Reduce temporary collection creation, deletion and relinking. A progressive
   tile currently creates output/import collections, the glTF importer can create
   child collections, and objects are moved again into the accumulated output.
   Reusing staging or importing into the intended destination could reduce
   dependency-relation rebuilds. This needs design and failure/undo tests; it is
   a follow-up hypothesis, not an implemented or measured optimization here.
   A private staging scene is another candidate: construct and position a tile
   away from the active scene, then publish it once. The glTF importer's context,
   shared resources and final main-scene update must be validated before assuming
   this works or improves responsiveness.
3. For very large scenes, consider optional spatial chunks if separate tile
   objects are not required. Reducing object/dependency count can address the
   remaining scaling cost. Joining after the entire import does not improve
   responsiveness while that import is running. Chunking changes editing and
   joining behavior and needs bounded publication and join work.

Disabling Undo, omitting safety snapshots, lowering checkpoint frequency alone,
or hiding the viewport should not be the first production change on these
measurements. Concurrent downloading cannot shorten synchronous Blender work.
Large or heavily textured tiles can add further decode, geometry and GPU costs
which the small control intentionally isolates.

## Diagnostic prototype and qualification

`--direct-finalize` is benchmark-only. It supports static, unit-scale, unshared
mesh fixtures and checks those assumptions. It is not installed into Blosm.
The final prototype preserves the original two rotation/decomposition steps.
An earlier experiment fused them and differed by about 2 mm on one Helsinki
tile; that version was replaced before the final parity check.

`check_tiles_profile_prototype.py` compares the prototype against the unchanged
renderer for the synthetic GLB and all three Helsinki B3DM samples. Final world
vertex coordinate differences are zero in all four cases; topology, UV, color
and material/embedded-image fingerprints also match. This is qualification for
the benchmark fixtures, not coverage of every supported glTF feature/version.

The first continuous 2,000-tile control attempt exited natively with code
`0xC0000409` after 639 mesh-creation log entries, without a final timing report.
It is retained as a failed run (`synthetic_2000`), not used as a completed
2,000-tile measurement. Large-scene comparisons instead use prepared scenes
and successful subsequent imports. The cause of that native exit is unresolved.
An initial 500-object trial and one repeated 2,100-object control also exited
with the same native code. The 500-object trial's retry completed. Its last
recorded working set was about 0.78 GB; no out-of-memory cause is established.
Native stability requires separate investigation; the scene-update diagnosis
does not explain those exits. Failed runs are excluded from the timing tables.
The first seed-file loader also failed during harness setup because opening a
file inside `--python` invalidated the screen context; the launcher now loads
the file before executing the benchmark. All failures remain in the artifacts.

## Running the experiments

Use an isolated GUI process, not background mode:

```powershell
& 'C:/Users/vladi/AppData/Local/Programs/Python/Python313/python.exe' -B `
  tests/run_tiles_profile.py `
  --artifacts C:/dev/workspace/artifacts/tiles_responsiveness/NEW_RUN `
  --profile C:/dev/workspace/artifacts/blender_profiles/CHAT_ID/tp/NEW_RUN `
  --count 100 --seed-objects 2000 --seed-sample --operators
```

Run with an explicit working directory of this checkout. Replace the profile
chat ID and output/run names; never share another chat's profile. Use short
profile names to avoid Windows path-length problems. Do not run CPU-heavy
comparison processes concurrently.

`--sample helsinki` reads the three approved benchmark files under
`artifacts/direct_gltf_import/helsinki/1`; `--helsinki PATH` overrides that folder.
It does not download or cache new dataset files. `--save-scene` saves a generated
development scene under the run's artifact directory; `--seed-scene PATH` loads
it through the launcher before executing Python.

Diagnostic comparisons: `--direct-finalize`, `--no-undo`, `--no-snapshots`,
`--checkpoint-every 8`, `--no-viewport`. These flags affect only the isolated
process and are not production settings. In particular, omitting snapshots
disables normal failure protection and is only suitable for controlled fixtures.

`--profile-python` writes `python.pstats`; `--debug-graph` enables native graph
timing output. `--prototype-check` selects the short headless parity check
instead of the GUI benchmark. Summarize successful runs with:

```powershell
& 'C:/Users/vladi/AppData/Local/Programs/Python/Python313/python.exe' -B `
  tests/summarize_tiles_profile.py `
  C:/dev/workspace/artifacts/tiles_responsiveness/20261117
```

Generated scenes, JSON, logs and profiles are not committed. The benchmark,
launcher, parity check and this report are source/documentation in the repository.

## Production implementation qualification

The subsequent `optimize_tile_transforms` development branch implements the
data API path in the renderer, with preflight checks and the original operator
fallback. See [DirectImport.md](../threed_tiles/doc/DirectImport.md#applying-tile-rotations-without-operators)
for eligibility, applied-rotation behavior and failure handling. Joining in the
blocking renderer retains its operator implementation; progressive tiles are
positioned through the fast path before the existing optional final join.

The same 2,100-object prepared scene and 30 replayed Helsinki tiles measured:

| Path | Import time | Mean UI timer gap | 95th-percentile gap |
| --- | ---: | ---: | ---: |
| Retained operator path | 22.50 s | 721 ms | 852 ms |
| Production data API path | 15.69 s | 502 ms | 629 ms |
| Production path repeat | 16.58 s | 530 ms | 634 ms |

That is 26–30% less total time in these observations. The normal path performed
60 selection and 60 transform-application calls; the optimized path performed
none of those calls. The positioning phase fell from about 290 to 11 ms per
tile. Main scene updates and checkpoints still dominate the remaining cost.
All three final scene fingerprints are identical. This is a controlled replay,
not a full new city download, and timings are not universal guarantees.

The new `tile_transforms` suite is part of the release core catalog and passes
on Blender 4.5, 5.0, 5.1 and 5.2, including the optional Helsinki fixtures.
Existing direct-import, operator, progressive, concurrent-download and
attribution checks pass on 5.2, as do the GUI history/cancellation/joining and
attribution suites. The concurrent/UI tests needed localhost permission outside
the default sandbox; their first socket-restricted failure was rerun successfully.

Blender 3.6 could not load the current reference addon's `__init__.py` because
its Python lacks `tomllib`. This happens before the transform code executes and
is not introduced by this change; the release's declared minimum is 4.5. No
bundled Python or normal Blender profile was modified to bypass that limitation.

Production qualification reports and logs are under
`C:/dev/workspace/artifacts/tiles_direct_transform/20261117/`.
Use the benchmark's `--operator-finalize` to force the retained path for paired
comparisons; the default now measures the production path. `--direct-finalize`
remains the historical diagnostic prototype. No runtime change is transferred
to the main checkout or reference branch until requested.
