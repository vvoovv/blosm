# Progressive importing

OpenStreetMap now uses the same controller and scheduler. See
[OSM progressive import](../docs/OSM_PROGRESSIVE_IMPORT.md) for supported modes,
separate-object behavior, provider hooks, tests and the `buildings_gn` handoff.

**Progressive import** is enabled by default for 3D Tiles. Its checkbox appears
in the import box directly above the data-type selector. Disable it to use
blocking mode. Progressive mode keeps the viewport responsive between tile imports.
Both GLB and B3DM are supported. Root and nested JSON manifests and tile content
are downloaded into memory; no dataset caches or temporary tile files are used.
Ordinary static tiles use [direct mesh rotation](../threed_tiles/doc/DirectImport.md#applying-tile-rotations-without-operators)
to reduce repeated scene updates; complex inputs retain the operator path.
**Concurrent downloads** is independently enabled by default, with four requests.
It also serves blocking mode; see [shared downloads](../downloads/README.md).

Tiles appear in their final geographic orientation as each finishes. Selection,
active object, active collection and the 3D cursor are restored after each tile.
The status in the Blosm panel shows imported tiles, errors and download waiting.
**Cancel import** stops outstanding work and retains completed tiles. Esc remains
available for ordinary Blender interactions and does not cancel this timer job.

**Join 3D Tiles objects when finished** retains the existing join option. Tiles remain
separate while loading; joining happens once on successful completion. Cancelled
imports keep their separate objects. A large tile or final join can visibly pause
the viewport: individual Blender API calls are synchronous and cannot be preempted
by the scheduler's time budget. Disable joining to avoid that final operation.

Deleting or renaming imported objects does not stop loading. If the destination
collection is deleted or unlinked from the import scene, loading stops with one
error report. No replacement collection is created, including when deletion
happens just before the final join. Renaming
or moving the collection within the same scene preserves it as the destination;
an unrelated collection with the old name is never adopted. The reported tile
count records successful imports, including tiles subsequently deleted by the user.

Only one Blosm import may run at a time. Settings are captured when starting.
Changing to another scene/view layer or entering Edit Mode pauses scene changes
until the original scene/view layer and Object Mode are active again. Consecutive
tile imports and the final join share one undo step: one Undo removes the entire
import if no scene edits intervene. Scene edits during loading keep their own undo
steps and split the import into groups. Separate imports also remain separate undo
groups. Undo or Redo cancels the active job before restoring Blender data. Redo
restores the saved geometry without restarting downloads. File loading, disabling
the add-on and closing the owning window also stop the operation. Disabling Global
Undo in Blender disables these checkpoints as well.

Completion, failure and explicit cancellation display a Blosm import result
popup in the owning window and retain the normal operator report. Per-tile
failures remain console details. Silent cleanup for Undo, loading another file
or disabling the add-on does not open a popup.

The API entry point is still bpy.ops.blosm.import_data(). In progressive mode it
returns FINISHED after scheduling the job; that does not mean all tiles have
finished. Use blocking mode for scripts that need the final result immediately.
Command-line mode retains blocking behavior. Interactive progressive importing
is rejected in background Blender rather than silently leaving a timer unrun.

## Shared core and data-type adapters

The top-level progressive package is independent of 3D Tiles:

- core.Job drives a generator in bounded turns (8 ms / 32 checkpoints by default).
  A yielded None is a cooperative checkpoint. A URL/Request awaits a reply, sent
  back into the generator (or an error thrown into it). Download schedules a
  deferred main-thread consumer and allows discovery to continue. Drain waits
  for all deferred consumers before finalization. The return value is the job result.
  close/cancel invokes generator finally blocks and closes the transport.
  An optional validate callback runs before each step, allowing adapters to
  stop with an error if user edits invalidate the import destination.
- downloads.transport.DownloadTransport implements submit, poll and close using a separate
  ordinary Python helper and nonblocking loopback sockets. It discovers Blender's
  bundled Python at runtime. The helper window is hidden on Windows. URLs travel
  through IPC, not command-line arguments or files. The parent authenticates the
  connection with a per-process nonce before sending requests.
- downloads/worker.py runs a bounded thread pool outside Blender for HTTP and
  optional gzip decoding. Outstanding requests and unconsumed replies occupy
  the same bounded window, including while paused. Manifest discovery remains
  ordered; content downloads and import can overlap. Socket operations time out
  within 30 seconds and the helper enforces each resource's 90-second deadline
  across attempts. Completed bytes do not time out while Blender is busy importing.
  A separate supervisor stops the helper if a worker remains stuck five seconds
  past its deadline, including during DNS resolution; committed tiles are retained.
  Incomplete HTTP bodies retry within the original deadline and attempt budget.
  Resource errors remain per-tile; helper/protocol failures stop the job. Shared
  origin cooldowns, credential policies and diagnostic details are documented in
  [downloads/README.md](../downloads/README.md). The downloaded and decoded response
  size limit is 256 MiB each. Cancellation terminates the helper; nonblocking
  polling reaps it afterward. Parent disconnect also stops busy network threads.
- blender.Controller owns the application timer, redraws, cancellation UI,
  undo checkpoints and lifecycle handlers. Adapters supply job, scene, view_layer,
  revision, status, cancel and finish. No HTTP or provider logic lives here.

threed_tiles.manager.BaseManager.iterRender is the 3D Tiles producer. It yields
download URLs while retaining traversal, bounding-volume/LOD decisions, transforms,
session query parameters, and per-tile errors. The blocking runner drives the same
generator synchronously. Manifest JSON parsing and glTF/B3DM decoding currently
run on the main thread as part of bounded steps; large individual parse/import
calls can exceed the cooperative time budget.

threed_tiles.progressive.TilesImport connects that producer to Job and the Blender
driver. ProgressiveRenderer uses the existing renderer to finish one tile at a
time and moves its objects into the destination collection before yielding.
Coordinate conversion uses private import data without shared glTF patches.
Failed-tile objects, temporary collections and newly created
unused mesh/material/image data are removed. Failures remain console details;
completion emits one aggregate UI report, rather than an error popup per tile.

OSM and future bitmap imagery and Mapillary adapters can supply their own generators,
transports and scene-commit steps. They do not need tile URLs, bounding volumes or
glTF concepts in the shared scheduler. Large OSM parsing should use a dedicated
helper task protocol; it is not automatically made non-blocking by this scheduler.
The concurrent downloader is shared. OSM uses lazy GET/read-only POST transport
and incremental XML parsing; local imports with cached relation members do not
launch a network helper. Bitmap/Mapillary integration remains separate work.

`YieldToUI` explicitly ends a callback, unlike the soft `None` checkpoint.
`LazyTransport` avoids starting a downloader for CPU-only producers.
`progressive.ownership.Ownership` records owned IDs across turns. Its legacy
allocation `capture()` context must contain only synchronous work; never put a
generator yield inside that context. Pending ID tags prevent user undo/save
snapshots from restoring an incomplete publication after cancellation.

## Validation

All development outputs belong under C:/dev/workspace/artifacts/progressive_import.

Run tests/test_progressive_transport.py using ordinary Python (not Blender).
It covers helper reuse, byte payload framing, gzip, HTTP errors, truncated replies,
cancellation, helper termination and generator cleanup. Its HTTP server threads
run only in that standalone test process.

Run tests/check_progressive_import.py in factory-startup background Blender with
--python-exit-code 1 and --artifacts pointing to an output directory. It checks
incremental visibility, geometry parity with blocking mode, selection restoration,
joining, cancellation, failed-tile cleanup, and add-on/lifecycle shutdown.
Pass --helsinki with the directory of the three existing Helsinki B3DM fixtures
to compare real tiles, including a nested transform, in both joined and separate
object modes. Joining after rotation can differ from the blocking renderer's
join-before-rotation path at float32 precision; this comparison allows four
float32 units in the last place at the fixture coordinate magnitude.

Run tests/check_progressive_ui.py in a separate factory-startup GUI Blender process
with --artifacts pointing to an output directory. It starts a separate local HTTP
fixture process, imports delayed tiles using the real helper, changes the viewport
on timer heartbeats and exercises grouped Undo/Redo with and without joining,
consecutive imports, an intervening user edit, Cancel, Edit Mode pause/resume,
scene switch and add-on disable. It saves no preferences. Its report distinguishes pass/failure;
check that report rather than relying solely on the GUI process exit code.
