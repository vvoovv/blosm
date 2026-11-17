"""Profile growing 3D Tiles scenes through the actual GUI progressive driver.

No network: replay synthetic or approved Helsinki benchmark bytes in memory.
Diagnostic switches modify this isolated process only; they are not product fixes.
See run_tiles_profile.py for a factory-startup launcher with an isolated profile.
"""
from collections import defaultdict, deque
from contextlib import contextmanager
import argparse
import cProfile
import functools
import hashlib
import importlib
import json
from math import radians
from math import atan, sqrt, pi
from pathlib import Path
import statistics
import sys
import time
import traceback

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gltf_test_support import bpy, PACKAGE, REPO, fixture, make_glb, reset_scene
import addon_utils

parser = argparse.ArgumentParser()
parser.add_argument('--artifacts', type=Path, required=True)
parser.add_argument('--count', type=int, default=2000)
parser.add_argument('--sample', choices=('synthetic', 'helsinki'), default='synthetic')
parser.add_argument('--helsinki', type=Path, default=Path('C:/dev/workspace/artifacts/direct_gltf_import/helsinki/1'))
parser.add_argument('--seed-objects', type=int, default=0)
parser.add_argument('--seed-sample', action='store_true')
parser.add_argument('--no-undo', action='store_true')
parser.add_argument('--no-snapshots', action='store_true')
parser.add_argument('--checkpoint-every', type=int, default=1)
parser.add_argument('--profile-python', action='store_true')
parser.add_argument('--no-viewport', action='store_true')
parser.add_argument('--operators', action='store_true')
parser.add_argument('--seed-scene', type=Path)
parser.add_argument('--save-scene', action='store_true')
parser.add_argument('--direct-finalize', action='store_true')
parser.add_argument('--operator-finalize', action='store_true',
                    help='Force the retained operator path for production comparisons')
args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
assert args.count > 0 and args.checkpoint_every > 0
assert not (args.direct_finalize and args.operator_finalize)
args.artifacts.mkdir(parents=True, exist_ok=True)
if args.seed_scene:
    assert Path(bpy.data.filepath).resolve() == args.seed_scene.resolve(), 'Launcher must load the seed before the script'
addon_utils.enable(PACKAGE, default_set=False)
driver = importlib.import_module(PACKAGE + '.progressive.blender')
tiles = importlib.import_module(PACKAGE + '.threed_tiles.progressive')
render = importlib.import_module(PACKAGE + '.threed_tiles.blender')
gltf = importlib.import_module(PACKAGE + '.threed_tiles.gltf_import')
manager_module = importlib.import_module(PACKAGE + '.threed_tiles.manager')
if args.operator_finalize:
    render.BlenderRenderer._finalize_direct = lambda self, manager: False
driver.show_result = lambda *a: None
if not args.seed_scene:
    reset_scene()
scene = bpy.context.scene
initial_objects = len(bpy.data.objects)
bpy.context.preferences.filepaths.temporary_directory = str(args.artifacts)
bpy.context.preferences.filepaths.file_preview_type = 'NONE'
bpy.context.preferences.filepaths.use_auto_save_temporary_files = False
bpy.context.preferences.edit.use_global_undo = not args.no_undo

current = None
rows, ticks, hearts, draws, idle_hearts = [], [], [], [], []
metrics = defaultdict(list)
phase, started, completed_at = 'setup', None, None
operation = None
errors = []
profiler = cProfile.Profile() if args.profile_python else None
last_heartbeat = None
last_draw = None


def measured(label, function):
    @functools.wraps(function)
    def call(*a, **kw):
        start = time.perf_counter()
        try:
            return function(*a, **kw)
        finally:
            elapsed = time.perf_counter() - start
            metrics[label].append(elapsed)
            if current is not None:
                current[label] = current.get(label, 0.) + elapsed
    return call


def wrap(target, name, label):
    setattr(target, name, measured(label, getattr(target, name)))


wrap(gltf, '_import_settings', 'settings')
wrap(render, 'import_gltf', 'gltf_total')
for name in ('prepare', 'renderGlb', 'renderB3dm', 'finalize_glb_import', 'process_rtc'):
    wrap(render.BlenderRenderer, name, name)

# Measure the explicit update without changing Blender's shared RNA methods.
class LayerProxy:
    def __getattr__(self, key):
        value = getattr(bpy.context.view_layer, key)
        return measured('explicit_update', value) if key == 'update' else value


class ContextProxy:
    def __getattr__(self, key):
        return LayerProxy() if key == 'view_layer' else getattr(bpy.context, key)


class BpyProxy:
    def __getattr__(self, key):
        return ContextProxy() if key == 'context' else getattr(bpy, key)


gltf.bpy = BpyProxy()
from io_scene_gltf2.blender.imp.blender_gltf import BlenderGlTF
BlenderGlTF.create = staticmethod(measured('gltf_create', BlenderGlTF.create))
# Blender 5.2 operators are C callables; bpy.ops._op_call is no longer the
# dispatch point. For optional detailed probes, proxy only our module's calls.
class OperatorCall:
    def __init__(self, path, function):
        self.function = function
        self.call = measured('operator:' + path, function)
    def __call__(self, *a, **kw):
        return self.call(*a, **kw)
    def __getattr__(self, name):
        return getattr(self.function, name)


class OperatorModule:
    def __init__(self, name):
        self.name = name
    def __getattr__(self, key):
        return OperatorCall(self.name + '.' + key, getattr(getattr(bpy.ops,self.name),key))


class OperatorProxy:
    def __getattr__(self, key):
        return OperatorModule(key)


class RendererBpyProxy:
    def __getattr__(self, key):
        return OperatorProxy() if key == 'ops' else getattr(bpy,key)


if args.operators:
    render.bpy = RendererBpyProxy()
original_set = set


def snapshots(values):
    return original_set() if args.no_snapshots else original_set(values)


tiles.set = measured('rollback_snapshots', snapshots)
original_preserve = tiles.preserve_context


@contextmanager
def preserve():
    scope = original_preserve()
    measured('preserve_enter', scope.__enter__)()
    try:
        yield
    except BaseException:
        if not measured('preserve_exit', scope.__exit__)(*sys.exc_info()):
            raise
    else:
        measured('preserve_exit', scope.__exit__)(None, None, None)


tiles.preserve_context = preserve
original_finalize = render.BlenderRenderer.finalize
placement = 0


def direct_finalize(self, manager):
    """Diagnostic prototype for these unit-scale static mesh fixtures only.

    Rotate mesh data directly instead of selection/transform_apply operators.
    Not a production implementation for arbitrary glTF objects or shared meshes.
    """
    from mathutils import Matrix
    assert not self.join3dTilesObjects
    bpy.data.collections.remove(self.collection_import)
    self.collection_import = self.layer_collection_import = None
    if not self.collection.objects:
        bpy.data.collections.remove(self.collection)
        self.collection = None
        return
    center = self.centerCoords
    latitude = atan(center[2]/sqrt(center[0]*center[0]+center[1]*center[1]))
    rotation = Matrix.Rotation(latitude-pi/2,4,'X') @ Matrix.Rotation(radians(-90-manager.centerLon),4,'Z')
    objects = list(self.collection.objects)
    matrices = {obj:obj.matrix_world.copy() for obj in objects if obj.type=='MESH'}
    if self.calculateHeightOffset:
        self.heightOffset = min((rotation @ matrix.translation).z for matrix in matrices.values())
    for obj,matrix in matrices.items():
        assert obj.data.shape_keys is None and obj.data.users==1
        assert all(abs(axis.length-1)<1e-5 for axis in matrix.to_3x3().col)
        obj.parent = None
        obj.matrix_world = matrix
        # Preserve the original two rotations and their RNA decomposition
        # order. Fusing them introduced millimetre differences in real tiles.
        obj.data.transform(obj.matrix_basis.to_3x3().to_4x4())
        location = obj.location.copy()
        obj.matrix_basis = Matrix.Translation(location)
        location = rotation @ location
        obj.matrix_local = Matrix.Translation(location) @ rotation
        obj.data.transform(obj.matrix_basis.to_3x3().to_4x4())
        obj.matrix_basis = Matrix.Translation(location)
        obj.select_set(False)
    for obj in objects:
        assert obj.type in ('EMPTY','MESH')
        if obj.type=='EMPTY':
            bpy.data.objects.remove(obj,do_unlink=True)
    if hasattr(self, 'recordAttribution'):
        if self.recordAttribution and self.num_imported_tiles:
            self._attribution()
    else:
        bpy.context.scene.blosm.copyright = '; '.join(sorted(self.copyrightHolders))
    self.collection = None


def finalize(self, manager):
    global placement
    objects = list(self.collection.objects)
    measured('finalize', direct_finalize if args.direct_finalize else original_finalize)(self, manager)
    # Separate repeated samples spatially. Only this tile's objects are visited.
    for obj in objects:
        obj.location.x += (placement % 50) * 35
        obj.location.y += (placement // 50) * 35
    placement += 1


render.BlenderRenderer.finalize = finalize
original_tile = tiles.ProgressiveRenderer._tile


def tile(self, *a, **kw):
    global current
    start = time.perf_counter()
    current = {'tile': len(rows) + 1, 'objects_before': len(bpy.data.objects),
               'seconds': start - started}
    try:
        result = original_tile(self, *a, **kw)
        current['objects_after'] = len(bpy.data.objects)
        current['tile_seconds'] = time.perf_counter() - start
        rows.append(current)
        return result
    finally:
        current = None


tiles.ProgressiveRenderer._tile = tile
original_tick = driver.Controller.tick
original_checkpoint = driver.Controller.checkpoint


def checkpoint(self, initial=False):
    if not initial and self.adapter.renderer.revision % args.checkpoint_every:
        return
    return measured('checkpoint', original_checkpoint)(self, initial)


driver.Controller.checkpoint = checkpoint


def tick(self):
    before, start = len(rows), time.perf_counter()
    checkpoints = len(metrics['checkpoint'])
    if profiler:
        profiler.enable()
    try:
        return original_tick(self)
    finally:
        if profiler:
            profiler.disable()
        ticks.append({'tiles': len(rows), 'objects': len(bpy.data.objects),
                      'seconds': start - started, 'duration': time.perf_counter()-start,
                      'checkpoint_seconds': sum(metrics['checkpoint'][checkpoints:]),
                      'published': len(rows)-before})


driver.Controller.tick = tick


class ReplayTransport:
    max_requests = 8
    def __init__(self, root, samples):
        self.root, self.samples, self.queue = root, samples, deque()
        self.next_id, self.closed = 0, False

    def submit(self, request):
        identity = self.next_id
        self.next_id += 1
        content = self.root if request.url.endswith('.json') else self.samples[
            int(request.url.rsplit('/', 1)[-1].split('.')[0]) % len(self.samples)]
        self.queue.append((identity, content, None))
        return identity

    def poll(self):
        return self.queue.popleft() if self.queue else None

    def close(self):
        self.closed = True


center = manager_module.BaseManager.fromGeographic(60.17, 24.94, 0)
if args.sample == 'synthetic':
    doc, binary, _ = fixture()
    doc['nodes'][0]['translation'] = [center.x, center.z, -center.y]
    samples = [make_glb(doc, binary)]
    extension = 'glb'
    provenance = [{'synthetic': True, 'bytes': len(samples[0])}]
else:
    paths = sorted(args.helsinki.glob('mesh_*.b3dm'))
    assert len(paths) == 3, 'Expected the three approved Helsinki benchmark tiles'
    samples = [path.read_bytes() for path in paths]
    extension = 'b3dm'
    provenance = [dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
                  for path, data in zip(paths, samples)]
region = [radians(24.93), radians(60.16), radians(24.95), radians(60.18), -100, 100]
children = [dict(geometricError=0, boundingVolume={'region': region},
                 content={'uri': str(i) + '.' + extension}) for i in range(args.count)]
root = json.dumps({'root': dict(geometricError=0, boundingVolume={'region': region},
                               children=children)}).encode()

# Isolate existing object count. The optional sampled seed duplicates independent
# meshes/materials/images, so every diagnostic starts from the same datablock load.
template = None
if args.seed_sample:
    assert args.seed_objects > 0
    if extension == 'b3dm':
        from gltf_test_support import convert_b3dm
        sample, _ = convert_b3dm(samples[0])
    else:
        sample = samples[0]
    gltf.import_gltf('seed.glb', content=sample, coordinate_offset=center)
    template = next(obj for obj in scene.objects if obj.type=='MESH')
for index in range(args.seed_objects):
    if template:
        obj = template.copy()
        obj.data = template.data.copy()
        for slot in obj.material_slots:
            if slot.material:
                slot.material = slot.material.copy()
                for node in slot.material.node_tree.nodes:
                    if node.type=='TEX_IMAGE' and node.image:
                        node.image = node.image.copy()
    else:
        mesh = bpy.data.meshes.new('Preexisting mesh ' + str(index))
        mesh.from_pydata([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [], [(0, 1, 2)])
        obj = bpy.data.objects.new('Preexisting ' + str(index), mesh)
    scene.collection.objects.link(obj)
    obj.location = ((index % 50) * 35, (index // 50) * 35, -20)
if template:
    bpy.data.objects.remove(template, do_unlink=True)
bpy.context.view_layer.update()
for area in bpy.context.screen.areas:
    if area.type == 'VIEW_3D':
        if args.no_viewport:
            area.type = 'CONSOLE'
        else:
            area.spaces.active.shading.type = 'SOLID'
            area.spaces.active.region_3d.view_location = (700, 600, 0)
            area.spaces.active.region_3d.view_distance = 2500


def draw():
    global last_draw
    now = time.perf_counter()
    if phase == 'import' and last_draw is not None:
        draws.append((len(rows), now - last_draw))
    last_draw = now


handle = None if args.no_viewport else bpy.types.SpaceView3D.draw_handler_add(draw, (), 'WINDOW', 'POST_PIXEL')


def summary(values):
    return dict(count=len(values), total=sum(values), mean=statistics.mean(values) if values else 0.,
                median=statistics.median(values) if values else 0.,
                p95=sorted(values)[int(.95*(len(values)-1))] if values else 0.,
                maximum=max(values, default=0.))


def save_progress():
    report=dict(status='partial',rows=rows,ticks=ticks,heartbeat=hearts,draw_intervals=draws,
                args={key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()},
                metrics={name:summary(values) for name,values in metrics.items()})
    (args.artifacts/'progress.json').write_text(json.dumps(report),encoding='utf-8')


def finish(error=None):
    global phase
    phase = 'finished'
    if driver.active:
        driver.active.close(silent=True)
    if handle is not None:
        bpy.types.SpaceView3D.draw_handler_remove(handle, 'WINDOW')
    if profiler:
        profiler.dump_stats(str(args.artifacts/'python.pstats'))
    report = dict(status='failed' if error else 'passed', error=error,
                  blender=bpy.app.version_string, source=str(REPO), args=vars(args).copy(),
                  source_samples=provenance, seed_objects=args.seed_objects,
                  initial_objects=initial_objects,
                  result_tiles=operation.renderer.num_imported_tiles if operation else None,
                  errors=operation.manager.errors if operation else errors,
                  elapsed=completed_at-started if completed_at and started else None,
                  object_count=len(bpy.data.objects), data_counts={name:len(getattr(bpy.data,name))
                    for name in ('meshes','materials','images','collections')},
                  metrics={name:summary(values) for name,values in metrics.items()},
                  rows=rows, ticks=ticks, heartbeat=hearts, draw_intervals=draws,
                  idle_heartbeat=summary(idle_hearts))
    import gpu
    report['build_hash']=bpy.app.build_hash.decode()
    report['gpu']=dict(vendor=gpu.platform.vendor_get(),renderer=gpu.platform.renderer_get(),
                       backend=gpu.platform.backend_type_get())
    if not error:
        from gltf_test_support import fingerprint
        report['geometry_sha256'] = hashlib.sha256(json.dumps(fingerprint(),sort_keys=True).encode()).hexdigest()
        coordinates = sorted(sorted(tuple(round(value,3) for value in obj.matrix_world @ vertex.co)
                                    for vertex in obj.data.vertices)
                             for obj in bpy.context.scene.objects if obj.type=='MESH')
        report['world_geometry_mm_sha256'] = hashlib.sha256(json.dumps(coordinates).encode()).hexdigest()
    report['args'] = {key:str(value) if isinstance(value,Path) else value for key,value in report['args'].items()}
    (args.artifacts/'profile.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('TILES_PROFILE', json.dumps({k:report[k] for k in ('status','error','elapsed','object_count','result_tiles')}), flush=True)
    if args.save_scene and not error:
        bpy.ops.wm.save_as_mainfile(filepath=str(args.artifacts/'completed.blend'), copy=True, check_existing=False)
    bpy.ops.wm.quit_blender()


def heartbeat():
    global phase, started, completed_at, last_heartbeat
    now = time.perf_counter()
    try:
        if phase == 'setup':
            phase = 'import'
            renderer = tiles.ProgressiveRenderer('Profile tiles', False, '')
            manager = manager_module.BaseManager('https://replay.invalid/root.json', renderer)
            manager.centerLat, manager.centerLon = 60.17, 24.94
            manager.concurrentDownloads = True
            manager.setGeometricError(1)
            global operation
            operation = tiles.TilesImport(manager, (24.93,60.16,24.95,60.18), bpy.context, True,
                                          ReplayTransport(root, samples))
            started = time.perf_counter()
            driver.start(operation, bpy.context)
            last_heartbeat = time.perf_counter()
            return .01
        gap = now-last_heartbeat
        last_heartbeat = now
        if phase == 'import':
            hearts.append((len(rows), gap))
            if len(rows) % 100 == 0 and (not hearts or len(rows) != getattr(heartbeat,'last_count',-1)):
                print('TILES_PROGRESS', len(rows), 'objects',len(bpy.data.objects),
                      'seconds',round(now-started,3), flush=True)
                heartbeat.last_count=len(rows)
                save_progress()
            if driver.active is None:
                completed_at=now
                assert not operation.job.error, operation.job.error
                assert not operation.manager.errors, operation.manager.errors
                assert operation.renderer.num_imported_tiles == args.count
                assert len(rows) == args.count
                assert len(bpy.data.objects) > initial_objects + args.seed_objects
                phase='idle'
            elif now-started > 1500:
                raise TimeoutError('Profiling exceeded 25 minutes')
        elif phase == 'idle':
            idle_hearts.append(gap)
            if now-completed_at > 1.5:
                finish()
                return None
        # Keep the view changing and request a redraw on each heartbeat.
        for area in bpy.context.screen.areas:
            if area.type == 'VIEW_3D':
                area.spaces.active.region_3d.view_distance *= 1.000001
                area.tag_redraw()
        return .01
    except BaseException:
        finish(traceback.format_exc())
        return None


bpy.app.timers.register(heartbeat, first_interval=.5)
