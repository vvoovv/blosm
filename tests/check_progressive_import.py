"""Blender-run progressive adapter/core checks; all outputs stay in artifacts."""
import argparse
import importlib
import io
import json
from math import radians, frexp, ldexp
from pathlib import Path
import sys
from unittest.mock import patch
from urllib.error import HTTPError

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gltf_test_support import bpy, REPO, PACKAGE, fixture, make_glb, make_b3dm, reset_scene, no_tile_writes_or_operator
import addon_utils
from io_scene_gltf2.io.imp.gltf2_io_gltf import glTFImporter

parser = argparse.ArgumentParser()
parser.add_argument('--artifacts', type=Path, required=True)
parser.add_argument('--helsinki', type=Path)
args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
args.artifacts.mkdir(parents=True, exist_ok=True)
package = PACKAGE
attribution = importlib.import_module(package + '.threed_tiles.attribution')
addon_utils.enable(package, default_set=True)
manager_module = importlib.import_module(package + '.threed_tiles.manager')
renderer_module = importlib.import_module(package + '.threed_tiles.blender')
adapter_module = importlib.import_module(package + '.threed_tiles.progressive')
runtime = importlib.import_module(package + '.progressive.blender')
BaseManager = manager_module.BaseManager


class MemoryTransport:
    max_requests = 4
    def __init__(self, payloads):
        self.payloads = payloads
        self.pending = {}
        self.requests = []
        self.closed = False
        self.polls = 0

    def submit(self, request):
        uri = request.url
        identity = len(self.requests)
        self.pending[identity] = uri
        self.requests.append(uri)
        self.polls = 0
        return identity

    def poll(self):
        self.polls += 1
        if self.polls < 3:
            return None
        if not self.pending:
            return None
        identity = next(iter(self.pending))
        uri = self.pending.pop(identity)
        result = self.payloads[uri]
        return (identity, None, result) if isinstance(result, Exception) else (identity, result, None)

    def close(self):
        self.closed = True


def geometry():
    bpy.context.view_layer.update()
    return sorted(tuple(round(v, 4) for v in obj.matrix_world @ vertex.co)
                  for obj in bpy.context.scene.objects if obj.type == 'MESH'
                  for vertex in obj.data.vertices)


def inputs():
    center = BaseManager.fromGeographic(60.17, 24.94, 0)
    doc, binary, glb = fixture()
    b3dm = make_b3dm(glb, list(center))
    doc['nodes'][0]['translation'] = [center.x, center.z, -center.y]
    glb = make_glb(doc, binary)
    region = [radians(24.93), radians(60.16), radians(24.95), radians(60.18), -100, 100]
    children = [{'geometricError': 0, 'boundingVolume': {'region': region}, 'content': {'uri': name}}
                for name in ('first.glb', 'second.b3dm')]
    children.append({'geometricError': 0, 'boundingVolume': {'region': [0, 0, .01, .01, 0, 10]},
                     'content': {'uri': 'excluded.glb'}})
    tileset = {'root': {'geometricError': 0, 'boundingVolume': {'region': region}, 'children': children}}
    root = {'root': {'geometricError': 0, 'boundingVolume': {'region': region},
                    'content': {'uri': 'nested/tileset.json'}}}
    return {'https://test.invalid/root.json': json.dumps(root).encode(),
            'https://test.invalid//nested/tileset.json': json.dumps(tileset).encode(),
            'https://test.invalid//nested/first.glb': glb,
            'https://test.invalid//nested/second.b3dm': b3dm}


def manager(renderer, payloads):
    result = BaseManager('https://test.invalid/root.json', renderer)
    result.centerLat, result.centerLon = 60.17, 24.94
    result.setGeometricError(1)
    result.download = payloads.__getitem__
    return result


def task(join=False, payloads=None):
    payloads = inputs() if payloads is None else payloads
    renderer = adapter_module.ProgressiveRenderer('Progressive test', join, '')
    renderer.calculateHeightOffset = True
    transport = MemoryTransport(payloads)
    tile_manager = manager(renderer, payloads)
    tile_manager.concurrentDownloads = True
    operation = adapter_module.TilesImport(tile_manager,
        (24.93, 60.16, 24.95, 60.18), bpy.context, True, transport)
    return operation, transport


def drive(operation, after_step=lambda: None):
    for _ in range(500):
        with no_tile_writes_or_operator(no_dataset_files=True):
            operation.job.step(max_steps=1)
        assert not hasattr(glTFImporter, '_patch_convert_functions')
        after_step()
        if operation.job.done:
            assert operation.job.error is None, operation.job.error
            return
    raise AssertionError('Progressive import did not complete')


checks = []
try:
    for join in (False, True):
        reset_scene()
        payloads = inputs()
        renderer = renderer_module.BlenderRenderer('Baseline', join, '')
        result = manager(renderer, payloads).render(24.93, 60.16, 24.95, 60.18)
        assert result == (2, [])
        expected = geometry()
        reset_scene()
        sentinel = bpy.data.objects.new('User selection', None)
        bpy.context.scene.collection.objects.link(sentinel)
        sentinel.select_set(True)
        bpy.context.view_layer.objects.active = sentinel
        layer_collection = bpy.context.view_layer.active_layer_collection
        operation, transport = task(join)
        seen = []
        first_geometry = []
        def observe():
            assert bpy.context.view_layer.objects.active == sentinel
            assert bpy.context.selected_objects == [sentinel]
            assert bpy.context.view_layer.active_layer_collection == layer_collection
            count = operation.renderer.num_imported_tiles
            if count == 1 and not first_geometry:
                first_geometry.extend(geometry())
            seen.append(count)
        drive(operation, observe)
        assert 0 in seen and 1 in seen and 2 in seen
        assert all(vertex in geometry() for vertex in first_geometry)
        assert geometry() == expected, (geometry(), expected)
        assert transport.closed and len(transport.requests) == 4
        assert operation.job.result == (2, [])
        assert 'City of Test' in attribution.get_copyright(bpy.context.scene.blosm)
        checks.append('incremental geometry/context/blocking parity, join=%s' % join)

    for join in (False, True):
        for imported_before_edit in (1, 2):
            for edit in ('delete_collection', 'delete_hierarchy', 'unlink', 'rename_and_move'):
                reset_scene()
                operation, transport = task(join)
                for _ in range(500):
                    operation.job.step(max_steps=1)
                    if operation.renderer.num_imported_tiles == imported_before_edit:
                        break
                assert operation.renderer.num_imported_tiles == imported_before_edit
                assert not operation.job.done  # Edit before finalization, even after the last tile.
                destination = operation.renderer.collection
                old_name = destination.name
                if edit == 'rename_and_move':
                    destination.name = 'Renamed by user'
                    parent = bpy.data.collections.new('User parent')
                    bpy.context.scene.collection.children.link(parent)
                    parent.children.link(destination)
                    bpy.context.scene.collection.children.unlink(destination)
                elif edit == 'unlink':
                    bpy.context.scene.collection.children.unlink(destination)
                    destination.name = 'Detached by user'
                else:
                    if edit == 'delete_hierarchy':
                        for obj in list(destination.objects):
                            bpy.data.objects.remove(obj, do_unlink=True)
                    bpy.data.collections.remove(destination)
                # A replacement with the same name must not be adopted or joined.
                unrelated = bpy.data.collections.new(old_name)
                bpy.context.scene.collection.children.link(unrelated)
                sentinel = bpy.data.objects.new('User mesh', bpy.data.meshes.new('User mesh'))
                unrelated.objects.link(sentinel)
                if edit == 'rename_and_move':
                    drive(operation)
                    assert operation.job.result == (2, [])
                else:
                    requests = list(transport.requests)
                    collections = set(bpy.data.collections)
                    operation.job.step()
                    assert operation.job.done
                    assert isinstance(operation.job.error, RuntimeError)
                    message = 'The destination collection was deleted or removed from the import scene'
                    assert str(operation.job.error) == message
                    assert operation.renderer.num_imported_tiles == imported_before_edit
                    assert not operation.manager.errors
                    assert transport.requests == requests
                    assert set(bpy.data.collections) == collections
                    reports = []
                    with patch.object(adapter_module, 'notify', side_effect=lambda *args: reports.append(args)):
                        operation.finish()
                    assert reports == [('ERROR', '3D Tiles import stopped: ' + message)]
                assert transport.closed
                assert list(unrelated.objects) == [sentinel] and sentinel.name == 'User mesh'
                if edit == 'rename_and_move':
                    assert operation.renderer.collection == destination
                    assert len(destination.objects) == (1 if join else 2)
                if edit == 'unlink':
                    assert len(destination.objects) == imported_before_edit
                    assert destination not in bpy.context.scene.collection.children_recursive
        checks.append('destination deletion/unlink/rename/move before next tile and final join, join=%s' % join)

    reset_scene()
    operation, transport = task()
    operation.manager.concurrentDownloads = False
    drive(operation)
    assert operation.job.result == (2, []) and len(geometry()) == 6
    checks.append('progressive import also works with concurrent downloads disabled')

    reset_scene()
    operation, transport = task(True)
    while operation.renderer.num_imported_tiles == 0:
        operation.job.step()
    retained = geometry()
    operation.cancel()
    assert operation.job.done and operation.job.cancelled and transport.closed
    assert geometry() == retained
    assert not hasattr(glTFImporter, '_patch_convert_functions')
    checks.append('cancel retains finished tiles and closes transport')

    for progressive in (False, True):
        for join in (False, True):
            reset_scene()
            payloads = inputs()
            renderer = (adapter_module.ProgressiveRenderer if progressive else renderer_module.BlenderRenderer)(
                'Worker failure', join, '')
            tile_manager = manager(renderer, payloads)
            tile_manager.concurrentDownloads = True
            transport = MemoryTransport(payloads)
            normal_poll = transport.poll
            def failed_worker():
                if renderer.num_imported_tiles:
                    raise TimeoutError('A download worker remained blocked past its deadline')
                if transport.pending and next(iter(transport.pending.values())).endswith('second.b3dm'):
                    return None  # Keep this resource pending until the first tile is committed.
                return normal_poll()
            transport.poll = failed_worker
            if progressive:
                operation = adapter_module.TilesImport(tile_manager,
                    (24.93, 60.16, 24.95, 60.18), bpy.context, False, transport)
                for _ in range(500):
                    operation.job.step(max_steps=1)
                    if operation.job.done:
                        break
                assert operation.job.done and isinstance(operation.job.error, TimeoutError)
                message = str(operation.job.error)
            else:
                with patch.object(manager_module, 'DownloadTransport', return_value=transport):
                    result = tile_manager.render(24.93, 60.16, 24.95, 60.18)
                assert len(result) == 1
                message = result[0]
            assert 'worker remained blocked' in message
            assert renderer.num_imported_tiles == 1 and len(geometry()) == 3
            assert transport.closed and not hasattr(glTFImporter, '_patch_convert_functions')
    checks.append('fatal helper error retains committed geometry and cleans up in blocking/progressive, joined/separate')

    reset_scene()
    payloads = inputs()
    payloads['https://test.invalid//nested/first.glb'] = b'invalid glTF'
    operation, transport = task(payloads=payloads)
    drive(operation)
    assert operation.renderer.num_imported_tiles == 1
    assert len(operation.manager.errors) == 1
    assert 'first.glb' in operation.manager.errors[0]
    checks.append('tile failure continues without individual UI errors')

    reset_scene()
    payloads['https://test.invalid//nested/second.b3dm'] = b'invalid B3DM'
    operation, transport = task(payloads=payloads)
    reports = []
    operation.renderer.report = lambda level, message: reports.append((level, message))
    drive(operation)
    assert reports == [] and len(operation.manager.errors) == 2
    with patch.object(adapter_module, 'notify', side_effect=lambda level, message: reports.append((level, message))):
        operation.finish()
    assert len(reports) == 1 and reports[0][0] == 'ERROR'
    assert 'No 3D tiles were imported. 2 errors' in reports[0][1]
    checks.append('all failed tiles produce one aggregate UI error')

    reset_scene()
    payloads = inputs()
    payloads['https://test.invalid/root.json'] = RuntimeError('Root unavailable')
    operation, transport = task(payloads=payloads)
    drive(operation)
    assert len(operation.job.result) == 1 and 'Root unavailable' in operation.job.result[0]
    assert transport.closed and not geometry()
    checks.append('root download failure closes the job without scene data')

    class BrokenErrorBody(io.BytesIO):
        def read(self, *args):
            raise OSError('incomplete error body')

    for progressive in (False, True):
        for broken_body in (False, True):
            reset_scene()
            uri = 'https://tile.googleapis.com/v1/3dtiles/root.json?key=private-key&session=private-session'
            body = (BrokenErrorBody() if broken_body else io.BytesIO(json.dumps({
                'error': {'message': 'Rejected private-key and private-session at ' + uri}
            }).encode()))
            failure = HTTPError(uri, 403, 'Permission denied', {}, body)
            operation, transport = task(payloads={uri: failure})
            tile_manager = operation.manager
            tile_manager.rootUri = uri
            tile_manager.uriServer = 'https://tile.googleapis.com'
            if progressive:
                drive(operation)
                result = operation.job.result
            else:
                operation.cancel()
                tile_manager.renderer = renderer_module.BlenderRenderer('Error test', False, '')
                tile_manager.concurrentDownloads = False
                def fail_download(uri):
                    raise failure
                tile_manager.download = fail_download
                result = tile_manager.render(24.93, 60.16, 24.95, 60.18)
            assert len(result) == 1 and '403' in result[0], result
            if not broken_body:
                assert 'Message from Google Cloud' in result[0]
            assert 'private-key' not in result[0] and 'private-session' not in result[0]
            tile_manager.errors = []
            tile_manager.processError(RuntimeError('Download failed at ' + uri), uri)
            assert 'root.json' in tile_manager.errors[0]
            assert 'private-key' not in tile_manager.errors[0] and 'private-session' not in tile_manager.errors[0]
    checks.append('blocking/progressive root and tile errors redact credentials; broken error body preserves HTTP status')

    reset_scene()
    tile_manager = manager(renderer_module.BlenderRenderer('Startup failure', False, ''), inputs())
    tile_manager.concurrentDownloads = True
    with patch.object(manager_module, 'DownloadTransport', side_effect=OSError('Permission denied')):
        result = tile_manager.render(24.93, 60.16, 24.95, 60.18)
    assert len(result) == 1 and 'Unable to start the download helper' in result[0]
    assert not geometry()
    checks.append('blocking helper startup failure uses the normal import error result')

    reset_scene()
    from io_scene_gltf2.blender.imp.blender_gltf import BlenderGlTF
    operation, transport = task()
    create = BlenderGlTF.create
    calls = []
    def fail_once(importer):
        create(importer)
        calls.append(True)
        if len(calls) == 1:
            raise RuntimeError('failure after creating scene data')
    with patch.object(BlenderGlTF, 'create', side_effect=fail_once):
        drive(operation)
    assert len([o for o in bpy.context.scene.objects if o.type == 'MESH']) == 1
    assert len(bpy.data.meshes) == 1
    assert not [c for c in bpy.data.collections if '_import' in c.name]
    checks.append('partial tile objects and temporary collections cleaned on failure')

    reset_scene()
    payloads = inputs()
    root = json.loads(payloads['https://test.invalid//nested/tileset.json'])
    root['root']['children'][0]['content']['uri'] = 'nested/first.glb'
    root['root']['children'] = [root['root']['children'][0], {'geometricError': 0}]
    payloads['https://test.invalid/root.json'] = json.dumps(root).encode()
    operation, transport = task(payloads=payloads)
    drive(operation)
    assert len(operation.job.result) == 1 and 'boundingVolume' in operation.job.result[0]
    assert operation.renderer.num_imported_tiles == 1 and geometry()
    assert transport.closed
    checks.append('traversal failure drains submitted content before renderer finalization')

    if args.helsinki:
        payloads = inputs()
        nested_uri = 'https://test.invalid//nested/tileset.json'
        nested = json.loads(payloads[nested_uri])
        bounds = nested['root']['boundingVolume']
        nested['root']['children'] = []
        for tile in sorted(args.helsinki.glob('*.b3dm')):
            nested['root']['children'].append({'geometricError': 0, 'boundingVolume': bounds,
                                               'content': {'uri': tile.name}})
            payloads['https://test.invalid//nested/' + tile.name] = tile.read_bytes()
        nested['root']['transform'] = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 5, -3, 2, 1]
        payloads[nested_uri] = json.dumps(nested).encode()
        for join in (False, True):
            reset_scene()
            renderer = renderer_module.BlenderRenderer('Helsinki', join, '')
            assert manager(renderer, payloads).render(24.93, 60.16, 24.95, 60.18)[0] == 3
            expected = geometry()
            reset_scene()
            operation, transport = task(join, payloads)
            drive(operation)
            assert operation.renderer.num_imported_tiles == 3
            actual = geometry()
            assert len(actual) == len(expected)
            from mathutils.kdtree import KDTree
            tree = KDTree(len(expected))
            for index, point in enumerate(expected):
                tree.insert(point, index)
            tree.balance()
            distance = max(tree.find(point)[2] for point in actual)
            # Joining after geographic rotation changes float32 arithmetic
            # order from the blocking renderer's join-before-rotation path.
            # Bound the difference to four float32 ULPs at this coordinate
            # magnitude (millimetres for these Helsinki fixtures).
            magnitude = max(abs(v) for point in expected for v in point)
            tolerance = max(.001, ldexp(4., frexp(magnitude)[1] - 24)) if join else .001
            print('HELSINKI_DISTANCE', join, distance, 'tolerance', tolerance)
            assert distance <= tolerance, 'Helsinki world-space geometry changed: %s' % distance
        checks.append('three Helsinki tiles with nested transform, joined and separate, match blocking')

    reset_scene()
    operation, transport = task()
    controller = runtime.start(operation, bpy.context)
    controller.tick()
    assert not bpy.ops.blosm.import_data.poll()
    runtime.stop_for_scene_change()
    assert runtime.active is None and transport.closed
    assert not bpy.app.timers.is_registered(runtime._tick)
    checks.append('single job guard and scene/undo lifecycle cancellation')

    reset_scene()
    operation, transport = task()
    controller = runtime.start(operation, bpy.context)
    controller.tick()
    addon_utils.disable(package, default_set=True)
    assert runtime.active is None and transport.closed
    checks.append('addon disable closes active import')
    report = {'blender': bpy.app.version_string, 'status': 'passed', 'checks': checks}
    (args.artifacts / 'checks.json').write_text(json.dumps(report, indent=2))
    print('PROGRESSIVE_CHECKS', json.dumps(report))
finally:
    if addon_utils.check(package)[1]:
        addon_utils.disable(package, default_set=True)
