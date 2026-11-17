"""Isolated Blender GUI: actual ion import HTTP, responsive timers and cancellation."""
import argparse
import base64
from dataclasses import replace
import importlib
import json
from math import radians
from pathlib import Path
import subprocess
import sys
import time
import traceback
from urllib.parse import parse_qs, urlsplit
from urllib.request import urlopen

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from release_target import REPO as ROOT, PACKAGE, PROFILE, assert_target
if PROFILE == 'general':
    sys.path.insert(0, str(ROOT.parent))
import bpy
import addon_utils
from gltf_test_support import fixture, make_glb

parser = argparse.ArgumentParser()
parser.add_argument('--artifacts', type=Path, required=True)
args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
args.artifacts.mkdir(parents=True, exist_ok=True)
bpy.context.preferences.filepaths.temporary_directory = str(args.artifacts)
addon_utils.enable(PACKAGE, default_set=True)
assert_target(importlib.import_module(PACKAGE))
assert PACKAGE in bpy.context.preferences.addons
bpy.context.preferences.system.use_online_access = True  # Isolated process only; never saved.
ion = importlib.import_module(PACKAGE + '.threed_tiles.cesium_ion')
attribution = importlib.import_module(PACKAGE + '.threed_tiles.attribution')
manager_module = importlib.import_module(PACKAGE + '.threed_tiles.manager')
progressive = importlib.import_module(PACKAGE + '.threed_tiles.progressive')
runtime = importlib.import_module(PACKAGE + '.progressive.blender')
transport_module = importlib.import_module(PACKAGE + '.downloads.transport')

doc, binary, _ = fixture()
center = manager_module.BaseManager.fromGeographic(60.17, 24.94, 0)
doc['nodes'][0]['translation'] = [center.x, center.z, -center.y]
doc['asset']['copyright'] = 'UI fixture data'
glb = make_glb(doc, binary)
bounds = {'region': [radians(24.93), radians(60.16), radians(24.95), radians(60.18), -100, 100]}
root = json.dumps({'root': {'geometricError': 0, 'boundingVolume': bounds,
    'content': {'uri': 'nested/tileset.json?session=ui-session'}}}).encode()
nested = json.dumps({'root': {'geometricError': 0, 'boundingVolume': bounds,
    'children': [{'geometricError': 0, 'boundingVolume': bounds, 'content': {'uri': name}}
                 for name in ('first.glb', '../second.glb')]}}).encode()
endpoint = json.dumps({'type': '3DTILES', 'externalType': '3DTILES',
    'options': {'url': 'https://fixture.invalid/google/root.json?key=provider-secret'},
    'attributions': [{'html': '<img alt="Google Maps"> Cesium ion'}]}).encode()
payloads = {'success/2275207': endpoint, 'cancel/2275207': endpoint,
            'invalid/2275207': b'not JSON', 'google/root.json': root,
            'google/nested/tileset.json': nested, 'google/nested/first.glb': glb,
            'google/second.glb': glb}
gate = args.artifacts / 'release_second_tile'
if gate.exists():
    gate.unlink()
server = subprocess.Popen([transport_module.python_executable(), '-I', '-B',
    str(ROOT / 'tests/progressive_http_fixture.py')], stdin=subprocess.PIPE,
    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
server.stdin.write((json.dumps({'payloads': {k: base64.b64encode(v).decode() for k, v in payloads.items()},
                               'delays': {'success/2275207': 1.0, 'cancel/2275207': 2.0,
                                          'invalid/2275207': 0.5},
                               'gates': {'google/second.glb': str(gate)},
                               'record_headers': ['Authorization']}) + '\n').encode())
server.stdin.flush()
port = int(server.stdout.readline())
server.stdin.close()
server.stdout.close()


class FixtureTransport(transport_module.DownloadTransport):
    def submit(self, request):
        # Only HTTP routing is replaced; resolution, query/header construction,
        # IPC helper, traversal, native glTF importing and UI controller are real.
        if request.url.startswith('https://fixture.invalid/'):
            request = replace(request, url=request.url.replace('https://fixture.invalid',
                                                                'http://127.0.0.1:%s' % port, 1))
        return super().submit(request)


progressive.DownloadTransport = FixtureTransport
manager_module.DownloadTransport = FixtureTransport
settings = bpy.context.scene.blosm
settings.dataType = '3d-tiles'
settings.threedTilesSource = 'google'
settings.google3dTilesAccessVia = 'cesium_ion'
settings.progressiveImport = True
settings.concurrentDownloads = True
settings.minLon, settings.maxLon = 24.93, 24.95
settings.minLat, settings.maxLat = 60.16, 60.18
settings.join3dTilesObjects = False
prefs = bpy.context.preferences.addons[PACKAGE].preferences
prefs.googleMapsApiKey = ''
prefs.cesiumIonAccessToken = 'ui-test-token'
messages, checks, processes, waits = [], [], [], []
progressive.notify = lambda level, message: messages.append((level, message))
phase = 'start'
started = time.monotonic()
ticks_waiting = 0
snapshot_before = None
saved_mesh = None


def snapshot():
    return ({name: tuple(item.as_pointer() for item in getattr(bpy.data, name))
             for name in ('objects', 'collections', 'meshes', 'materials', 'images')},
            {key: bpy.context.scene.get(key) for key in ('lat', 'lon', 'height_offset')})


def start(case):
    global ticks_waiting, snapshot_before
    snapshot_before = snapshot()
    ion.API_URL = 'http://127.0.0.1:%s/%s/{asset_id}' % (port, case)
    before = time.monotonic()
    assert bpy.ops.blosm.import_data('EXEC_DEFAULT', True) == {'FINISHED'}
    assert time.monotonic() - before < 1., 'Progressive import blocked on endpoint HTTP'
    assert runtime.active is not None
    processes.append(runtime.active.adapter.job.transport.process)
    ticks_waiting = 0


def finish(success, error=''):
    if runtime.active is not None:
        runtime.active.close(silent=True)
    gate.touch()
    server.terminate()
    server.wait(timeout=5)
    for process in processes:
        process.wait(timeout=5)
    (args.artifacts / 'results.json').write_text(json.dumps({
        'success': success, 'status': 'passed' if success else 'failed', 'checks': checks, 'waiting_timer_ticks': waits,
        'blender': bpy.app.version_string, 'error': error}, indent=2), encoding='utf-8')
    bpy.ops.wm.quit_blender()


def tick():
    global phase, ticks_waiting, saved_mesh
    try:
        assert time.monotonic() - started < 60, 'UI check timed out'
        active = runtime.active.adapter if runtime.active else None
        if active and active.job.waiting:
            ticks_waiting += 1
        if phase == 'start':
            start('success')
            phase = 'partial'
        elif phase == 'partial' and active and active.renderer.num_imported_tiles == 1:
            assert ticks_waiting >= 5, 'No responsive timer turns while awaiting HTTP'
            waits.append(ticks_waiting)
            saved_mesh = next(obj for obj in active.renderer.collection.objects if obj.type == 'MESH')
            assert 'lat' in bpy.context.scene
            bpy.ops.blosm.cancel_progressive()
            assert runtime.active is None
            assert saved_mesh.name in bpy.context.scene.objects
            assert messages[-1][1].startswith('Import cancelled. Kept 1 3D Tiles')
            checks.append('Partial import cancellation retains a visible mesh and scene origin')
            gate.touch()
            start('success')
            phase = 'success'
        elif phase == 'success' and runtime.active is None:
            assert messages[-1] == ('INFO', 'Successfully imported 2 3D Tiles.'), messages
            assert 'Google Maps' in attribution.get_copyright(settings) and 'UI fixture data' in attribution.get_copyright(settings)
            checks.append('Concurrent progressive import resolves ion and imports nested GLBs with attribution')
            settings.concurrentDownloads = False
            settings.join3dTilesObjects = True
            start('success')
            phase = 'sequential'
        elif phase == 'sequential' and runtime.active is None:
            assert messages[-1] == ('INFO', 'Successfully imported 2 3D Tiles.'), messages
            checks.append('Single-download progressive import and final joining succeed')
            settings.progressiveImport = False
            for concurrent in (False, True):
                settings.concurrentDownloads = concurrent
                assert bpy.ops.blosm.import_data() == {'FINISHED'}
            checks.append('Blocking operator imports through real HTTP with single and concurrent downloads')
            settings.progressiveImport = True
            start('cancel')
            phase = 'cancel'
        elif phase == 'cancel' and active and active.job.waiting and ticks_waiting >= 3:
            bpy.ops.blosm.cancel_progressive()
            assert runtime.active is None
            assert messages[-1][1].startswith('Import cancelled. Kept 0 3D Tiles')
            assert snapshot() == snapshot_before
            checks.append('Cancellation during resolution leaves the scene unchanged')
            start('invalid')
            phase = 'invalid'
        elif phase == 'invalid' and runtime.active is None:
            assert messages[-1] == ('ERROR', 'Cesium ion returned invalid JSON'), messages
            assert snapshot() == snapshot_before
            assert prefs.googleMapsApiKey == '' and prefs.cesiumIonAccessToken == 'ui-test-token'
            checks.append('Endpoint failure reports an error without scene changes')
            with urlopen('http://127.0.0.1:%s/stats' % port, timeout=3) as response:
                stats = json.load(response)
            for path, headers in zip(stats['requests'], stats['request_headers']):
                if path.startswith('/google/'):
                    query = parse_qs(urlsplit(path).query)
                    assert query['key'] == ['provider-secret']
                    if '/root.json' not in path:
                        assert query['session'] == ['ui-session']
                    assert headers['Authorization'] is None, 'Ion token forwarded to Google'
                else:
                    assert headers['Authorization'] == 'Bearer ui-test-token'
            checks.append('HTTP requests carry the provider key/session and keep the ion token on endpoint requests')
            finish(True)
            return None
    except Exception:
        error = traceback.format_exc()
        print(error)
        finish(False, error)
        return None
    return .05


bpy.app.timers.register(tick, first_interval=.5)
