"""Run in an isolated Blender GUI: real timers, HTTP helper and cancellation."""
import argparse
import base64
import importlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
from urllib.request import urlopen

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
import bpy
import addon_utils

parser = argparse.ArgumentParser()
parser.add_argument('--artifacts', type=Path, required=True)
args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
args.artifacts.mkdir(parents=True, exist_ok=True)
bpy.context.preferences.filepaths.temporary_directory = str(args.artifacts)
addon_utils.enable(ROOT.name, default_set=True)
assert ROOT.name in bpy.context.preferences.addons
bpy.context.preferences.system.use_online_access = True  # Isolated process only; never saved.
addon = importlib.import_module(ROOT.name)
ion = importlib.import_module(ROOT.name + '.threed_tiles.cesium_ion')
access = importlib.import_module(ROOT.name + '.threed_tiles.ion_access')
runtime = importlib.import_module(ROOT.name + '.progressive.blender')
transport_module = importlib.import_module(ROOT.name + '.downloads.transport')

endpoint = json.dumps({'type': '3DTILES', 'externalType': '3DTILES',
    'options': {'url': 'https://unused.invalid/root.json?key=provider-secret'}}).encode()
payloads = {'success/2275207': endpoint, 'cancel/2275207': endpoint, 'invalid/2275207': b'not JSON'}
server = subprocess.Popen([transport_module.python_executable(), '-I', '-B',
    str(ROOT / 'tests/progressive_http_fixture.py')], stdin=subprocess.PIPE,
    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
server.stdin.write((json.dumps({'payloads': {k: base64.b64encode(v).decode() for k, v in payloads.items()},
                               'delays': {k: 1.5 for k in payloads}}) + '\n').encode())
server.stdin.flush()
port = int(server.stdout.readline())
server.stdin.close()
server.stdout.close()

settings = bpy.context.scene.blosm
settings.dataType = '3d-tiles'
settings.threedTilesSource = 'google'
settings.google3dTilesAccessVia = 'cesium_ion'
settings.progressiveImport = False  # Access resolution must still be responsive.
prefs = bpy.context.preferences.addons[ROOT.name].preferences
prefs.googleMapsApiKey = ''
prefs.cesiumIonAccessToken = 'ui-test-token'
messages, checks, processes, waits = [], [], [], []
access.notify_result = lambda level, message: messages.append((level, message))
initial = {name: tuple(item.as_pointer() for item in getattr(bpy.data, name))
           for name in ('objects', 'collections', 'meshes', 'materials', 'images')}
origin = {key: bpy.context.scene.get(key) for key in ('lat', 'lon', 'height_offset')}
phase = 'start'
started = time.monotonic()
phase_started = started
ticks_waiting = 0


def start(case):
    global phase_started, ticks_waiting
    ion.API_URL = 'http://127.0.0.1:%s/%s/{asset_id}' % (port, case)
    before = time.monotonic()
    assert bpy.ops.blosm.import_data('EXEC_DEFAULT', True) == {'FINISHED'}
    assert time.monotonic() - before < 1., 'Import blocked on endpoint HTTP'
    assert runtime.active is not None
    processes.append(runtime.active.adapter.job.transport.process)
    phase_started, ticks_waiting = time.monotonic(), 0


def unchanged():
    assert initial == {name: tuple(item.as_pointer() for item in getattr(bpy.data, name)) for name in initial}
    assert origin == {key: bpy.context.scene.get(key) for key in origin}
    assert prefs.googleMapsApiKey == '' and prefs.cesiumIonAccessToken == 'ui-test-token'


def finish(success, error=''):
    if runtime.active is not None:
        runtime.active.close(silent=True)
    server.terminate()
    server.wait(timeout=5)
    for process in processes:
        process.wait(timeout=5)
    (args.artifacts / 'results.json').write_text(json.dumps({
        'success': success, 'checks': checks, 'waiting_timer_ticks': waits,
        'blender': bpy.app.version_string, 'error': error}, indent=2), encoding='utf-8')
    bpy.ops.wm.quit_blender()


def tick():
    global phase, ticks_waiting
    try:
        assert time.monotonic() - started < 60, 'UI check timed out'
        if runtime.active is not None and runtime.active.adapter.job.waiting:
            ticks_waiting += 1
        if phase == 'start':
            start('success')
            phase = 'success'
        elif phase == 'success' and runtime.active is None:
            assert messages == [('INFO', access.RESOLUTION_MESSAGE)], messages
            assert ticks_waiting >= 5, 'No responsive timer turns while awaiting HTTP'
            waits.append(ticks_waiting)
            unchanged()
            checks.append('Responsive successful resolution with progressive tile importing disabled')
            start('cancel')
            phase = 'cancel'
        elif phase == 'cancel' and runtime.active.adapter.job.waiting and ticks_waiting >= 3:
            bpy.ops.blosm.cancel_progressive()
            assert runtime.active is None
            assert messages[-1] == ('INFO', 'Cesium ion access resolution cancelled')
            unchanged()
            checks.append('Cancellation closes a pending access job without scene changes')
            settings.progressiveImport = True
            start('invalid')
            phase = 'invalid'
        elif phase == 'invalid' and runtime.active is None:
            assert messages[-1] == ('ERROR', 'Cesium ion returned invalid JSON')
            unchanged()
            checks.append('Asynchronous endpoint failure is reported and leaves the scene unchanged')
            with urlopen('http://127.0.0.1:%s/stats' % port, timeout=3) as response:
                requests = json.load(response)['requests']
            assert '/success/2275207' in requests and '/invalid/2275207' in requests
            assert all(path in ('/success/2275207', '/cancel/2275207', '/invalid/2275207') for path in requests), requests
            checks.append('Only asset endpoint requests, no root manifests or tiles')
            finish(True)
            return None
    except Exception:
        error = traceback.format_exc()
        print(error)
        finish(False, error)
        return None
    return .05


bpy.app.timers.register(tick, first_interval=.5)
