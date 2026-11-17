"""Actual panel draw restrictions, undo/redo and progressive attribution commits."""
import argparse
import importlib
import json
from math import radians
from pathlib import Path
import sys
import time
import traceback
from urllib.parse import urlsplit

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from release_target import REPO as ROOT, PACKAGE, PROFILE, assert_target
if PROFILE == 'general':
    sys.path.insert(0, str(ROOT.parent))
import bpy
import addon_utils
from gltf_test_support import fixture, make_glb, reset_scene

parser = argparse.ArgumentParser()
parser.add_argument('--artifacts', type=Path, required=True)
args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
args.artifacts.mkdir(parents=True, exist_ok=True)
bpy.context.preferences.filepaths.temporary_directory = str(args.artifacts)
addon_utils.enable(PACKAGE, default_set=True)
assert_target(importlib.import_module(PACKAGE))
addon = importlib.import_module(PACKAGE)
attribution = importlib.import_module(PACKAGE + '.threed_tiles.attribution')
manager_module = importlib.import_module(PACKAGE + '.threed_tiles.manager')
progressive = importlib.import_module(PACKAGE + '.threed_tiles.progressive')
runtime = importlib.import_module(PACKAGE + '.progressive.blender')

region = {'region': [radians(24.93), radians(60.16), radians(24.95), radians(60.18), -100, 100]}
root = json.dumps({'root': {'geometricError': 0, 'boundingVolume': region,
    'children': [{'geometricError': 0, 'boundingVolume': region, 'content': {'uri': name}}
                 for name in ('first.glb', 'second.glb')]}}).encode()
payloads = {}
second_ready = True
draws = 0
draw_error = ''
checks = []
started = time.monotonic()
original_draw = addon.gui.BLOSM_PT_Copyright.draw


def state():
    settings = bpy.context.scene.blosm
    return (settings.get('copyright', ''), [
        (entry.as_pointer(), entry.collection.as_pointer() if entry.collection else None,
         list(entry.get('copyrights', ())), list(entry.get('counts', ())),
         list(entry.get('providerCredits', ()))) for entry in settings.importAttributions])


def draw(self, context):
    global draws, draw_error
    try:
        before = state()
        original_draw(self, context)
        assert state() == before, 'Panel draw modified persistent attribution data'
        draws += 1
    except Exception:
        draw_error = traceback.format_exc()
        raise


addon.gui.BLOSM_PT_Copyright.draw = draw


def open_panel():
    before = draws
    bpy.ops.wm.call_panel(name='BLOSM_PT_Copyright')
    assert not draw_error, draw_error
    assert draws > before, 'Copyright panel was not drawn'


class MemoryTransport:
    def __init__(self, max_requests=4):
        self.max_requests = max_requests
        self.pending, self.requests = {}, []

    def submit(self, request):
        identity = len(self.requests)
        self.requests.append(request.url)
        self.pending[identity] = urlsplit(request.url).path.rsplit('/', 1)[-1]
        return identity

    def poll(self):
        if self.pending:
            identity = next(iter(self.pending))
            name = self.pending[identity]
            if name == 'second.glb' and not second_ready:
                return None
            del self.pending[identity]
            return identity, payloads[name] if name.endswith('.glb') else root, None

    def close(self):
        self.pending.clear()


manager_module.DownloadTransport = progressive.DownloadTransport = MemoryTransport
runtime.show_result = lambda *_: None


class BLOSM_OT_AttributionDeleteTest(bpy.types.Operator):
    bl_idname = 'blosm.attribution_delete_test'
    bl_label = 'Delete imported collection for attribution test'
    bl_options = {'UNDO'}

    def execute(self, context):
        bpy.data.collections.remove(context.scene.blosm.importAttributions[0].collection)
        return {'FINISHED'}


bpy.utils.register_class(BLOSM_OT_AttributionDeleteTest)


def configure(first, second):
    settings = bpy.context.scene.blosm
    settings.dataType = '3d-tiles'
    settings.threedTilesSource = 'custom'
    settings.threedTilesUrl = 'https://fixture.invalid/root.json'
    settings.minLon, settings.maxLon = 24.93, 24.95
    settings.minLat, settings.maxLat = 60.16, 60.18
    settings.concurrentDownloads = True
    settings.join3dTilesObjects = False
    for name, text in zip(('first.glb', 'second.glb'), (first, second)):
        doc, binary, _ = fixture()
        center = manager_module.BaseManager.fromGeographic(60.17, 24.94, 0)
        doc['nodes'][0]['translation'] = [center.x, center.z, -center.y]
        doc['asset']['copyright'] = text
        payloads[name] = make_glb(doc, binary)


def steps():
    global second_ready
    reset_scene()
    bpy.context.scene.property_unset('blosm')
    configure('Blocking A; Shared', 'Blocking B; Shared')
    bpy.context.scene.blosm['copyright'] = 'Legacy data'
    bpy.context.scene.blosm.progressiveImport = False
    bpy.ops.ed.undo_push(message='Before attribution import')
    assert bpy.ops.blosm.import_data('EXEC_DEFAULT', True) == {'FINISHED'}
    yield
    assert len(bpy.context.scene.blosm.importAttributions) == 1
    assert 'Blocking A' in attribution.get_copyright(bpy.context.scene.blosm)
    open_panel()
    checks.append('Real read-only panel callback displays the registry without writes')
    bpy.ops.ed.undo()
    yield
    assert len(bpy.context.scene.blosm.importAttributions) == 0
    assert attribution.get_copyright(bpy.context.scene.blosm) == 'Legacy data'
    bpy.ops.ed.redo()
    yield
    assert len(bpy.context.scene.blosm.importAttributions) == 1
    assert 'Blocking A' in attribution.get_copyright(bpy.context.scene.blosm)
    checks.append('Blocking import registry and credits follow ordinary Undo/Redo')

    assert bpy.ops.blosm.attribution_delete_test('EXEC_DEFAULT', True) == {'FINISHED'}
    yield
    assert len(bpy.context.scene.blosm.importAttributions) == 1
    assert bpy.context.scene.blosm.importAttributions[0].collection is None
    assert attribution.get_copyright(bpy.context.scene.blosm) == 'Legacy data'
    open_panel()
    assert len(bpy.context.scene.blosm.importAttributions) == 1
    bpy.ops.ed.undo()
    yield
    assert bpy.context.scene.blosm.importAttributions[0].collection is not None
    assert 'Blocking A' in attribution.get_copyright(bpy.context.scene.blosm)
    checks.append('Deleting a collection filters credits during draw; Undo restores them')

    bpy.ops.ed.redo()
    yield
    configure('Replacement A', 'Replacement B')
    assert bpy.ops.blosm.import_data('EXEC_DEFAULT', True) == {'FINISHED'}
    yield
    assert len(bpy.context.scene.blosm.importAttributions) == 1
    assert 'Replacement A' in attribution.get_copyright(bpy.context.scene.blosm)
    assert 'Blocking A' not in attribution.get_copyright(bpy.context.scene.blosm)
    bpy.ops.ed.undo()
    yield
    assert len(bpy.context.scene.blosm.importAttributions) == 1
    assert bpy.context.scene.blosm.importAttributions[0].collection is None
    assert attribution.get_copyright(bpy.context.scene.blosm) == 'Legacy data'
    bpy.ops.ed.redo()
    yield
    assert bpy.context.scene.blosm.importAttributions[0].collection is not None
    checks.append('Stale-entry cleanup belongs to the next import undo step')

    reset_scene()
    bpy.context.scene.property_unset('blosm')
    configure('Progressive A; Shared', 'Progressive B; Shared')
    bpy.context.scene.blosm['copyright'] = 'Legacy data'
    bpy.context.scene.blosm.progressiveImport = True
    second_ready = False
    bpy.ops.ed.undo_push(message='Before progressive attribution import')
    assert bpy.ops.blosm.import_data('EXEC_DEFAULT', True) == {'FINISHED'}
    while runtime.active is not None and runtime.active.adapter.renderer.num_imported_tiles == 0:
        yield
    assert runtime.active is not None
    assert len(bpy.context.scene.blosm.importAttributions) == 1
    assert 'Progressive A' in attribution.get_copyright(bpy.context.scene.blosm)
    assert 'Progressive B' not in attribution.get_copyright(bpy.context.scene.blosm)
    open_panel()
    bpy.ops.blosm.cancel_progressive()
    yield
    assert runtime.active is None
    assert len(bpy.context.scene.blosm.importAttributions) == 1
    bpy.ops.ed.undo()
    yield
    assert len(bpy.context.scene.blosm.importAttributions) == 0
    assert attribution.get_copyright(bpy.context.scene.blosm) == 'Legacy data'
    bpy.ops.ed.redo()
    yield
    assert len(bpy.context.scene.blosm.importAttributions) == 1
    assert 'Progressive A' in attribution.get_copyright(bpy.context.scene.blosm)
    assert 'Progressive B' not in attribution.get_copyright(bpy.context.scene.blosm)
    checks.append('Partial progressive cancellation retains credits, and Undo/Redo follows geometry')

    second_ready = True
    assert bpy.ops.blosm.import_data('EXEC_DEFAULT', True) == {'FINISHED'}
    while runtime.active is not None:
        yield
    assert len(bpy.context.scene.blosm.importAttributions) == 2
    assert 'Progressive B' in attribution.get_copyright(bpy.context.scene.blosm)
    assert attribution.get_copyright(bpy.context.scene.blosm).count('Shared') == 1
    expected = attribution.get_copyright(bpy.context.scene.blosm)
    addon.gui.BLOSM_PT_Copyright.draw = original_draw
    addon_utils.disable(PACKAGE, default_set=True)
    assert not hasattr(bpy.types.Scene, 'blosm')
    addon_utils.enable(PACKAGE, default_set=True)
    assert attribution.get_copyright(bpy.context.scene.blosm) == expected
    checks.append('Completed imports aggregate and registered properties survive addon disable/enable')


operation = steps()


def finish(success, error=''):
    if runtime.active is not None:
        runtime.active.close(silent=True)
    (args.artifacts / 'results.json').write_text(json.dumps({
        'success': success, 'status': 'passed' if success else 'failed', 'blender': bpy.app.version_string, 'checks': checks,
        'draws': draws, 'error': error}, indent=2), encoding='utf-8')
    bpy.ops.wm.quit_blender()


def tick():
    try:
        assert time.monotonic() - started < 60, 'Attribution UI check timed out'
        assert not draw_error, draw_error
        next(operation)
    except StopIteration:
        finish(True)
        return None
    except Exception:
        error = traceback.format_exc()
        print(error)
        finish(False, error)
        return None
    return .05


bpy.app.timers.register(tick, first_interval=.5)
