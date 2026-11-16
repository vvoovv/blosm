"""Blender registration, UI, operator routing and resolution-only lifecycle."""
import argparse
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

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
assert ROOT.name in bpy.context.preferences.addons, 'Addon registration failed'
addon = importlib.import_module(ROOT.name)
ion = importlib.import_module(ROOT.name + '.threed_tiles.cesium_ion')
access = importlib.import_module(ROOT.name + '.threed_tiles.ion_access')
manager = importlib.import_module(ROOT.name + '.threed_tiles.manager')
renderer = importlib.import_module(ROOT.name + '.threed_tiles.blender')
prefs = bpy.context.preferences.addons[ROOT.name].preferences


class MemoryTransport:
    max_requests = 1

    def __init__(self, payload=None, error=None, wait=False):
        self.payload = payload if payload is not None else json.dumps({
            'type': '3DTILES', 'externalType': '3DTILES',
            'options': {'url': 'https://tile.googleapis.com/v1/3dtiles/root.json?key=provider-secret'},
            'attributions': [{'html': 'Cesium ion / Google Maps', 'collapsible': False}]}).encode()
        self.error, self.wait = error, wait
        self.requests = []
        self.pending = False
        self.closed = False

    def submit(self, request):
        self.requests.append(request)
        self.pending = True
        return 1

    def poll(self):
        if self.pending and not self.wait:
            self.pending = False
            return 1, self.payload if self.error is None else None, self.error

    def close(self):
        self.closed = True


class Layout:
    def __init__(self):
        self.events = []

    def row(self, **_):
        return self

    box = column = split = row

    def label(self, text='', **_):
        self.events.append(('label', text))

    def prop(self, owner, name, **kwargs):
        self.events.append(('prop', name, kwargs.get('text')))

    def operator(self, name, **_):
        self.events.append(('operator', name))
        return SimpleNamespace()

    def separator(self, **_):
        pass


def scene_state():
    return {name: tuple(item.as_pointer() for item in getattr(bpy.data, name))
            for name in ('objects', 'collections', 'meshes', 'materials', 'images')}, {
            key: bpy.context.scene.get(key) for key in ('lat', 'lon', 'height_offset')}


class BlenderChecks(unittest.TestCase):
    def setUp(self):
        bpy.context.scene.property_unset('blosm')
        self.settings = bpy.context.scene.blosm
        self.settings.dataType = '3d-tiles'
        self.settings.threedTilesSource = 'google'
        self.settings.google3dTilesAccessVia = 'cesium_ion'
        prefs.googleMapsApiKey = ''
        prefs.cesiumIonAccessToken = ''

    def rejected(self, expected):
        before = scene_state()
        with self.assertRaisesRegex(RuntimeError, expected):
            bpy.ops.blosm.import_data()
        self.assertEqual(scene_state(), before)

    def test_defaults_ui_visibility_and_preferences_order(self):
        self.settings.property_unset('google3dTilesAccessVia')
        self.assertEqual(self.settings.google3dTilesAccessVia, 'google')
        layout = Layout()
        addon.gui.BLOSM_PT_Settings.draw3dTiles(SimpleNamespace(layout=layout), bpy.context)
        self.assertIn(('prop', 'google3dTilesAccessVia', 'Access via'), layout.events)
        self.settings.threedTilesSource = 'custom'
        layout.events.clear()
        addon.gui.BLOSM_PT_Settings.draw3dTiles(SimpleNamespace(layout=layout), bpy.context)
        self.assertFalse(any(e[:2] == ('prop', 'google3dTilesAccessVia') for e in layout.events))
        self.assertIn(('prop', 'threedTilesUrl', None), layout.events)
        layout.events.clear()
        addon.BlosmPreferences.draw(SimpleNamespace(layout=layout, screenType='preferences',
                                                   enableExperimentalFeatures=False), bpy.context)
        ordered = [('label', '3D Tiles:'), ('prop', 'googleMapsApiKey', None),
                   ('prop', 'cesiumIonAccessToken', None), ('label', 'Overlays:'),
                   ('prop', 'arcgisAccessToken', None), ('prop', 'mapboxAccessToken', None)]
        indices = [layout.events.index(event) for event in ordered]
        self.assertEqual(indices, sorted(indices))
        self.assertIn(('operator', 'blosm.get_cesium_ion_token'), layout.events)
        prop = prefs.bl_rna.properties['cesiumIonAccessToken']
        self.assertEqual(prop.name, 'Cesium ion access token')
        self.assertEqual(prop.subtype, 'PASSWORD')

    def test_missing_ion_token_never_starts_network_or_renderer(self):
        for token in ('', ' \t\n '):
            prefs.cesiumIonAccessToken = token
            prefs.googleMapsApiKey = 'direct-google-key'
            with patch.object(access, 'DownloadTransport') as transport:
                self.rejected('Cesium ion access token is not set')
                transport.assert_not_called()

    def test_online_access_preference_is_respected(self):
        prefs.cesiumIonAccessToken = 'ion-token'
        original = bpy.context.preferences.system.use_online_access
        try:
            bpy.context.preferences.system.use_online_access = False
            self.rejected('Online access is disabled')
        finally:
            bpy.context.preferences.system.use_online_access = original

    def test_direct_google_still_requires_its_own_key(self):
        self.settings.google3dTilesAccessVia = 'google'
        self.settings.progressiveImport = False
        prefs.cesiumIonAccessToken = 'ion-token'
        with patch.object(access, 'DownloadTransport') as transport:
            self.rejected('Google 3D Tiles Key')
            transport.assert_not_called()

    def test_direct_google_still_imports_mesh_tiles(self):
        sys.path.insert(0, str(ROOT / 'tests'))
        from gltf_test_support import fixture, make_glb
        from math import radians
        self.settings.google3dTilesAccessVia = 'google'
        self.settings.progressiveImport = False
        self.settings.concurrentDownloads = False
        self.settings.minLon, self.settings.maxLon = 24.93, 24.95
        self.settings.minLat, self.settings.maxLat = 60.16, 60.18
        prefs.googleMapsApiKey = 'direct-google-key'
        doc, binary, _ = fixture()
        center = manager.BaseManager.fromGeographic(60.17, 24.94, 0)
        doc['nodes'][0]['translation'] = [center.x, center.z, -center.y]
        glb = make_glb(doc, binary)
        root = json.dumps({'root': {'geometricError': 0,
            'boundingVolume': {'region': [radians(24.93), radians(60.16), radians(24.95), radians(60.18), -100, 100]},
            'content': {'uri': 'tile.glb'}}}).encode()
        requests = []
        def download(tile_manager, url):
            requests.append(url)
            return glb if '.glb' in url else root
        before = set(bpy.data.objects)
        try:
            with patch.object(manager.BaseManager, 'download', download), patch.object(access, 'DownloadTransport') as transport:
                self.assertEqual(bpy.ops.blosm.import_data(), {'FINISHED'})
                transport.assert_not_called()
            self.assertTrue(any(obj.type == 'MESH' for obj in set(bpy.data.objects) - before))
            self.assertEqual(requests[0], 'https://tile.googleapis.com/v1/3dtiles/root.json?key=direct-google-key')
            self.assertEqual(len(requests), 2)
        finally:
            for obj in set(bpy.data.objects) - before:
                bpy.data.objects.remove(obj, do_unlink=True)

    def test_resolution_only_operator_uses_ion_token_without_gltf_or_scene_changes(self):
        for progressive in (False, True):
            self.settings.progressiveImport = progressive
            prefs.cesiumIonAccessToken = '  ion-token\n'
            transport = MemoryTransport()
            before = scene_state()
            outcomes = []
            run = access.IonAccessResolution.run_blocking

            def record(task):
                value = run(task)
                outcomes.append(value)
                self.assertIsNone(task.job.result)
                return value

            with patch.object(access, 'DownloadTransport', return_value=transport), patch.object(
                    access.IonAccessResolution, 'run_blocking', record), patch.object(
                    manager.BaseManager, '__init__', side_effect=AssertionError('Unexpected tile manager')), patch.object(
                    renderer.BlenderRenderer, '__init__', side_effect=AssertionError('Unexpected renderer')):
                self.assertEqual(bpy.ops.blosm.import_data(), {'FINISHED'})
            self.assertEqual(scene_state(), before)
            self.assertEqual(outcomes, [('INFO', access.RESOLUTION_MESSAGE)])
            self.assertEqual(len(transport.requests), 1)
            self.assertEqual(transport.requests[0].headers['Authorization'], 'Bearer ion-token')
            self.assertEqual(transport.requests[0].url, ion.API_URL.format(asset_id=2275207))
            self.assertEqual(prefs.googleMapsApiKey, '')
            self.assertEqual(prefs.cesiumIonAccessToken, '  ion-token\n')
            self.assertTrue(transport.closed)

    def test_resolution_errors_are_reported_without_credentials(self):
        prefs.cesiumIonAccessToken = 'ion-token'
        for payload, error in ((b'not JSON provider-secret', None),
                               (None, RuntimeError('Access denied for ion-token'))):
            transport = MemoryTransport(payload, error)
            with patch.object(access, 'DownloadTransport', return_value=transport):
                self.rejected('invalid JSON|Access denied for \\[redacted\\]')
            self.assertTrue(transport.closed)

    def test_pending_job_is_cooperative_and_cancellable(self):
        transport = MemoryTransport(wait=True)
        before = scene_state()
        task = access.IonAccessResolution('ion-token', bpy.context, transport)
        task.job.step()
        self.assertTrue(task.job.waiting)
        self.assertFalse(task.job.done)
        self.assertEqual(task.revision, 0)
        task.cancel()
        self.assertTrue(task.job.done)
        self.assertTrue(transport.closed)
        with patch.object(access, 'notify_result') as notify:
            task.finish()
            notify.assert_called_once_with('INFO', 'Cesium ion access resolution cancelled')
        self.assertEqual(scene_state(), before)

    def test_get_token_button_opens_token_page(self):
        with patch('webbrowser.open_new_tab') as browser:
            self.assertEqual(bpy.ops.blosm.get_cesium_ion_token(), {'FINISHED'})
        browser.assert_called_once_with('https://ion.cesium.com/tokens')


suite = unittest.defaultTestLoader.loadTestsFromTestCase(BlenderChecks)
result = unittest.TextTestRunner(verbosity=2).run(suite)
(args.artifacts / 'results.json').write_text(json.dumps({
    'success': result.wasSuccessful(), 'tests_run': result.testsRun,
    'failures': len(result.failures), 'errors': len(result.errors),
    'blender': bpy.app.version_string, 'checkout': str(ROOT)}, indent=2), encoding='utf-8')
if not result.wasSuccessful():
    raise SystemExit(1)
