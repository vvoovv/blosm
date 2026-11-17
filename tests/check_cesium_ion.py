"""Blender UI, actual ion imports, credentials, failures and cancellation."""
import argparse
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from math import radians
from urllib.parse import urlparse, urlsplit, parse_qs
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from release_target import REPO as ROOT, PACKAGE, PROFILE, assert_target
if PROFILE == 'general':
    sys.path.insert(0, str(ROOT.parent))
import bpy
import addon_utils

parser = argparse.ArgumentParser()
parser.add_argument('--artifacts', type=Path, required=True)
args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
args.artifacts.mkdir(parents=True, exist_ok=True)
bpy.context.preferences.filepaths.temporary_directory = str(args.artifacts)
addon_utils.enable(PACKAGE, default_set=True)
assert_target(importlib.import_module(PACKAGE))
assert PACKAGE in bpy.context.preferences.addons, 'Addon registration failed'
addon = importlib.import_module(PACKAGE)
ion = importlib.import_module(PACKAGE + '.threed_tiles.cesium_ion')
attribution = importlib.import_module(PACKAGE + '.threed_tiles.attribution')
access = importlib.import_module(PACKAGE + '.threed_tiles.ion_access')
manager = importlib.import_module(PACKAGE + '.threed_tiles.manager')
renderer = importlib.import_module(PACKAGE + '.threed_tiles.blender')
prefs = bpy.context.preferences.addons[PACKAGE].preferences


class MemoryTransport:
    max_requests = 4

    def __init__(self, payloads=None, wait=False):
        self.payloads, self.wait = payloads or {}, wait
        self.requests, self.pending = [], {}
        self.closed = False

    def submit(self, request):
        identity = len(self.requests)
        self.requests.append(request)
        self.pending[identity] = request.url
        return identity

    def poll(self):
        if self.pending and not self.wait:
            identity = next(iter(self.pending))
            value = self.payloads[self.pending.pop(identity)]
            return (identity, None, value) if isinstance(value, Exception) else (identity, value, None)

    def close(self):
        self.closed = True


def inputs(hosted=False):
    from gltf_test_support import fixture, make_glb
    doc, binary, _ = fixture()
    center = manager.BaseManager.fromGeographic(60.17, 24.94, 0)
    doc['nodes'][0]['translation'] = [center.x, center.z, -center.y]
    doc['asset']['copyright'] = 'Tile fixture credit'
    glb = make_glb(doc, binary)
    bounds = {'region': [radians(24.93), radians(60.16), radians(24.95), radians(60.18), -100, 100]}
    root = {'root': {'geometricError': 0, 'boundingVolume': bounds,
                    'content': {'uri': '/v1/3dtiles/nested/tileset.json?session=fixture-session'}}}
    nested = {'root': {'geometricError': 0, 'boundingVolume': bounds,
                      'children': [{'geometricError': 0, 'boundingVolume': bounds,
                                    'content': {'uri': name}} for name in ('first.glb', '../second.glb')]}}
    origin = 'https://assets.ion.cesium.com' if hosted else 'https://tile.googleapis.com'
    query = '' if hosted else '?key=provider-secret'
    suffix = '?session=fixture-session' + ('' if hosted else '&key=provider-secret')
    url = origin + '/v1/3dtiles/root.json' + query
    endpoint = {'type': '3DTILES', 'attributions': [{'html': '<a href="https://credit.invalid"><img alt="Google Maps"></a> Cesium ion'}]}
    if hosted:
        endpoint.update(url=url, accessToken='hosted-secret')
    else:
        endpoint.update(externalType='3DTILES', options={'url': url})
    return {ion.API_URL.format(asset_id=2275207): json.dumps(endpoint).encode(),
            url: json.dumps(root).encode(),
            origin + '/v1/3dtiles/nested/tileset.json' + suffix: json.dumps(nested).encode(),
            origin + '/v1/3dtiles/nested/first.glb' + suffix: glb,
            origin + '/v1/3dtiles/second.glb' + suffix: glb}


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
        self.settings.progressiveImport = False
        self.settings.concurrentDownloads = False
        self.settings.minLon, self.settings.maxLon = 24.93, 24.95
        self.settings.minLat, self.settings.maxLat = 60.16, 60.18
        self.settings.dataType = '3d-tiles'
        self.initial_data = {name: set(getattr(bpy.data, name)) for name in ('objects', 'collections', 'meshes', 'materials', 'images')}
        self.settings.threedTilesSource = 'google'
        self.settings.google3dTilesAccessVia = 'cesium_ion'
        prefs.googleMapsApiKey = ''
        prefs.cesiumIonAccessToken = ''

    def tearDown(self):
        for name, initial in self.initial_data.items():
            for item in set(getattr(bpy.data, name)) - initial:
                getattr(bpy.data, name).remove(item, do_unlink=True)
        for key in ('lat', 'lon', 'height_offset'):
            if key in bpy.context.scene:
                del bpy.context.scene[key]

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
        self.assertIn(('prop', 'google3dTilesAccessVia', ''), layout.events)
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
        self.assertEqual(prop.subtype, 'NONE')

    def test_missing_ion_token_never_starts_network_or_renderer(self):
        for token in ('', ' \t\n '):
            prefs.cesiumIonAccessToken = token
            prefs.googleMapsApiKey = 'direct-google-key'
            with patch.object(manager, 'DownloadTransport') as transport:
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
        with patch.object(manager, 'DownloadTransport') as transport:
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
            with patch.object(manager.BaseManager, 'download', download), patch.object(manager, 'DownloadTransport') as transport:
                self.assertEqual(bpy.ops.blosm.import_data(), {'FINISHED'})
                transport.assert_not_called()
            self.assertTrue(any(obj.type == 'MESH' for obj in set(bpy.data.objects) - before))
            self.assertEqual(requests[0], 'https://tile.googleapis.com/v1/3dtiles/root.json?key=direct-google-key')
            self.assertEqual(len(requests), 2)
        finally:
            for obj in set(bpy.data.objects) - before:
                bpy.data.objects.remove(obj, do_unlink=True)

    def test_blocking_ion_imports_meshes_with_sequential_and_concurrent_downloads(self):
        prefs.cesiumIonAccessToken = '  ion-token\n'
        for concurrent in (False, True):
            self.settings.concurrentDownloads = concurrent
            transport = MemoryTransport(inputs())
            before = set(bpy.data.objects)
            with patch.object(manager, 'DownloadTransport', return_value=transport):
                self.assertEqual(bpy.ops.blosm.import_data(), {'FINISHED'})
            meshes = [obj for obj in set(bpy.data.objects) - before if obj.type == 'MESH']
            self.assertTrue(meshes)
            self.assertEqual(transport.requests[0].headers['Authorization'], 'Bearer ion-token')
            self.assertEqual(len(transport.requests), 5)
            self.assertTrue(all(not req.headers for req in transport.requests[1:]))
            self.assertTrue(all(req.verify_tls for req in transport.requests))
            self.assertIn('Google Maps', attribution.get_copyright(self.settings))
            self.assertIn('Cesium ion', attribution.get_copyright(self.settings))
            self.assertIn('Tile fixture credit', attribution.get_copyright(self.settings))
            self.assertTrue(transport.closed)
        self.assertEqual(prefs.googleMapsApiKey, '')
        self.assertEqual(prefs.cesiumIonAccessToken, '  ion-token\n')

    def test_resolution_errors_leave_scene_unchanged_and_redact_ion_token(self):
        prefs.cesiumIonAccessToken = 'ion-token'
        for value in (b'not JSON provider-secret', RuntimeError('Access denied for ion-token')):
            transport = MemoryTransport({ion.API_URL.format(asset_id=2275207): value})
            with patch.object(manager, 'DownloadTransport', return_value=transport):
                self.rejected('invalid JSON|Access denied for \\[redacted\\]')
            self.assertEqual(len(transport.requests), 1)
            self.assertTrue(transport.closed)

    def make_task(self, payloads, concurrent=False, wait=False):
        progressive = importlib.import_module(PACKAGE + '.threed_tiles.progressive')
        target = progressive.ProgressiveRenderer('Ion fixture', False, '')
        tile_manager = access.CesiumIonManager('', target)
        tile_manager.setToken('ion-token')
        tile_manager.centerLat, tile_manager.centerLon = 60.17, 24.94
        tile_manager.setGeometricError(1)
        tile_manager.concurrentDownloads = concurrent
        return progressive.TilesImport(tile_manager, (24.93, 60.16, 24.95, 60.18), bpy.context, True,
                                       MemoryTransport(payloads, wait))

    def test_pending_resolution_is_cancellable_without_scene_changes(self):
        task = self.make_task({}, wait=True)
        before = scene_state()
        task.job.step()
        self.assertTrue(task.job.waiting)
        self.assertEqual(task.revision, 0)
        self.assertIn('Resolving', task.status)
        task.cancel()
        task.finish(silent=True)
        self.assertTrue(task.job.done)
        self.assertTrue(task.job.transport.closed)
        self.assertEqual(task.manager._token, '')
        self.assertEqual(scene_state(), before)

    def test_progressive_imports_and_clears_provider_credentials(self):
        for concurrent in (False, True):
            for hosted in (False, True):
                task = self.make_task(inputs(hosted), concurrent)
                for _ in range(100):
                    task.job.step()
                    if task.job.done:
                        break
                self.assertTrue(task.job.done)
                self.assertIsNone(task.job.error)
                self.assertEqual(task.job.result, (2, []))
                task.finish(silent=True)
                requests = task.job.transport.requests
                expected = {'Authorization': 'Bearer hosted-secret'} if hosted else {}
                self.assertTrue(all(req.headers == expected for req in requests[1:]))
                self.assertEqual(task.manager._headers, {})
                self.assertEqual(task.manager._rootQuery, {})
                self.assertEqual(task.manager.rootUri, '')
                self.assertAlmostEqual(bpy.context.scene['lat'], 60.17)
                self.assertIn('Tile fixture credit', attribution.get_copyright(self.settings))

    def test_provider_credentials_and_sessions_do_not_cross_origins(self):
        target = access.CesiumIonManager('', SimpleNamespace())
        target.rootUri = 'https://tile.googleapis.com/root.json?key=provider-secret'
        target._origin = access.origin(target.rootUri)
        target._rootQuery = {'key': 'provider-secret', 'session': 'old-root-session'}
        target._headers = {'Authorization': 'Bearer hosted-secret'}
        uri, base = target.getUri(urlparse('nested/root.json?session=first'), 'https://tile.googleapis.com/')
        self.assertEqual(parse_qs(urlsplit(uri).query), {'key': ['provider-secret'], 'session': ['first']})
        other, _ = target.getUri(urlparse('https://other.invalid/tile.glb?own=value'), base)
        self.assertEqual(other, 'https://other.invalid/tile.glb?own=value')
        self.assertEqual(target.downloadRequest(other).headers, {})
        uri, _ = target.getUri(urlparse('../tile.glb?session=second&key=wrong'), base)
        self.assertEqual(parse_qs(urlsplit(uri).query), {'key': ['provider-secret'], 'session': ['second']})
        message = target.redactError('provider-secret hosted-secret', uri)
        self.assertNotIn('provider-secret', message)
        self.assertNotIn('hosted-secret', message)

    def test_tile_failure_keeps_successful_tiles_and_redacts_credentials(self):
        payloads = inputs()
        failed = next(url for url in payloads if 'second.glb' in url)
        payloads[failed] = RuntimeError('Denied provider-secret fixture-session')
        task = self.make_task(payloads, concurrent=True)
        for _ in range(100):
            task.job.step()
            if task.job.done:
                break
        self.assertEqual(task.job.result[0], 1)
        self.assertEqual(len(task.manager.errors), 1)
        self.assertNotIn('provider-secret', task.manager.errors[0])
        self.assertNotIn('fixture-session', task.manager.errors[0])
        self.assertIn('[redacted]', task.manager.errors[0])
        task.finish(silent=True)

    def test_get_token_button_opens_token_page(self):
        with patch('webbrowser.open_new_tab') as browser:
            self.assertEqual(bpy.ops.blosm.get_cesium_ion_token(), {'FINISHED'})
        browser.assert_called_once_with('https://ion.cesium.com/tokens')


suite = unittest.defaultTestLoader.loadTestsFromTestCase(BlenderChecks)
result = unittest.TextTestRunner(verbosity=2).run(suite)
(args.artifacts / 'results.json').write_text(json.dumps({
    'status': 'passed' if result.wasSuccessful() else 'failed',
    'success': result.wasSuccessful(), 'tests_run': result.testsRun,
    'failures': len(result.failures), 'errors': len(result.errors),
    'blender': bpy.app.version_string, 'checkout': str(ROOT)}, indent=2), encoding='utf-8')
if not result.wasSuccessful():
    raise SystemExit(1)
