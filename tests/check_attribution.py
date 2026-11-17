"""Registry persistence and real 3D Tiles imports in isolated Blender."""
import argparse
import importlib
import json
from math import radians
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
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
assert PACKAGE in bpy.context.preferences.addons
addon = importlib.import_module(PACKAGE)
attribution = importlib.import_module(PACKAGE + '.threed_tiles.attribution')
manager_module = importlib.import_module(PACKAGE + '.threed_tiles.manager')
renderers = importlib.import_module(PACKAGE + '.threed_tiles.blender')
progressive = importlib.import_module(PACKAGE + '.threed_tiles.progressive')
ion = importlib.import_module(PACKAGE + '.threed_tiles.cesium_ion')
access = importlib.import_module(PACKAGE + '.threed_tiles.ion_access')

BOUNDS = (24.93, 60.16, 24.95, 60.18)
REGION = {'region': [radians(24.93), radians(60.16), radians(24.95), radians(60.18), -100, 100]}


class MemoryTransport:
    max_requests = 4

    def __init__(self, credits=('Shared; First data', 'Shared; Second data'), *, wait=False):
        self.requests, self.pending = [], {}
        self.closed, self.wait = False, wait
        self.root = json.dumps({'root': {'geometricError': 0, 'boundingVolume': REGION,
            'children': [{'geometricError': 0, 'boundingVolume': REGION, 'content': {'uri': name}}
                         for name in ('first.glb', 'second.glb')]}}).encode()
        self.tiles = {}
        for name, text in zip(('first.glb', 'second.glb'), credits):
            doc, binary, _ = fixture()
            center = manager_module.BaseManager.fromGeographic(60.17, 24.94, 0)
            doc['nodes'][0]['translation'] = [center.x, center.z, -center.y]
            doc['asset']['copyright'] = text
            self.tiles[name] = make_glb(doc, binary)

    def resolve(self, url):
        if url == ion.API_URL.format(asset_id=2275207):
            return json.dumps({'type': '3DTILES', 'externalType': '3DTILES',
                'options': {'url': 'https://ion-fixture.invalid/root.json?key=provider-secret'},
                'attributions': [{'html': '<span>Cesium ion</span>'}]}).encode()
        filename = urlsplit(url).path.rsplit('/', 1)[-1]
        return self.tiles[filename] if filename.endswith('.glb') else self.root

    def submit(self, request):
        identity = len(self.requests)
        self.requests.append(request)
        self.pending[identity] = request.url
        return identity

    def poll(self):
        if self.pending and not self.wait:
            identity = next(iter(self.pending))
            value = self.resolve(self.pending.pop(identity))
            return (identity, None, value) if isinstance(value, Exception) else (identity, value, None)

    def close(self):
        self.closed = True


def state(settings):
    return (settings.get('copyright', ''), [
        (entry.as_pointer(), entry.collection.as_pointer() if entry.collection else None,
         list(entry.get('copyrights', ())), list(entry.get('counts', ())),
         list(entry.get('providerCredits', ()))) for entry in settings.importAttributions])


class AttributionChecks(unittest.TestCase):
    def setUp(self):
        reset_scene()
        bpy.context.scene.property_unset('blosm')
        self.settings = bpy.context.scene.blosm
        prefs = bpy.context.preferences.addons[PACKAGE].preferences
        prefs.googleMapsApiKey = 'direct-key'
        prefs.cesiumIonAccessToken = 'ion-token'

    def collection(self, name):
        collection = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(collection)
        return collection

    def task(self, *, via_ion=False, transport=None, concurrent=True):
        renderer = progressive.ProgressiveRenderer('Attribution fixture', False, '')
        manager_type = access.CesiumIonManager if via_ion else manager_module.BaseManager
        manager = manager_type('https://fixture.invalid/root.json', renderer)
        if via_ion:
            manager.setToken('ion-token')
        manager.centerLat, manager.centerLon = 60.17, 24.94
        manager.setGeometricError(1)
        manager.concurrentDownloads = concurrent
        return progressive.TilesImport(manager, BOUNDS, bpy.context, True, transport or MemoryTransport())

    def drive(self, task):
        for _ in range(300):
            task.job.step(max_steps=1)
            # Every published record must reference the permanent destination.
            for entry in self.settings.importAttributions:
                if attribution.live_collection(entry) is not None:
                    self.assertEqual(entry.collection, task.renderer.collection)
            if task.job.done:
                self.assertIsNone(task.job.error)
                return
        self.fail('Import did not finish')

    def test_mixed_blocking_imports_combine_and_deduplicate(self):
        settings = self.settings
        settings.dataType = '3d-tiles'
        settings.progressiveImport = False
        settings.minLon, settings.minLat, settings.maxLon, settings.maxLat = BOUNDS
        for index, (source, route, text) in enumerate((
                ('google', 'google', 'Google data'), ('google', 'cesium_ion', 'Ion data'),
                ('custom', 'google', 'Custom data'))):
            settings.threedTilesSource = source
            settings.google3dTilesAccessVia = route
            settings.threedTilesUrl = 'https://custom.invalid/root.json'
            settings.concurrentDownloads = index == 2
            settings.join3dTilesObjects = index == 1
            transport = MemoryTransport(('Shared; ' + text,) * 2)
            with patch.object(manager_module, 'DownloadTransport', return_value=transport), patch.object(
                    manager_module.BaseManager, 'download', lambda manager, url: transport.resolve(url)):
                self.assertEqual(bpy.ops.blosm.import_data(), {'FINISHED'})
            self.assertEqual(len(settings.importAttributions), index + 1)
            self.assertIn(text, attribution.get_copyright(settings))
        self.assertEqual(attribution.get_copyright(settings).count('Shared'), 1)
        self.assertTrue(all(text in attribution.get_copyright(settings) for text in ('Google data', 'Ion data', 'Custom data', 'Cesium ion')))
        self.assertEqual(settings.get('copyright', ''), '')  # No persisted display cache.
        for entry in settings.importAttributions:
            self.assertEqual(len(entry.get('copyrights')), 2)
            self.assertEqual(list(entry.get('counts')), [2, 2])

    def test_display_is_read_only_and_ignores_deleted_references(self):
        first, second = self.collection('First'), self.collection('Second')
        attribution.record_import(self.settings, first, {'A': 1})
        attribution.record_import(self.settings, second, {'B': 2})
        bpy.data.collections.remove(first)
        self.assertIsNone(self.settings.importAttributions[0].collection)
        before = state(self.settings)
        labels = []
        panel = SimpleNamespace(layout=SimpleNamespace(label=lambda **kw: labels.append(kw['text'])))
        for _ in range(5):
            self.assertTrue(addon.gui.BLOSM_PT_Copyright.poll(bpy.context))
            addon.gui.BLOSM_PT_Copyright.draw(panel, bpy.context)
        self.assertEqual(labels, ['B'] * 5)
        self.assertEqual(state(self.settings), before)
        self.assertEqual(len(self.settings.importAttributions), 2)
        bpy.data.collections.remove(second)
        self.assertFalse(addon.gui.BLOSM_PT_Copyright.poll(bpy.context))
        self.assertEqual(len(self.settings.importAttributions), 2)

    def test_display_requires_no_collection_hierarchy_or_scene_lookup(self):
        class ReferenceOnly:
            name = 'Valid reference'

            def __getattr__(self, name):
                raise AssertionError('Display must not inspect collection ' + name)

        properties = {'copyrights': ['Notice'], 'counts': [1], 'providerCredits': ['Provider']}
        entry = SimpleNamespace(collection=ReferenceOnly(), get=properties.get)
        settings = SimpleNamespace(importAttributions=[entry], get={}.get)
        self.assertEqual(attribution.get_copyright(settings), 'Notice; Provider')

    def test_successful_import_prunes_stale_entries(self):
        stale = self.collection('Deleted')
        attribution.record_import(self.settings, stale, {'Deleted data': 1})
        bpy.data.collections.remove(stale)
        task = self.task()
        self.drive(task)
        self.assertEqual(len(self.settings.importAttributions), 1)
        self.assertNotIn('Deleted data', attribution.get_copyright(self.settings))
        self.assertIn('First data', attribution.get_copyright(self.settings))

    def test_rename_unlink_and_hide_preserve_attribution(self):
        collection = self.collection('Original')
        attribution.record_import(self.settings, collection, {'Original data': 1})
        collection.name = 'Renamed'
        collection.hide_viewport = True
        bpy.context.scene.collection.children.unlink(collection)
        self.assertEqual(attribution.get_copyright(self.settings), 'Original data')
        self.assertEqual(self.settings.importAttributions[0].collection, collection)

    def test_recreated_collection_name_does_not_adopt_deleted_credits(self):
        collection = self.collection('Same name')
        attribution.record_import(self.settings, collection, {'Old data': 1})
        bpy.data.collections.remove(collection)
        replacement = self.collection('Same name')
        self.assertEqual(attribution.get_copyright(self.settings), '')
        attribution.record_import(self.settings, replacement, {'New data': 1})
        self.assertEqual(attribution.get_copyright(self.settings), 'New data')
        self.assertEqual(len(self.settings.importAttributions), 1)

    def test_progressive_import_has_one_record_and_keeps_ion_credits(self):
        for concurrent in (False, True):
            self.settings.importAttributions.clear()
            task = self.task(via_ion=True, concurrent=concurrent)
            self.drive(task)
            self.assertEqual(task.job.result, (2, []))
            self.assertEqual(len(self.settings.importAttributions), 1)
            self.assertEqual(attribution.get_copyright(self.settings).count('Shared'), 1)
            self.assertIn('Cesium ion', attribution.get_copyright(self.settings))
            self.assertEqual(self.settings.importAttributions[0]['providerCredits'], ['Cesium ion'])

    def test_cancellation_before_content_leaves_registry_unchanged(self):
        stale = self.collection('Stale')
        attribution.record_import(self.settings, stale, {'Old data': 1})
        bpy.data.collections.remove(stale)
        before = state(self.settings)
        task = self.task(via_ion=True, transport=MemoryTransport(wait=True))
        task.job.step()
        task.cancel()
        task.finish(silent=True)
        self.assertEqual(state(self.settings), before)

    def test_partial_cancellation_retains_only_committed_credits(self):
        task = self.task(via_ion=True)
        for _ in range(200):
            task.job.step(max_steps=1)
            if task.renderer.num_imported_tiles == 1:
                break
        self.assertEqual(task.renderer.num_imported_tiles, 1)
        self.assertEqual(len(self.settings.importAttributions), 1)
        task.cancel()
        task.finish(silent=True)
        self.assertIn('First data', attribution.get_copyright(self.settings))
        self.assertIn('Cesium ion', attribution.get_copyright(self.settings))
        self.assertNotIn('Second data', attribution.get_copyright(self.settings))

    def test_failed_tiles_do_not_contribute_credits(self):
        transport = MemoryTransport()
        transport.tiles['second.glb'] = b'invalid glTF'
        task = self.task(transport=transport)
        self.drive(task)
        self.assertEqual(task.job.result[0], 1)
        self.assertEqual(len(task.manager.errors), 1)
        self.assertIn('First data', attribution.get_copyright(self.settings))
        self.assertNotIn('Second data', attribution.get_copyright(self.settings))

    def test_successful_tiles_can_have_no_copyright_strings(self):
        for via_ion in (False, True):
            self.settings.importAttributions.clear()
            task = self.task(via_ion=via_ion, transport=MemoryTransport(('', '')))
            self.drive(task)
            self.assertEqual(task.job.result, (2, []))
            self.assertEqual(len(self.settings.importAttributions), 1)
            self.assertEqual(list(self.settings.importAttributions[0]['copyrights']), [])
            self.assertEqual(attribution.get_copyright(self.settings), 'Cesium ion' if via_ion else '')

    def test_all_failed_tiles_leave_registry_unchanged(self):
        transport = MemoryTransport()
        transport.tiles = {name: b'invalid glTF' for name in transport.tiles}
        before = state(self.settings)
        task = self.task(transport=transport, via_ion=True)
        self.drive(task)
        self.assertEqual(task.job.result[0], 0)
        self.assertEqual(state(self.settings), before)
        self.assertEqual(attribution.get_copyright(self.settings), '')

    def test_endpoint_or_root_failure_does_not_prune_registry(self):
        stale = self.collection('Deleted import')
        attribution.record_import(self.settings, stale, {'Old data': 1})
        bpy.data.collections.remove(stale)
        before = state(self.settings)
        for via_ion in (False, True):
            transport = MemoryTransport()
            with patch.object(transport, 'resolve', return_value=b'not JSON'):
                task = self.task(via_ion=via_ion, transport=transport)
                self.drive(task)
            self.assertEqual(len(task.job.result), 1)
            self.assertEqual(state(self.settings), before)

    def test_existing_legacy_text_is_preserved_and_deduplicated(self):
        self.assertNotIn('copyright', self.settings.bl_rna.properties)
        self.settings['copyright'] = 'Legacy data; Shared'
        collection = self.collection('New import')
        attribution.record_import(self.settings, collection, {'Shared': 3, 'New data': 1}, ['Provider'])
        self.assertEqual(attribution.get_copyright(self.settings), 'Shared; Legacy data; New data; Provider')
        self.assertEqual(self.settings['copyright'], 'Legacy data; Shared')
        bpy.data.collections.remove(collection)
        self.assertEqual(attribution.get_copyright(self.settings), 'Legacy data; Shared')

    def test_native_strings_and_collection_references_survive_save_reload(self):
        collection = self.collection('Persisted import')
        self.settings['copyright'] = 'Legacy data'
        attribution.record_import(self.settings, collection,
                                  {'Example © 2026': 2, 'Example © 2025': 1}, ['Cesium ion'])
        expected = attribution.get_copyright(self.settings)
        filepath = args.artifacts / 'attribution.blend'
        bpy.ops.wm.save_as_mainfile(filepath=str(filepath))
        bpy.ops.wm.open_mainfile(filepath=str(filepath))
        settings = bpy.context.scene.blosm
        self.assertEqual(attribution.get_copyright(settings), expected)
        self.assertEqual(settings.importAttributions[0].collection, bpy.data.collections['Persisted import'])
        self.assertEqual(settings.importAttributions[0]['copyrights'], ['Example © 2026', 'Example © 2025'])
        self.assertEqual(settings.importAttributions[0]['providerCredits'], ['Cesium ion'])

    def test_scene_registries_are_independent(self):
        collection = self.collection('Scene one import')
        attribution.record_import(self.settings, collection, {'First scene data': 1})
        other = bpy.data.scenes.new('Other scene')
        try:
            self.assertEqual(attribution.get_copyright(other.blosm), '')
            self.assertEqual(len(other.blosm.importAttributions), 0)
        finally:
            bpy.data.scenes.remove(other)


result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(AttributionChecks))
(args.artifacts / 'results.json').write_text(json.dumps({
    'status': 'passed' if result.wasSuccessful() else 'failed',
    'success': result.wasSuccessful(), 'tests_run': result.testsRun,
    'failures': len(result.failures), 'errors': len(result.errors),
    'blender': bpy.app.version_string}, indent=2), encoding='utf-8')
if not result.wasSuccessful():
    raise SystemExit(1)
