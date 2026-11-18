"""Realistic and import-for-export paths, both grouping states, real Pro textures."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import unittest
import hashlib
import random
import importlib
from unittest.mock import patch
from types import SimpleNamespace
import bpy
from release_support import (REPO, PACKAGE, arguments, enable, catalog, configure_osm,
                             meshes, geometry, run)

args = arguments()
enable(args)


class ExportChecks(unittest.TestCase):
    def check_import(self, fixture_id, single, export, experimental=True, folder=None):
        self.assertIsNotNone(args.assets, 'Required Pro assets: supply --assets')
        for name in ('building_materials.blend', 'building_material_templates.blend', 'version.txt'):
            self.assertTrue((args.assets / name).is_file(), f'Required Pro asset missing: {name}')
        entry = next(e for e in catalog() if e['id'] == fixture_id)
        settings = configure_osm(args, entry, '3Drealistic', single)
        settings.assetsDir = str(args.assets)
        settings.importForExport = export
        bpy.context.preferences.addons[PACKAGE].preferences.enableExperimentalFeatures = experimental
        # Keep each case's texture cache separate so every export case renders.
        folder = folder or args.artifacts / f'{fixture_id}_single_{single}_export_{export}_experimental_{experimental}'
        folder.mkdir(parents=True, exist_ok=True)
        bpy.context.preferences.addons[PACKAGE].preferences.dataDir = str(folder)
        real_render = bpy.ops.render.render
        rendered = []
        def render(*pos, **kw):
            scene = bpy.data.scenes[kw['scene']]
            rendered.append(scene.name)
            # Small deterministic CPU renders retain the actual compositor and asset inputs.
            scene.render.engine = 'CYCLES'
            scene.cycles.device = 'CPU'
            scene.cycles.samples = 1
            scene.render.threads_mode = 'FIXED'
            scene.render.threads = 2
            scene.render.resolution_percentage = 25
            return real_render(*pos, **kw)
        with patch.object(bpy.ops, 'render', SimpleNamespace(render=render)):
            self.assertEqual(bpy.ops.blosm.import_data(), {'FINISHED'})
        objects = meshes()
        self.assertTrue(objects)
        geometry(objects)
        if single:
            self.assertEqual(len(objects), 1)
        elif fixture_id == 'karl_marx_allee':
            self.assertGreater(len(objects), 1)
        for obj in objects:
            self.assertTrue(obj.data.materials)
            if experimental:
                self.assertEqual(len(obj.data.uv_layers), 1 if export else 2)
            else:
                self.assertIn('UVMap', obj.data.uv_layers)
                self.assertIn('size', obj.data.uv_layers)
            self.assertTrue(all(p.material_index < len(obj.data.materials) for p in obj.data.polygons))
        if export and experimental:
            self.assertTrue(rendered, 'Import-for-export did not generate any textures')
            textures = list(folder.rglob('*.png')) + list(folder.rglob('*.jpg'))
            self.assertTrue(textures)
            for path in textures:
                self.assertGreater(path.stat().st_size, 0)
            self.assertFalse([s for s in bpy.data.scenes if s.library], 'Linked template scene leaked')
            cached = {p: (p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest()) for p in textures}
            renders_before = len(rendered)
            random.seed(1729)
            with patch.object(bpy.ops, 'render', SimpleNamespace(render=render)):
                self.assertEqual(bpy.ops.blosm.import_data(), {'FINISHED'})
            self.assertEqual(len(rendered), renders_before, 'Unchanged import regenerated cached textures')
            self.assertEqual(cached, {p: (p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest())
                                     for p in textures})
            output = folder / 'roundtrip.glb'
            before_triangles = sum(sum(len(p.vertices)-2 for p in o.data.polygons) for o in objects)
            bpy.ops.object.select_all(action='DESELECT')
            for obj in objects:
                obj.select_set(True)
            self.assertEqual(bpy.ops.export_scene.gltf(filepath=str(output), export_format='GLB',
                                                      use_selection=True), {'FINISHED'})
            old_objects = set(bpy.data.objects)
            self.assertEqual(bpy.ops.import_scene.gltf(filepath=str(output)), {'FINISHED'})
            imported = [o for o in bpy.data.objects if o not in old_objects and o.type == 'MESH']
            self.assertEqual(sum(len(o.data.polygons) for o in imported), before_triangles)
            self.assertTrue(all(o.data.materials and o.data.uv_layers for o in imported))
        else:
            self.assertFalse(rendered, 'Normal realistic import unexpectedly baked export textures')

    def test_export_flag_requires_experimental_mode(self):
        self.check_import('rectangle', True, True, experimental=False)

    def test_export_with_deep_data_directory(self):
        # Reproduce the Windows failure: the old descriptive compositor filename
        # exceeded 260 characters, while the final cache filename still fitted.
        prefix = args.artifacts / 'deep-export'
        folder = prefix / ('x' * max(1, 180 - len(str(prefix)) - 1))
        self.check_import('rectangle', False, True, folder=folder)
        self.assertFalse(list(folder.rglob('.blosm-textures-*')), 'Private render directory leaked')

    def test_texture_render_errors_are_explicit_and_cleaned_up(self):
        TextureExporter = importlib.import_module(PACKAGE + '.building2.texture_exporter').TextureExporter
        for failure in ('cancelled', 'exception', 'missing', 'multiple', 'empty', 'stale'):
            with self.subTest(failure=failure):
                folder = args.artifacts / ('render-error-' + failure)
                folder.mkdir(exist_ok=True)
                exporter = TextureExporter(str(args.assets), str(args.assets))
                exporter.setTmpTextureDir(str(folder))
                private = Path(exporter.tmpTextureDir)
                destination = folder / 'result.png'
                def render(**kw):
                    if failure == 'exception':
                        raise RuntimeError('Injected compositor failure')
                    if failure in ('multiple', 'empty'):
                        (private / 't.png').write_bytes(b'image' if failure == 'multiple' else b'')
                    if failure == 'multiple':
                        (private / 'other.png').write_bytes(b'image')
                    return {'CANCELLED'} if failure == 'cancelled' else {'FINISHED'}
                if failure == 'stale':
                    (private / 'stale.png').write_bytes(b'old image')
                try:
                    with patch.object(bpy.ops, 'render', SimpleNamespace(render=render)):
                        with self.assertRaisesRegex(RuntimeError, 'scene "test scene".*destination .*temporary output'):
                            exporter.renderTexture(SimpleNamespace(name='test scene'), str(destination))
                    self.assertFalse(destination.exists(), 'Failed render published a texture')
                finally:
                    exporter.cleanup()
                self.assertFalse(private.exists(), 'Failed render directory leaked')


for fixture_id in ('rectangle', 'karl_marx_allee'):
    for single in (False, True):
        for export in (False, True):
            def check(self, fixture_id=fixture_id, single=single, export=export):
                self.check_import(fixture_id, single, export)
            setattr(ExportChecks, f'test_{fixture_id}_single_{single}_export_{export}', check)

run(args, ExportChecks)
