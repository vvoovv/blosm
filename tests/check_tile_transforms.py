"""Direct tile transforms versus Blender's original operator path.

Run using run_release_tests.py --suite tile_transforms. Optional --helsinki
adds the approved real B3DM fixtures. No normal Blender profile is used.
"""
import argparse
from contextlib import nullcontext
import importlib
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gltf_test_support import bpy, PACKAGE, fixture, make_glb, make_b3dm, reset_scene
from mathutils import Matrix, Quaternion
import addon_utils

parser = argparse.ArgumentParser()
parser.add_argument('--artifacts', required=True, type=Path)
parser.add_argument('--helsinki', type=Path)
args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
args.artifacts.mkdir(parents=True, exist_ok=True)
addon_utils.enable(PACKAGE, default_set=False)
bpy.context.preferences.filepaths.temporary_directory = str(args.artifacts)
bpy.context.preferences.filepaths.file_preview_type = 'NONE'
render = importlib.import_module(PACKAGE + '.threed_tiles.blender')
progressive = importlib.import_module(PACKAGE + '.threed_tiles.progressive')
manager_module = importlib.import_module(PACKAGE + '.threed_tiles.manager')
driver = importlib.import_module(PACKAGE + '.progressive.blender')
BaseManager, Renderer = manager_module.BaseManager, render.BlenderRenderer


class Ops:
    def __init__(self, calls, path=''):
        self.calls, self.path = calls, path
    def __getattr__(self, name):
        path = self.path + '.' + name if self.path else name
        if not self.path:
            return Ops(self.calls, path)
        original = getattr(getattr(bpy.ops, self.path), name)
        def call(*a, **kw):
            self.calls.append(path)
            return original(*a, **kw)
        return call


class BpyProxy:
    def __init__(self, calls):
        self.calls = calls
    def __getattr__(self, name):
        return Ops(self.calls) if name == 'ops' else getattr(bpy, name)


def mesh_object(collection, name='Tile'):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([(0, 0, 0), (3, 0, 0), (0, 4, 0), (0, 0, 5)], [],
                     [(0, 1, 2), (0, 3, 1)])
    for polygon in mesh.polygons:
        polygon.use_smooth = True
    uv = mesh.uv_layers.new(name='Tile UV')
    for index, value in enumerate(uv.data):
        value.uv = (index / 10, index / 20)
    color = mesh.color_attributes.new(name='Tile color', type='FLOAT_COLOR', domain='CORNER')
    for value in color.data:
        value.color = (.2, .4, .6, 1.)
    attr = mesh.attributes.new('Tile identifier', 'INT', 'FACE')
    for value in attr.data:
        value.value = 19
    mesh.normals_split_custom_set_from_vertices([(0., .6, .8)] * 4)
    material = bpy.data.materials.new(name + ' material')
    mesh.materials.append(material)
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    obj.rotation_euler = (.31, -.27, .53)
    obj.location = (15, -25, 40)
    return obj


def state():
    bpy.context.view_layer.update()
    result = {}
    for obj in bpy.context.scene.objects:
        data = dict(type=obj.type, parent=obj.parent.name if obj.parent else None,
                    matrix=[value for row in obj.matrix_world for value in row],
                    scale=list(obj.scale), selected=obj.select_get())
        if obj.type == 'MESH':
            mesh = obj.data
            if hasattr(mesh, 'calc_normals_split'):
                mesh.calc_normals_split()
            data.update(vertices=[list(obj.matrix_world @ vertex.co) for vertex in mesh.vertices],
                        polygons=[list(face.vertices) for face in mesh.polygons],
                        uv=[[list(value.uv) for value in layer.data] for layer in mesh.uv_layers],
                        colors=[[list(value.color) for value in layer.data] for layer in mesh.color_attributes],
                        materials=[value.name for value in mesh.materials],
                        normals=[list(value.vector) for value in mesh.corner_normals]
                                if hasattr(mesh, 'corner_normals') else [list(value.normal) for value in mesh.loops],
                        attributes={attr.name:[value.value for value in attr.data]
                                    for attr in mesh.attributes if attr.data_type == 'INT'},
                        keys=[[list(value.co) for value in key.data] for key in mesh.shape_keys.key_blocks]
                             if mesh.shape_keys else [])
        result[obj.name] = data
    return result


class TransformChecks(unittest.TestCase):
    def compare(self, expected, actual, tolerance=2e-5):
        if isinstance(expected, dict):
            self.assertEqual(set(expected), set(actual))
            for key in expected:
                with self.subTest(field=key):
                    self.compare(expected[key], actual[key], tolerance)
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(expected), len(actual))
            for a, b in zip(expected, actual):
                self.compare(a, b, tolerance)
        elif isinstance(expected, float):
            self.assertLessEqual(abs(expected - actual), tolerance, (expected, actual))
        else:
            self.assertEqual(expected, actual)

    def generate(self, setup, *, forced=False, join=False, height=True):
        reset_scene()
        outside = mesh_object(bpy.context.scene.collection, 'Existing user object')
        outside.select_set(True)
        bpy.context.view_layer.objects.active = outside
        renderer = Renderer('Transform test', join, '', recordAttribution=False)
        renderer.calculateHeightOffset = height
        manager = BaseManager('https://test.invalid/root.json', renderer)
        manager.centerLat, manager.centerLon = 60.17, 24.94
        renderer.prepare(manager)
        renderer.num_imported_tiles = 1
        after_update = setup(renderer.collection, outside)
        bpy.context.view_layer.update()
        if callable(after_update):
            after_update()
        calls = []
        scope = patch.object(Renderer, '_finalize_direct', return_value=False) if forced else patch.object(
            render, 'bpy', BpyProxy(calls))
        with scope:
            renderer.finalize(manager)
        return state(), renderer.heightOffset, calls

    def parity(self, setup, *, fast=True, join=False):
        expected, height, _ = self.generate(setup, forced=True, join=join)
        actual, actual_height, calls = self.generate(setup, join=join)
        self.compare(expected, actual)
        self.assertAlmostEqual(height, actual_height, places=5)
        if fast:
            self.assertEqual(calls, [], 'Eligible tile called a Blender operator')
            for name, obj in actual.items():
                if name != 'Existing user object' and obj['type'] == 'MESH':
                    self.assertEqual(obj['parent'], None)
                    matrix = Matrix([obj['matrix'][i:i+4] for i in range(0,16,4)])
                    self.assertLess(matrix.to_quaternion().angle, 1e-5)
        else:
            self.assertIn('object.transform_apply', calls)
        return actual

    def test_static_mesh_modes_attributes_normals_and_user_object(self):
        for mode in ('XYZ', 'ZYX', 'QUATERNION'):
            with self.subTest(mode=mode):
                def setup(collection, outside):
                    obj = mesh_object(collection)
                    obj.rotation_mode = mode
                    if mode == 'QUATERNION':
                        obj.rotation_quaternion = Quaternion((1., 2., 3.), .72)
                self.parity(setup)

    def test_empty_and_mesh_parent_hierarchies_and_fresh_rtc(self):
        for mesh_parent in (False, True):
            with self.subTest(mesh_parent=mesh_parent):
                def setup(collection, outside):
                    parent = mesh_object(collection, 'Parent') if mesh_parent else bpy.data.objects.new('Parent', None)
                    if not mesh_parent:
                        collection.objects.link(parent)
                    parent.location = (100, -200, 30)
                    parent.rotation_euler = (.2, -.1, .3)
                    obj = mesh_object(collection, 'Child')
                    obj.parent = parent
                    obj.matrix_parent_inverse = Matrix.Translation((3, -2, 7))
                    return lambda: setattr(parent, 'matrix_world', Matrix.Translation((7, 8, 9)) @ parent.matrix_world)
                self.parity(setup)

    def test_unsupported_objects_use_original_path(self):
        for kind in ('scale', 'negative_scale', 'shape_keys', 'modifier', 'constraint',
                     'animation', 'axis_angle', 'delta', 'hidden', 'unselectable',
                     'standalone_empty'):
            with self.subTest(kind=kind):
                def setup(collection, outside):
                    obj = mesh_object(collection)
                    if kind == 'scale': obj.scale = (2., 3., 4.)
                    elif kind == 'negative_scale': obj.scale = (-1., 1., 1.)
                    elif kind == 'shape_keys':
                        obj.shape_key_add(name='Basis')
                        key = obj.shape_key_add(name='Raised')
                        key.data[0].co.z += 2
                    elif kind == 'modifier': obj.modifiers.new('Bevel', 'BEVEL')
                    elif kind == 'constraint': obj.constraints.new('LIMIT_LOCATION')
                    elif kind == 'animation': obj.keyframe_insert('location', frame=1)
                    elif kind == 'axis_angle':
                        obj.rotation_mode = 'AXIS_ANGLE'
                        obj.rotation_axis_angle = (.4, 0, 0, 1)
                    elif kind == 'delta': obj.delta_rotation_euler = (.2, 0., 0.)
                    elif kind == 'hidden': obj.hide_set(True)
                    elif kind == 'unselectable': obj.hide_select = True
                    else: collection.objects.link(bpy.data.objects.new('Empty', None))
                self.parity(setup, fast=False)

    def test_external_parent_or_child_falls_back_before_mutation(self):
        for external in ('parent','child'):
            with self.subTest(external=external):
                reset_scene()
                renderer = Renderer('External hierarchy',False,'',recordAttribution=False)
                manager = BaseManager('https://test.invalid/root.json',renderer)
                manager.centerLat,manager.centerLon = 60.17,24.94
                renderer.prepare(manager)
                inside = mesh_object(renderer.collection,'Inside')
                outside = mesh_object(bpy.context.scene.collection,'Outside')
                if external=='parent': inside.parent=outside
                else: outside.parent=inside
                expected=state()
                self.assertFalse(renderer._finalize_direct(manager))
                self.compare(expected,state())

    def test_interleaved_mesh_empty_hierarchy_uses_legacy_flattening(self):
        def setup(collection, outside):
            root = bpy.data.objects.new('Root empty',None)
            collection.objects.link(root)
            root.location = (100,200,300)
            parent = mesh_object(collection,'Mesh parent')
            parent.parent = root
            middle = bpy.data.objects.new('Middle empty',None)
            collection.objects.link(middle)
            middle.parent = parent
            leaf = mesh_object(collection,'Leaf mesh')
            leaf.parent = middle
        self.parity(setup,fast=False)

    def test_shared_mesh_fallback_does_not_transform_unrelated_owner(self):
        reset_scene()
        renderer = Renderer('Shared mesh', False, '', recordAttribution=False)
        manager = BaseManager('https://test.invalid/root.json', renderer)
        manager.centerLat, manager.centerLon = 60.17, 24.94
        renderer.prepare(manager)
        obj = mesh_object(renderer.collection)
        other = bpy.data.objects.new('Outside shared user', obj.data)
        bpy.context.scene.collection.objects.link(other)
        before = [tuple(vertex.co) for vertex in other.data.vertices]
        bpy.context.view_layer.update()
        self.assertFalse(renderer._finalize_direct(manager))
        self.assertEqual(before, [tuple(vertex.co) for vertex in other.data.vertices])
        calls = []
        with patch.object(render, 'bpy', BpyProxy(calls)):
            try:
                renderer.finalize(manager)
            except RuntimeError:
                pass  # Blender rejects a selected mesh with an outside data user.
        self.assertIn('object.transform_apply', calls)
        self.assertEqual(before, [tuple(vertex.co) for vertex in other.data.vertices])

    def test_join_keeps_original_operator_behavior(self):
        def setup(collection, outside):
            mesh_object(collection, 'First')
            mesh_object(collection, 'Second').location.x += 20
        self.parity(setup, fast=False, join=True)

    def test_progressive_failed_transform_rolls_back_and_restores_context(self):
        reset_scene()
        outside = mesh_object(bpy.context.scene.collection, 'User object')
        outside.select_set(True)
        bpy.context.view_layer.objects.active = outside
        expected = state()
        before = {kind:set(getattr(bpy.data,kind)) for kind in ('objects','meshes','materials','images','collections')}
        renderer = progressive.ProgressiveRenderer('Failed tile', False, '')
        manager = BaseManager('https://test.invalid/root.json', renderer)
        manager.centerLat, manager.centerLon = 60.17, 24.94
        renderer.prepare(manager)
        doc, binary, _ = fixture()
        center = manager.fromGeographic(manager.centerLat, manager.centerLon, 0)
        doc['nodes'][0]['translation'] = [center.x, center.z, -center.y]
        original = Renderer._apply_rotation
        rotations = []
        def fail_second(obj):
            rotations.append(obj)
            if len(rotations) == 2:
                raise RuntimeError('injected second rotation failure')
            return original(obj)
        with patch.object(Renderer, '_apply_rotation', side_effect=fail_second):
            with self.assertRaisesRegex(RuntimeError, 'second rotation failure'):
                renderer.renderGlb(manager, 'failed', 'failed.glb', None, content=make_glb(doc,binary))
        self.compare(expected, state())
        self.assertEqual(bpy.context.view_layer.objects.active, outside)
        self.assertTrue(outside.select_get())
        self.assertEqual(renderer.num_imported_tiles, 0)
        for kind, values in before.items():
            self.assertEqual(values, set(getattr(bpy.data,kind)), kind)

    def test_real_and_synthetic_progressive_imports_match_operator_path(self):
        doc, binary, _ = fixture()
        center = BaseManager.fromGeographic(60.17,24.94,0)
        doc['nodes'][0]['translation'] = [center.x,center.z,-center.y]
        world_glb = make_glb(doc,binary)
        _, _, local_glb = fixture()
        samples = [('synthetic.glb', world_glb),
                   ('synthetic_rtc.b3dm', make_b3dm(local_glb,list(center))),
                   ('synthetic_no_rtc.b3dm', make_b3dm(world_glb))]
        if args.helsinki:
            files = sorted(args.helsinki.glob('*.b3dm'))
            self.assertTrue(files, 'No Helsinki benchmark files')
            samples.extend((path.name,path.read_bytes()) for path in files)
        for name, content in samples:
            with self.subTest(sample=name):
                snapshots = []
                for forced in (True,False):
                    reset_scene()
                    renderer = progressive.ProgressiveRenderer('Tile',False,'')
                    manager = BaseManager('https://test.invalid/root.json',renderer)
                    manager.centerLat,manager.centerLon = 60.17,24.94
                    renderer.prepare(manager)
                    calls=[]
                    with patch.object(render,'bpy',BpyProxy(calls)):
                        with patch.object(Renderer,'_finalize_direct',return_value=False) if forced else nullcontext():
                            method = renderer.renderB3dm if name.endswith('.b3dm') else renderer.renderGlb
                            method(manager,name,name,None,content=content)
                    self.assertEqual(renderer.num_imported_tiles,1)
                    if not forced:
                        self.assertNotIn('object.transform_apply',calls)
                    snapshots.append(state())
                self.compare(*snapshots,tolerance=.001)


class Result(unittest.TextTestResult):
    def __init__(self,*a,**kw):
        super().__init__(*a,**kw)
        self.records=[]
    def addSuccess(self,test):
        super().addSuccess(test)
        self.records.append(dict(id=test.id(),status='passed'))


suite=unittest.defaultTestLoader.loadTestsFromTestCase(TransformChecks)
result=unittest.TextTestRunner(verbosity=2,resultclass=Result).run(suite)
report=dict(status='passed' if result.wasSuccessful() else 'failed',blender=bpy.app.version_string,
            cases=result.records,failures=[dict(id=test.id(),detail=detail) for test,detail in result.failures+result.errors])
(args.artifacts/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
if not result.wasSuccessful():
    raise SystemExit(1)
