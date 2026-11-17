"""Install and exercise a built extension in an isolated Blender profile.

Run Blender with --background --factory-startup --python-exit-code 1
--python tests/check_extension.py -- --package ZIP --artifacts DIRECTORY.
Set BLENDER_USER_RESOURCES to a disposable directory before starting Blender.
"""
import argparse
import importlib
import json
import os
from pathlib import Path
import sys
from types import ModuleType
from unittest.mock import patch
import zipfile

import bpy
import addon_utils

sys.dont_write_bytecode = True
parser = argparse.ArgumentParser()
parser.add_argument('--package', type=Path, required=True)
parser.add_argument('--edition', choices=('base', 'pro'), default='pro')
parser.add_argument('--artifacts', type=Path, required=True)
args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
args.artifacts.mkdir(parents=True, exist_ok=True)
checks = []

with zipfile.ZipFile(args.package) as archive:
    names = archive.namelist()
    assert 'blender_manifest.toml' in names and '__init__.py' in names
    assert not any('__pycache__' in name or name.startswith(('tests/', 'script/'))
                   or name.endswith(('_access_token.txt', 'api_key.txt')) for name in names)
    assert {name for name in names if name.endswith('.blend')} == {
        'assets/base.blend', 'assets/roofs.blend', 'assets/way_profiles.blend'}
checks.append('package contents')

repo_dir = args.artifacts / 'repository'
repo_dir.mkdir(exist_ok=True)
repo = bpy.context.preferences.extensions.repos.new(
    name='Blosm compatibility test', module='blosm_test', custom_directory=str(repo_dir))
package = 'bl_ext.' + repo.module + '.blosm'
# Conflicting global module names must neither be replaced nor imported.
sentinels = {name: ModuleType(name) for name in ('app', 'util', 'building', 'pml', 'antlr4', 'py3dtiles')}
sys.modules.update(sentinels)
paths_before = list(sys.path)
result = bpy.ops.extensions.package_install_files(
    filepath=str(args.package.resolve()), repo=repo.module, enable_on_install=True)
assert result == {'FINISHED'}, result
assert package in bpy.context.preferences.addons
assert addon_utils.check(package)[1]
addon = importlib.import_module(package)
attribution = importlib.import_module(package + '.threed_tiles.attribution')
assert addon.BlosmPreferences.bl_idname == package
assert addon.blenderApp.app.isPro == (args.edition == 'pro')
assert ('realistic/__init__.py' in names) == (args.edition == 'pro')
assert sys.path == paths_before
assert all(sys.modules[name] is value for name, value in sentinels.items())
checks.append('install, enable, preferences and namespace isolation')

for name in ('setup.base', 'threed_tiles.blender', 'threed_tiles.progressive',
             'threed_tiles.py3dtiles.tileset.content.tile_content_reader'):
    importlib.import_module(package + '.' + name)
checks.append('shared OSM and 3D Tiles imports')

if args.edition == 'pro':
    for name in ('setup.premium_default', 'setup.realistic_dev', 'style'):
        importlib.import_module(package + '.' + name)
    pml_file = args.artifacts / 'test.pml'
    pml_file.write_text('footprint { height: 10; }', encoding='utf-8')
    store = importlib.import_module(package + '.style').StyleStore(
        type('App', (), {'pmlFilepath': str(pml_file), 'assetsDir': str(args.artifacts)})())
    assert 'test' in store.styles
    checks.append('Pro PML parse and style compilation')
else:
    assert not any(name.startswith(('realistic/', 'pml/', 'util/blender_extra/')) for name in names)
    checks.append('Base excludes Pro modules')

prefs = bpy.context.preferences.addons[package].preferences
prefs.dataDir = ''
addon.blenderApp.app.setDataDir(bpy.context, str(Path(addon.__file__).parent), package)
data_dir = Path(addon.blenderApp.app.dataDir)
assert data_dir.is_dir() and not data_dir.is_relative_to(repo_dir)
checks.append('writable data directory outside installation')

setup_file = args.artifacts / 'custom_setup.py'
setup_file.write_text('from ..app.blender import app\ndef setup(*args):\n    return app\n', encoding='utf-8')
loader = importlib.import_module(package + '.util.setup_script')
assert loader.load_setup_script(setup_file)() is addon.blenderApp.app
assert 'custom_setup' not in sys.modules
checks.append('namespaced custom setup script')

bpy.context.preferences.system.use_online_access = False
assert not bpy.app.online_access
transport = importlib.import_module(package + '.downloads.transport')
manager = importlib.import_module(package + '.threed_tiles.manager')
with patch.object(transport.subprocess, 'Popen', side_effect=AssertionError('offline helper started')), \
        patch('urllib.request.urlopen', side_effect=AssertionError('offline network request')):
    for operation in (
        lambda: transport.DownloadTransport(),
        lambda: manager.BaseManager('https://example.invalid/', None).download('https://example.invalid/'),
        lambda: addon.blenderApp.app.download('https://example.invalid/', str(args.artifacts / 'offline.data')),
    ):
        try:
            operation()
        except RuntimeError as exc:
            assert 'Online access is disabled' in str(exc)
        else:
            raise AssertionError('offline operation succeeded')
checks.append('offline requests blocked before network/helper creation')

overlay = importlib.import_module(package + '.overlay')
tile = object.__new__(overlay.Overlay)
tile.overlayDir = str(args.artifacts / 'imagery-cache')
tile.imageExtension = 'png'
tile.getTileUrl = lambda *args: 'https://example.invalid/tile.png'
tile.tileCounter = tile.numTiles = 1
with patch('urllib.request.urlopen', side_effect=AssertionError('offline imagery request')):
    try:
        tile.getTileData(1, 0, 0)
    except RuntimeError as exc:
        assert 'Online access is disabled' in str(exc)
    else:
        raise AssertionError('uncached offline imagery was allowed')
checks.append('offline imagery cache miss blocked')

# Exercise the real installed renderer with GLB and B3DM fixtures, including
# million-metre geocentric coordinates and failures, without external services.
assert sys.path == paths_before
os.environ['BLOSM_TEST_PACKAGE'] = package
sys.path.insert(0, str(Path(__file__).parent))
from gltf_test_support import fixture, make_glb, make_b3dm, reset_scene, fingerprint
from io_scene_gltf2.blender.imp.blender_gltf import BlenderGlTF
from io_scene_gltf2.blender.imp.scene import BlenderScene
from io_scene_gltf2.io.imp.gltf2_io_gltf import glTFImporter
render_module = importlib.import_module(package + '.threed_tiles.blender')
original_convert = BlenderGlTF.set_convert_functions
original_select = BlenderScene.select_imported_objects
center = manager.BaseManager.fromGeographic(60.17, 24.94, 0.)
doc, binary, glb = fixture()
b3dm = make_b3dm(glb, list(center))
doc['nodes'][0]['translation'] = [center.x + 1, center.z + 2, -center.y + 3]
glb = make_glb(doc, binary)
outputs = []
for kind, data in (('glb', glb), ('b3dm', b3dm)):
    reset_scene()
    renderer = render_module.BlenderRenderer('Extension test', False, '')
    test_manager = manager.BaseManager('https://example.invalid/', renderer)
    test_manager.centerLat, test_manager.centerLon = 60.17, 24.94
    renderer.prepare(test_manager)
    getattr(renderer, 'render' + ('Glb' if kind == 'glb' else 'B3dm'))(
        test_manager, 'fixture', 'fixture.' + kind, None, content=data)
    renderer.finalize(test_manager)
    meshes = [obj for obj in bpy.data.objects if obj.type == 'MESH']
    assert len(meshes) == 1 and len(meshes[0].data.vertices) == 3
    assert meshes[0].location.length < 10
    assert 'City of Test' in attribution.get_copyright(bpy.context.scene.blosm)
    assert BlenderGlTF.set_convert_functions is original_convert
    assert BlenderScene.select_imported_objects is original_select
    assert not hasattr(glTFImporter, '_patch_convert_functions')
    outputs.append(fingerprint())
assert outputs[0] == outputs[1]
checks.append('installed GLB/B3DM geometry, attribution and glTF isolation')
sys.path[:] = paths_before

reset_scene()
osm_file = args.artifacts / 'local.osm'
osm_file.write_text('''<osm version="0.6">
  <node id="1" lat="60.1700" lon="24.9400"/>
  <node id="2" lat="60.1700" lon="24.9402"/>
  <node id="3" lat="60.1701" lon="24.9402"/>
  <node id="4" lat="60.1701" lon="24.9400"/>
  <way id="1"><nd ref="1"/><nd ref="2"/><nd ref="3"/><nd ref="4"/><nd ref="1"/>
    <tag k="building" v="yes"/><tag k="height" v="10"/>
  </way></osm>''', encoding='utf-8')
settings = bpy.context.scene.blosm
prefs.dataDir = str(args.artifacts.resolve())
settings.dataType, settings.osmSource, settings.mode = 'osm', 'file', '2D'
settings.osmFilepath = str(osm_file)
settings.loadMissingMembers = False
settings.forests = settings.vegetation = settings.highways = settings.railways = settings.water = False
settings.buildings = True
with patch('urllib.request.urlopen', side_effect=AssertionError('local OSM attempted network')):
    assert bpy.ops.blosm.import_data() == {'FINISHED'}
assert any(obj.type in {'MESH', 'CURVE'} for obj in bpy.context.scene.objects)
checks.append('offline local OSM operator import')

assert sys.path == paths_before
assert all(sys.modules[name] is value for name, value in sentinels.items())
addon_utils.disable(package, default_set=True)
assert not hasattr(bpy.types.Scene, 'blosm')
assert addon_utils.enable(package, default_set=True) is not None
addon_utils.disable(package, default_set=True)
checks.append('disable, re-enable and cleanup')

report = {'edition': args.edition, 'status': 'passed', 'blender': bpy.app.version_string, 'checks': checks}
(args.artifacts / 'extension_validation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
