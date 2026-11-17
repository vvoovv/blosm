"""Check the actual Blosm import operator with a deterministic mixed tileset."""
import argparse
from contextlib import redirect_stdout
import importlib
import io
import json
from math import radians
from pathlib import Path
import sys
from unittest.mock import patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gltf_test_support import (
    bpy, REPO, PACKAGE, fixture, make_glb, make_b3dm, reset_scene, no_tile_writes_or_operator,
)
import addon_utils

parser = argparse.ArgumentParser()
parser.add_argument("--artifacts", type=Path, required=True)
args = parser.parse_args(sys.argv[sys.argv.index("--")+1:])
args.artifacts.mkdir(parents=True, exist_ok=True)
existing_json_files = set(args.artifacts.rglob("*.json"))
package = PACKAGE
attribution = importlib.import_module(package + '.threed_tiles.attribution')
addon_utils.enable(package, default_set=True)
try:
    assert package in bpy.context.preferences.addons
    unused_data_dir = args.artifacts / "unused_data_directory"
    bpy.context.preferences.addons[package].preferences.dataDir = str(unused_data_dir)
    reset_scene()
    settings = bpy.context.scene.blosm
    settings.dataType = "3d-tiles"
    settings.progressiveImport = False  # This check exercises the blocking operator.
    settings.concurrentDownloads = False  # In-process fixture replaces sequential HTTP.
    settings.threedTilesSource = "custom"
    settings.threedTilesUrl = "https://fixture.invalid/tileset.json"
    settings.minLon, settings.maxLon = 24.93, 24.95
    settings.minLat, settings.maxLat = 60.16, 60.18
    settings.join3dTilesObjects = False
    assert "cacheJsonFiles" not in settings.bl_rna.properties
    assert "cache3dFiles" not in settings.bl_rna.properties
    manager_module = importlib.import_module(package + ".threed_tiles.manager")
    center = manager_module.BaseManager.fromGeographic(60.17, 24.94, 0)
    doc, binary, glb = fixture()
    b3dm = make_b3dm(glb, list(center))
    doc["nodes"][0]["translation"] = [center.x, center.z, -center.y]
    glb = make_glb(doc, binary)
    bounds = {"region": [radians(24.93), radians(60.16),
                         radians(24.95), radians(60.18), -100, 100]}
    tileset = {"asset": {"version": "1.0"}, "geometricError": 0, "root": {
        "geometricError": 0, "boundingVolume": bounds, "children": [
            {"geometricError": 0, "boundingVolume": bounds, "content": {"uri": "test.glb"}},
            {"geometricError": 0, "boundingVolume": bounds, "content": {"uri": "test.b3dm"}},
        ]}}
    root_tileset = {"asset": {"version": "1.0"}, "geometricError": 0, "root": {
        "geometricError": 0, "boundingVolume": bounds, "content": {"uri": "nested/tileset.json"}}}
    payloads = {"tileset.json": json.dumps(root_tileset).encode(),
                "nested/tileset.json": json.dumps(tileset).encode(),
                "nested/test.glb": glb, "nested/test.b3dm": b3dm}
    expected_requests = list(payloads)
    requested = []
    def download(self, uri):
        name = uri.removeprefix("https://fixture.invalid/").lstrip("/")
        requested.append(name)
        return payloads[name]
    renderer_module = importlib.import_module(package + ".threed_tiles.blender")
    real_import = renderer_module.import_gltf
    from io_scene_gltf2.blender.imp.blender_gltf import BlenderGlTF
    real_create = BlenderGlTF.create
    forwarded = []
    def import_with_report_check(filepath, *, content=None, report=None, **kwargs):
        assert report is not None, "Blosm did not supply its report callback"
        def forward(level, message):
            forwarded.append((sorted(level), message))
            report(level, message)
        return real_import(filepath, content=content, report=forward, **kwargs)
    def create_with_warning(importer):
        importer.log.warning("fixture glTF warning", popup=True)
        real_create(importer)
    with patch.object(manager_module.BaseManager, "download", download), \
            patch.object(renderer_module, "import_gltf", side_effect=import_with_report_check), \
            patch.object(BlenderGlTF, "create", side_effect=create_with_warning), \
            no_tile_writes_or_operator(no_dataset_files=True):
        result = bpy.ops.blosm.import_data()
    assert result == {"FINISHED"}
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    assert len(meshes) == 2
    assert all(len(obj.data.vertices) == 3 and len(obj.data.polygons) == 1 for obj in meshes)
    assert "City of Test" in attribution.get_copyright(settings) and "Café Example" in attribution.get_copyright(settings)
    assert requested == expected_requests
    assert forwarded == [(["WARNING"], "fixture glTF warning")] * 2
    successful_reports = list(forwarded)

    # A failed tile enters Blosm's error accounting without an individual UI
    # report; a later tile still imports and the operator finishes normally.
    reset_scene()
    payloads["nested/test.glb"] = b"invalid glTF JSON"
    requested.clear()
    forwarded.clear()
    tile_errors = []
    real_process_error = manager_module.BaseManager.processError
    def process_error(self, error, uri):
        tile_errors.append((str(error), uri))
        real_process_error(self, error, uri)
    ui_reports = []
    addon_module = sys.modules[package]
    real_import_tiles = addon_module.BLOSM_OT_ImportData.import3dTiles
    class ReportCapture:
        def __init__(self, operator):
            self.operator = operator
        def __getattr__(self, name):
            return getattr(self.operator, name)
        def report(self, level, message):
            ui_reports.append((sorted(level), message))
            self.operator.report(level, message)
    def import_with_ui_capture(self, context):
        return real_import_tiles(ReportCapture(self), context)
    console = io.StringIO()
    with patch.object(manager_module.BaseManager, "download", download), \
            patch.object(manager_module.BaseManager, "processError", process_error), \
            patch.object(renderer_module, "import_gltf", side_effect=import_with_report_check), \
            patch.object(addon_module.BLOSM_OT_ImportData, "import3dTiles", import_with_ui_capture), \
            redirect_stdout(console), \
            no_tile_writes_or_operator(no_dataset_files=True):
        assert bpy.ops.blosm.import_data() == {"FINISHED"}
    assert len(tile_errors) == 1 and tile_errors[0][1].endswith("test.glb")
    assert forwarded == []
    assert ui_reports == [(["INFO"],
        "Imported 1 3D Tiles. 1 errors occured during the import. See the System Console.")]
    assert tile_errors[0][0] in console.getvalue() and tile_errors[0][1] in console.getvalue()
    assert len([obj for obj in bpy.context.scene.objects if obj.type == "MESH"]) == 1
    assert requested == expected_requests
    partial_import_reports = list(ui_reports)

    # Reproduce the unsupported Gaussian splatting case: all 18 GLBs fail
    # validation, but only the final aggregate ERROR reaches the UI.
    reset_scene()
    requested.clear()
    tile_errors.clear()
    ui_reports.clear()
    doc["extensionsUsed"] = ["KHR_gaussian_splatting"]
    doc["extensionsRequired"] = ["KHR_gaussian_splatting"]
    unsupported_glb = make_glb(doc, binary)
    tileset["root"]["children"] = [
        {"geometricError": 0, "boundingVolume": bounds,
         "content": {"uri": "unsupported_%s.glb" % i}} for i in range(18)]
    payloads["nested/tileset.json"] = json.dumps(tileset).encode()
    payloads.update({"nested/unsupported_%s.glb" % i: unsupported_glb for i in range(18)})
    summary = "No 3D tiles were imported. 18 errors occured during the import. See the System Console."
    console = io.StringIO()
    with patch.object(manager_module.BaseManager, "download", download), \
            patch.object(manager_module.BaseManager, "processError", process_error), \
            patch.object(renderer_module, "import_gltf", side_effect=import_with_report_check), \
            patch.object(addon_module.BLOSM_OT_ImportData, "import3dTiles", import_with_ui_capture), \
            redirect_stdout(console), \
            no_tile_writes_or_operator(no_dataset_files=True):
        try:
            bpy.ops.blosm.import_data()
        except RuntimeError as exc:
            assert summary in str(exc) and "KHR_gaussian_splatting" not in str(exc)
        else:
            raise AssertionError("Expected Blosm's aggregate error when all tiles fail")
    assert len(tile_errors) == 18
    assert all("Extension KHR_gaussian_splatting is not available" in message
               for message, uri in tile_errors)
    assert ui_reports == [(["ERROR"], summary)]
    assert forwarded == []
    assert not [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    assert requested == ["tileset.json", "nested/tileset.json"] + [
        "nested/unsupported_%s.glb" % i for i in range(18)]
    assert all(message in console.getvalue() and uri in console.getvalue()
               for message, uri in tile_errors)
    (args.artifacts / "failed_tiles_console.log").write_text(console.getvalue(), encoding="utf-8")
    assert not unused_data_dir.exists(), "3D Tiles import created a data directory"
    assert set(args.artifacts.rglob("*.json")) == existing_json_files
    assert not list(args.artifacts.rglob("*.glb"))
    assert not list(args.artifacts.rglob("*.b3dm"))
    report = {"blender": bpy.app.version_string, "status": "passed",
              "meshes": len(meshes), "requests": requested,
              "tile_files_written": 0, "gltf_operator_calls": 0,
              "forwarded_gltf_reports": successful_reports,
              "failed_tile_reports": forwarded, "continued_after_tile_error": True,
              "partial_import_reports": partial_import_reports,
              "all_failed_reports": ui_reports, "failed_tiles": len(tile_errors),
              "copyright": attribution.get_copyright(settings)}
    (args.artifacts / "operator_check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("BLOSM_OPERATOR_CHECK", json.dumps(report))
finally:
    addon_utils.disable(package, default_set=True)
