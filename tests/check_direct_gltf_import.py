"""Run with Blender --background --factory-startup --python-exit-code 1
--python this_file -- --artifacts PATH [--helsinki PATH].
"""
import argparse
import base64
import copy
import json
from pathlib import Path
import sys
from unittest.mock import patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gltf_test_support import (
    bpy, BlenderRenderer, fixture, make_glb, make_b3dm, convert_b3dm,
    reset_scene, fingerprint, no_tile_writes_or_operator, operator_import,
    baseline_renderer, manager_for, install_scene_settings, import_gltf, handler_counts, get_copyright,
)
from mathutils import Matrix
from io_scene_gltf2.io.imp.gltf2_io_gltf import glTFImporter

parser = argparse.ArgumentParser()
parser.add_argument("--artifacts", type=Path, required=True)
parser.add_argument("--helsinki", type=Path)
args = parser.parse_args(sys.argv[sys.argv.index("--")+1:])
args.artifacts.mkdir(parents=True, exist_ok=True)
install_scene_settings()
doc, binary, glb = fixture()
path = args.artifacts / "fixture.glb"
path.write_bytes(glb)
checks = []


def passed(name):
    checks.append(name)
    print("PASS", name)


# Compare the standard operator, direct file route, and direct bytes route.
reset_scene()
operator_import(path)
expected = fingerprint()
assert expected and expected[0]["materials"][0][0]["size"] == [2, 2]
for source in ("file", "bytes"):
    reset_scene()
    before = handler_counts()
    with no_tile_writes_or_operator():
        result = import_gltf(path if source == "file" else args.artifacts / "never_saved.glb",
                             content=glb if source == "bytes" else None)
    assert result.copyright == doc["asset"]["copyright"]
    assert fingerprint() == expected
    assert handler_counts() == before
    passed("operator parity: " + source)
assert not (args.artifacts / "never_saved.glb").exists()

# A JSON glTF with an embedded buffer uses the same direct entry point.
json_doc = copy.deepcopy(doc)
json_doc["buffers"][0]["uri"] = "data:application/octet-stream;base64," + base64.b64encode(binary).decode()
reset_scene()
with no_tile_writes_or_operator():
    import_gltf(args.artifacts / "never_saved.gltf", content=json.dumps(json_doc).encode())
assert fingerprint() == expected
passed("JSON glTF with embedded buffer")

# A meaningful base path still resolves external buffer and image files.
external = copy.deepcopy(doc)
external["buffers"][0]["uri"] = "fixture.bin"
image_data = binary[68:]
external["images"][0].pop("bufferView")
external["images"][0]["uri"] = "fixture.png"
(args.artifacts / "fixture.bin").write_bytes(binary)
(args.artifacts / "fixture.png").write_bytes(image_data)
reset_scene()
with no_tile_writes_or_operator():
    import_gltf(args.artifacts / "external.gltf", content=json.dumps(external).encode())
assert fingerprint() == expected
passed("external resource base path")

# All decoding/validation errors must propagate and remove logger handlers.
for name, bad in (
    ("truncated header", b"glTF"),
    ("truncated body", glb[:-9]),
    ("invalid JSON", b"{invalid"),
    ("unsupported version", b'{"asset":{"version":"9.0"}}'),
    ("unsupported extension", json.dumps({
        "asset": {"version": "2.0"}, "extensionsUsed": ["TEST_required"],
        "extensionsRequired": ["TEST_required"]}).encode()),
):
    reset_scene()
    before = handler_counts()
    reported = []
    try:
        with no_tile_writes_or_operator():
            import_gltf(args.artifacts / "bad.glb", content=bad,
                        report=lambda level, message: reported.append((level, message)))
    except Exception:
        assert reported == []
    else:
        raise AssertionError("Expected failure: " + name)
    assert handler_counts() == before
    assert not fingerprint()
    passed("error cleanup: " + name)

# Enabled importer extensions must still run.
from types import SimpleNamespace
seen = []
class Extension:
    def gather_import_gltf_before_hook(self, gltf):
        seen.append("before")
module_name = "blosm_direct_import_test_extension"
sys.modules[module_name] = SimpleNamespace(glTF2ImportUserExtension=Extension)
addon = bpy.context.preferences.addons.new()
addon.module = module_name
try:
    reset_scene()
    with no_tile_writes_or_operator():
        import_gltf(path, content=glb)
    assert seen == ["before"]
finally:
    bpy.context.preferences.addons.remove(addon)
    del sys.modules[module_name]
passed("enabled user extension")

if bpy.app.version >= (4, 5, 0):
    from io_scene_gltf2.blender.imp.blender_gltf import BlenderGlTF
    create = BlenderGlTF.create
    for failure in (False, True):
        reset_scene()
        reported = []
        before = handler_counts()
        def create_with_reports(importer):
            importer.log.warning("fixture warning", popup=True)
            importer.log.error("fixture nonfatal error", popup=True)
            if failure:
                raise RuntimeError("fixture fatal error")
            create(importer)
        with patch.object(BlenderGlTF, "create", side_effect=create_with_reports):
            try:
                result = import_gltf(path, content=glb,
                                     report=lambda level, message: reported.append((level, message)))
            except RuntimeError as exc:
                assert failure and str(exc) == "fixture fatal error"
            else:
                assert not failure
                assert result.messages == (('WARNING', 'fixture warning'), ('ERROR', 'fixture nonfatal error'))
        assert reported == ([] if failure else [
            ({'WARNING'}, "fixture warning"), ({'ERROR'}, "fixture nonfatal error")])
        assert handler_counts() == before
    passed("successful-import reports and no per-tile failure reports")

    ReleaseRenderer = baseline_renderer()
    transform = Matrix.Translation((5.0, -3.0, 2.0))

    def render(renderer_type, content, kind, join=False):
        reset_scene()
        renderer = renderer_type("Tile check", join, "")
        directory = args.artifacts / "renderer_baseline"
        directory.mkdir(exist_ok=True)
        filename = "test_tile." + kind
        manager = manager_for(renderer, directory, content)
        renderer.prepare(manager)
        try:
            method = renderer.renderB3dm if kind == "b3dm" else renderer.renderGlb
            if renderer_type is BlenderRenderer:
                with no_tile_writes_or_operator(no_dataset_files=True):
                    method(manager, "fixture", filename, transform)
            else:
                method(manager, "fixture", filename, transform, False)
            assert renderer.num_imported_tiles == 1
        finally:
            renderer.finalize(manager)
        return fingerprint(), get_copyright(bpy.context.scene.blosm)

    # B3DM with/without RTC: placement matches the exact release renderer.
    for rtc in (None, [2877435.0, 1343478.0, 5512247.0], [0.0, 0.0, 0.0]):
        b3dm = make_b3dm(glb, rtc)
        for join in (False, True):
            baseline, _ = render(ReleaseRenderer, b3dm, "b3dm", join=join)
            actual, attribution = render(BlenderRenderer, b3dm, "b3dm", join=join)
            assert baseline == actual, json.dumps({"rtc": rtc, "join": join, "baseline": baseline, "actual": actual})
            assert "City of Test" in attribution and "Café Example" in attribution
        passed("B3DM placement/RTC/join parity: " + str(rtc))

    # Both tile formats import downloaded bytes without local dataset access.
    glb_expected, copyright_expected = render(BlenderRenderer, glb, "glb")
    assert glb_expected
    assert "City of Test" in copyright_expected
    b3dm = make_b3dm(glb, [2877435.0, 1343478.0, 5512247.0])
    b3dm_expected = render(BlenderRenderer, b3dm, "b3dm")
    assert b3dm_expected[0] and "City of Test" in b3dm_expected[1]
    passed("GLB/B3DM imports without local dataset access")

    # Renderer leaves shared importer state untouched even if scene creation fails.
    from io_scene_gltf2.blender.imp.blender_gltf import BlenderGlTF
    reset_scene()
    renderer = BlenderRenderer("Failure", False, "")
    manager = manager_for(renderer, args.artifacts, b3dm)
    renderer.prepare(manager)
    before = handler_counts()
    try:
        with patch.object(BlenderGlTF, "create", side_effect=RuntimeError("injected failure")):
            try:
                renderer.renderB3dm(manager, "fixture", "failure.b3dm", transform)
            except RuntimeError as error:
                assert str(error) == "injected failure"
            else:
                raise AssertionError("Expected injected failure")
        assert not hasattr(glTFImporter, "_patch_convert_functions")
        assert handler_counts() == before
    finally:
        renderer.finalize(manager)
    assert not hasattr(glTFImporter, "_patch_convert_functions")
    passed("Importer isolation and logger cleanup after scene-creation failure")

    if args.helsinki:
        for tile in sorted(args.helsinki.glob("*.b3dm")):
            content = tile.read_bytes()
            expected_tile, _ = render(ReleaseRenderer, content, "b3dm")
            actual_tile, _ = render(BlenderRenderer, content, "b3dm")
            assert expected_tile == actual_tile and actual_tile
            extracted, _ = convert_b3dm(content)
            reset_scene()
            extracted_path = args.artifacts / (tile.stem + ".glb")
            extracted_path.write_bytes(extracted)
            operator_import(extracted_path)
            expected_glb = fingerprint()
            reset_scene()
            with no_tile_writes_or_operator():
                import_gltf(extracted_path, content=extracted)
            assert fingerprint() == expected_glb
            passed("Helsinki GLB and B3DM parity: " + tile.name)

report = {"blender": bpy.app.version_string, "status": "passed", "checks": checks}
(args.artifacts / "correctness.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print("DIRECT_IMPORT_CHECKS", json.dumps(report))
