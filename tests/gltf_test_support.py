"""Fixtures/helpers for Blender-run direct glTF checks and benchmarks."""

from contextlib import contextmanager
import hashlib
import importlib.util
import importlib
import io
import json
import logging
import os
from pathlib import Path
import struct
import subprocess
import sys
from unittest.mock import patch
import zlib

import bpy
import numpy as np

from release_target import REPO, PACKAGE, PROFILE
sys.dont_write_bytecode = True
if PROFILE == 'general' and PACKAGE == REPO.name:
    sys.path.insert(0, str(REPO.parent))
BlenderRenderer = importlib.import_module(PACKAGE + '.threed_tiles.blender').BlenderRenderer
get_copyright = importlib.import_module(PACKAGE + '.threed_tiles.attribution').get_copyright
import_gltf = importlib.import_module(PACKAGE + '.threed_tiles.gltf_import').import_gltf
BaseManager = importlib.import_module(PACKAGE + '.threed_tiles.manager').BaseManager
read_array = importlib.import_module(PACKAGE + '.threed_tiles.py3dtiles.tileset.content.tile_content_reader').read_array


def png_bytes():
    def chunk(kind, payload):
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(
            ">I", zlib.crc32(kind + payload) & 0xffffffff)
    row = b"\xff\x00\x00\xff\x00\xff\x00\xff"
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 2, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(b"\0" + row + b"\0" + row))
            + chunk(b"IEND", b""))


def make_glb(document, binary):
    text = json.dumps(document, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 4)
    body = binary + b"\0" * (-len(binary) % 4)
    return (struct.pack("<4sII", b"glTF", 2, 28 + len(text) + len(body))
            + struct.pack("<I4s", len(text), b"JSON") + text
            + struct.pack("<I4s", len(body), b"BIN\0") + body)


def fixture():
    positions = struct.pack("<9f", 0, 0, 0, 2, 0, 0, 0, 3, 0)
    uv = struct.pack("<6f", 0, 0, 1, 0, 0, 1)
    indices = struct.pack("<3H", 0, 1, 2)
    image = png_bytes()
    binary = positions + uv + indices + b"\0\0" + image
    doc = {
        "asset": {"version": "2.0", "copyright": "City of Test; Café Example © 2026"},
        "scene": 0, "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "Fixture", "translation": [1, 2, 3]}],
        "meshes": [{"primitives": [{
            "attributes": {"POSITION": 0, "TEXCOORD_0": 1}, "indices": 2, "material": 0}]}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
             "min": [0, 0, 0], "max": [2, 3, 0]},
            {"bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC2"},
            {"bufferView": 2, "componentType": 5123, "count": 3, "type": "SCALAR"}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": 36, "target": 34962},
            {"buffer": 0, "byteOffset": 36, "byteLength": 24, "target": 34962},
            {"buffer": 0, "byteOffset": 60, "byteLength": 6, "target": 34963},
            {"buffer": 0, "byteOffset": 68, "byteLength": len(image)}],
        "buffers": [{"byteLength": len(binary)}],
        "images": [{"bufferView": 3, "mimeType": "image/png", "name": "FixtureTexture"}],
        "textures": [{"source": 0}],
        "materials": [{"pbrMetallicRoughness": {
            "baseColorTexture": {"index": 0}, "metallicFactor": 0}}],
    }
    return doc, binary, make_glb(doc, binary)


def make_b3dm(glb, rtc=None):
    feature = {"BATCH_LENGTH": 0}
    if rtc is not None:
        feature["RTC_CENTER"] = rtc
    text = json.dumps(feature, separators=(",", ":")).encode()
    text += b" " * (-(28 + len(text)) % 8)
    return struct.pack("<4s6I", b"b3dm", 1, 28 + len(text) + len(glb),
                       len(text), 0, 0, 0) + text + glb


def convert_b3dm(content):
    tile = read_array(np.frombuffer(content, dtype=np.uint8))
    return tile.body.gltf.to_array(), tile.body.feature_table.header.data.get("RTC_CENTER")


def reset_scene():
    bpy.context.view_layer.active_layer_collection = bpy.context.view_layer.layer_collection
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for collection in list(bpy.data.collections):
        bpy.data.collections.remove(collection)
    for data in (bpy.data.meshes, bpy.data.materials, bpy.data.images, bpy.data.cameras, bpy.data.lights):
        for item in list(data):
            if item.users == 0:
                data.remove(item)


def fingerprint():
    """Compare content and placement, ignoring generated object/material names."""
    bpy.context.view_layer.update()
    meshes = []
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue
        mesh = obj.data
        coords = np.empty(len(mesh.vertices)*3, dtype=np.float32)
        mesh.vertices.foreach_get("co", coords)
        verts = np.empty(len(mesh.loops), dtype=np.int32)
        mesh.loops.foreach_get("vertex_index", verts)
        material_indices = np.empty(len(mesh.polygons), dtype=np.int32)
        mesh.polygons.foreach_get("material_index", material_indices)
        uv = []
        for layer in mesh.uv_layers:
            values = np.empty(len(layer.data)*2, dtype=np.float32)
            layer.data.foreach_get("uv", values)
            uv.append(hashlib.sha256(values.tobytes()).hexdigest())
        colors = []
        for layer in mesh.color_attributes:
            values = np.empty(len(layer.data)*4, dtype=np.float32)
            layer.data.foreach_get("color", values)
            colors.append(hashlib.sha256(values.tobytes()).hexdigest())
        materials = []
        for material in mesh.materials:
            textures = []
            for node in material.node_tree.nodes if material and material.use_nodes else []:
                if node.type == "TEX_IMAGE" and node.image:
                    img = node.image
                    textures.append({
                        "size": list(img.size),
                        "packed": hashlib.sha256(bytes(img.packed_file.data)).hexdigest()
                        if img.packed_file else None,
                    })
            materials.append(textures)
        meshes.append({
            "vertices": len(mesh.vertices), "polygons": len(mesh.polygons),
            "coordinates": hashlib.sha256(coords.tobytes()).hexdigest(),
            "topology": hashlib.sha256(verts.tobytes()).hexdigest(),
            "material_indices": hashlib.sha256(material_indices.tobytes()).hexdigest(),
            "matrix": [round(v, 6) for row in obj.matrix_world for v in row],
            "uv": uv, "colors": colors, "materials": materials,
        })
    return sorted(meshes, key=lambda m: json.dumps(m, sort_keys=True))


@contextmanager
def no_tile_writes_or_operator(*, no_dataset_files=False):
    """Guard both actual file writes and invocation of the glTF import operator."""
    import builtins
    old_open, old_io_open, old_op = builtins.open, io.open, bpy.ops._op_call
    def guard_open(original, filename, mode="r", *args, **kwargs):
        if isinstance(filename, (str, bytes, Path)):
            suffix = Path(filename).suffix.lower()
            if no_dataset_files and suffix in {".json", ".glb", ".gltf", ".b3dm"}:
                raise AssertionError("3D Tiles import attempted to access a dataset file on disk")
            if suffix in {".glb", ".gltf", ".b3dm"} and any(c in mode for c in "wax+"):
                raise AssertionError("Direct import attempted to write a tile")
        return original(filename, mode, *args, **kwargs)
    def guard_op(name, *args, **kwargs):
        if name in {"IMPORT_SCENE_OT_gltf", "import_scene.gltf"}:
            raise AssertionError("Direct import invoked the glTF operator")
        return old_op(name, *args, **kwargs)
    with patch("builtins.open", lambda *a, **kw: guard_open(old_open, *a, **kw)), \
            patch("io.open", lambda *a, **kw: guard_open(old_io_open, *a, **kw)), \
            patch("bpy.ops._op_call", guard_op):
        yield


def operator_import(filepath):
    kwargs = {"filepath": str(filepath)}
    if "import_scene_as_collection" in bpy.ops.import_scene.gltf.get_rna_type().properties:
        kwargs["import_scene_as_collection"] = True
    assert bpy.ops.import_scene.gltf(**kwargs) == {"FINISHED"}


BASELINE_REF = "4adff00270f4492667f3bf1305d0b338f3cd1268"


def baseline_renderer():
    """Load the pre-direct-import renderer from its fixed historical revision."""
    source = subprocess.check_output(
        ["git", "-C", str(REPO), "show", BASELINE_REF + ":threed_tiles/blender.py"], text=True)
    # Adapt only the historical module's imports to the current package layout.
    source = source.replace('from util.blender import', 'from ..util.blender import')
    source = source.replace('from py3dtiles.', 'from .py3dtiles.')
    # The historical renderer writes the removed RNA property. Preserve that
    # baseline behavior in the legacy custom-property slot for this test only.
    source = source.replace('blosm.copyright', 'blosm["copyright"]')
    name = PACKAGE + ".threed_tiles._release_renderer"
    spec = importlib.util.spec_from_loader(name, loader=None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(source, BASELINE_REF + ":threed_tiles/blender.py", "exec"), module.__dict__)
    return module.BlenderRenderer


def manager_for(renderer, directory, content):
    manager = BaseManager("https://fixture.invalid/root.json", renderer)
    manager.centerLat, manager.centerLon = 60.17, 24.94
    # Only the historical operator renderer needs a temporary file directory.
    if renderer.__class__.__module__ == PACKAGE + ".threed_tiles._release_renderer":
        manager.tilesDir = str(directory)
    manager.download = lambda uri: content
    return manager


def install_scene_settings():
    if not hasattr(bpy.types.Scene, "blosm"):
        attribution = importlib.import_module(PACKAGE + '.threed_tiles.attribution')
        bpy.utils.register_class(attribution.BlosmImportAttribution)
        class TestSettings(bpy.types.PropertyGroup):
            importAttributions: bpy.props.CollectionProperty(type=attribution.BlosmImportAttribution)
        bpy.utils.register_class(TestSettings)
        bpy.types.Scene.blosm = bpy.props.PointerProperty(type=TestSettings)


def handler_counts():
    return tuple(len(logging.getLogger(name).handlers)
                 for name in ("glTFImporter", "glTFImporter_errors"))
