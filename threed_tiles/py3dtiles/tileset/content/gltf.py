from __future__ import annotations

import json
import struct
from typing import Any, Sequence

import numpy as np
import numpy.typing as npt


class GlTF:
    HEADER_LENGTH = 12
    CHUNK_HEADER_LENGTH = 8

    def __init__(self) -> None:
        self.header = {}
        self.body = None

    def to_array(self) -> npt.NDArray[np.uint8]:  # glb
        scene = json.dumps(self.header, separators=(",", ":"))

        # body must start with 4-byte boundary
        scene += " " * ((4 - len(scene) % 4) % 4)

        length = GlTF.HEADER_LENGTH + (2 * GlTF.CHUNK_HEADER_LENGTH)
        length += len(self.body) + len(scene)

        # gltf must end with an 8-byte boundary
        # the padding is added inside the gltf body
        # https://github.com/CesiumGS/3d-tiles/tree/main/specification/TileFormats/Batched3DModel#padding
        body_padding = np.zeros((8 - length % 8) % 8, dtype=np.uint8)
        length += len(body_padding)

        binary_header = np.array(
            [0x46546C67, 2, length], dtype=np.uint32  # "glTF" magic  # version
        )
        json_chunk_header = np.array(
            [len(scene), 0x4E4F534A], dtype=np.uint32  # JSON chunck length
        )  # "JSON"

        bin_chunk_header = np.array(
            [
                len(self.body) + len(body_padding),
                # BIN chunck length
                0x004E4942,
            ],
            dtype=np.uint32,
        )  # "BIN"

        return np.concatenate(
            (
                binary_header.view(np.uint8),
                json_chunk_header.view(np.uint8),
                np.frombuffer(scene.encode("utf-8"), dtype=np.uint8),
                bin_chunk_header.view(np.uint8),
                self.body,
                body_padding,
            )
        )

    @staticmethod
    def from_array(array: npt.NDArray[np.uint8]) -> GlTF:
        gltf = GlTF()

        if struct.unpack("4s", array[0:4])[0] != b"glTF":
            raise RuntimeError("Array does not contain a binary glTF")

        version = struct.unpack("i", array[4:8])[0]
        if version != 1 and version != 2:
            raise RuntimeError("Unsupported glTF version")

        length = struct.unpack("i", array[8:12])[0]
        json_chunk_length = struct.unpack("i", array[12:16])[0]

        chunk_type = struct.unpack("i", array[16:20])[0]
        if chunk_type != 0 and chunk_type != 1313821514:  # 1313821514 => 'JSON'
            raise RuntimeError("Unsupported binary glTF content type")

        index = (
            GlTF.HEADER_LENGTH + GlTF.CHUNK_HEADER_LENGTH
        )  # Skip the header and the JSON chunk header
        header = struct.unpack(
            str(json_chunk_length) + "s", array[index : index + json_chunk_length]
        )[0]
        gltf.header = json.loads(header.decode("ascii"))

        index += (
            json_chunk_length + GlTF.CHUNK_HEADER_LENGTH
        )  # Skip the JSON chunk data and the binary chunk header
        gltf.body = array[index:length]

        return gltf

    @staticmethod
    def from_binary_arrays(
        arrays: list[dict[str, Sequence[Any]]],
        transform: npt.NDArray[np.float64],
        batched: bool = True,
        uri: str | None = None,
        texture_uri: str | None = None,
    ) -> GlTF:
        """
        Parameters
        ----------
        arrays : array of dictionaries
            Each dictionary has the data for one geometry
            arrays['position']: binary array of vertex positions
            arrays['normal']: binary array of vertex normals
            arrays['uv']: binary array of vertex texture coordinates
                          (Not implemented yet)
            arrays['bbox']: geometry bounding box (numpy.array)

        transform : numpy.array
            World coordinates transformation flattend matrix

        Returns
        -------
        gltf : GlTF
        """

        gltf = GlTF()

        textured = "uv" in arrays[0]
        bin_vertices = []
        bin_normals = []
        bin_ids = []
        bin_uvs = []
        n_vertices = []
        bb = []
        batch_length = 0
        for i, geometry in enumerate(arrays):
            bin_vertices.append(geometry["position"])
            bin_normals.append(geometry["normal"])
            n = round(len(geometry["position"]) / 12)
            n_vertices.append(n)
            bb.append(geometry["bbox"])
            if batched:
                bin_ids.append(np.full(n, i, dtype=np.float32))
            if textured:
                bin_uvs.append(geometry["uv"])

        if batched:
            bin_vertices = [b"".join(bin_vertices)]
            bin_normals = [b"".join(bin_normals)]
            bin_uvs = [b"".join(bin_uvs)]
            bin_ids = [b"".join(bin_ids)]
            n_vertices = [sum(n_vertices)]
            batch_length = len(arrays)
            [minx, miny, minz] = bb[0][0]
            [maxx, maxy, maxz] = bb[0][1]
            for box in bb[1:]:
                minx = min(minx, box[0][0])
                miny = min(miny, box[0][1])
                minz = min(minz, box[0][2])
                maxx = max(maxx, box[1][0])
                maxy = max(maxy, box[1][1])
                maxz = max(maxz, box[1][2])
            bb = [[[minx, miny, minz], [maxx, maxy, maxz]]]

        gltf.header = compute_header(
            bin_vertices,
            n_vertices,
            bb,
            transform,
            textured,
            batched,
            batch_length,
            uri,
            texture_uri,
        )
        gltf.body = np.frombuffer(
            compute_binary(bin_vertices, bin_normals, bin_ids, bin_uvs), dtype=np.uint8
        )

        return gltf


def compute_binary(bin_vertices, bin_normals, bin_ids, bin_uvs):
    bv = b"".join(bin_vertices)
    bn = b"".join(bin_normals)
    bid = b"".join(bin_ids)
    buv = b"".join(bin_uvs)
    return bv + bn + buv + bid


def compute_header(
    bin_vertices,
    n_vertices,
    bb,
    transform,
    textured,
    batched,
    batch_length,
    uri,
    texture_uri,
):
    # Buffer
    mesh_nb = len(bin_vertices)
    size_vce = []
    for i in range(mesh_nb):
        size_vce.append(len(bin_vertices[i]))

    byte_length = 2 * sum(size_vce)
    if textured:
        byte_length += int(round(2 * sum(size_vce) / 3))
    if batched:
        byte_length += int(round(sum(size_vce) / 3))
    buffers = [{"byteLength": byte_length}]
    if uri is not None:
        buffers.append({"binary_glTF": {"uri": uri}})

    # Buffer view
    buffer_views = [
        {"buffer": 0, "byteLength": sum(size_vce), "byteOffset": 0, "target": 34962},
        {
            "buffer": 0,
            "byteLength": sum(size_vce),
            "byteOffset": sum(size_vce),
            "target": 34962,
        },
    ]
    # vertices
    if textured:
        buffer_views.append(
            {
                "buffer": 0,
                "byteLength": int(round(2 * sum(size_vce) / 3)),
                "byteOffset": 2 * sum(size_vce),
                "target": 34962,
            }
        )
    if batched:
        buffer_views.append(
            {
                "buffer": 0,
                "byteLength": int(round(sum(size_vce) / 3)),
                "byteOffset": int(round(8 / 3 * sum(size_vce)))
                if textured
                else 2 * sum(size_vce),
                "target": 34962,
            }
        )

    # Accessor
    accessors = []
    for i in range(mesh_nb):
        # vertices
        accessors.append(
            {
                "bufferView": 0,
                "byteOffset": sum(size_vce[0:i]),
                "componentType": 5126,
                "count": n_vertices[i],
                "min": [bb[i][0][0], bb[i][0][1], bb[i][0][2]],
                "max": [bb[i][1][0], bb[i][1][1], bb[i][1][2]],
                "type": "VEC3",
            }
        )
        # normals
        accessors.append(
            {
                "bufferView": 1,
                "byteOffset": sum(size_vce[0:i]),
                "componentType": 5126,
                "count": n_vertices[i],
                "max": [1, 1, 1],
                "min": [-1, -1, -1],
                "type": "VEC3",
            }
        )
        if textured:
            accessors.append(
                {
                    "bufferView": 2,
                    "byteOffset": int(round(2 / 3 * sum(size_vce[0:i]))),
                    "componentType": 5126,
                    "count": sum(n_vertices),
                    "max": [1, 1],
                    "min": [0, 0],
                    "type": "VEC2",
                }
            )
    if batched:
        accessors.append(
            {
                "bufferView": 3 if textured else 2,
                "byteOffset": 0,
                "componentType": 5126,
                "count": n_vertices[0],
                "max": [batch_length],
                "min": [0],
                "type": "SCALAR",
            }
        )

    # Meshes
    meshes = []
    n_attributes = 3 if textured else 2
    for i in range(mesh_nb):
        meshes.append(
            {
                "primitives": [
                    {
                        "attributes": {
                            "POSITION": n_attributes * i,
                            "NORMAL": n_attributes * i + 1,
                        },
                        "material": 0,
                        "mode": 4,
                    }
                ]
            }
        )
        if textured:
            meshes[i]["primitives"][0]["attributes"]["TEXCOORD_0"] = (
                n_attributes * i + 2
            )
    if batched:
        meshes[0]["primitives"][0]["attributes"]["_BATCHID"] = n_attributes

    # Nodes
    nodes = []
    for i in range(mesh_nb):
        nodes.append({"matrix": [float(e) for e in transform], "mesh": i})

    # Materials
    materials = [
        {
            "pbrMetallicRoughness": {"metallicFactor": 0},
            "name": "Material",
        }
    ]

    # Final glTF
    header = {
        "asset": {"generator": "py3dtiles", "version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": list(range(len(nodes)))}],
        "nodes": nodes,
        "meshes": meshes,
        "materials": materials,
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": buffers,
    }

    # Texture data
    if textured:
        header["textures"] = [{"sampler": 0, "source": 0}]
        header["images"] = [{"uri": texture_uri}]
        header["samplers"] = [
            {"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497}
        ]
        header["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"] = {
            "index": 0
        }

    return header
