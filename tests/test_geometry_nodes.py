"""Run with Blender --background --factory-startup --python-exit-code 1 --python FILE."""

import importlib.util
from pathlib import Path
import unittest

import bpy

spec = importlib.util.spec_from_file_location(
    "gn_compat", Path(__file__).resolve().parents[1] / "util/geometry_nodes.py"
)
gn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gn)


class GeometryNodesTests(unittest.TestCase):
    def setUp(self):
        self.group = bpy.data.node_groups.new("GN compatibility test", "GeometryNodeTree")
        self.mesh = bpy.data.meshes.new("GN compatibility test")
        self.obj = bpy.data.objects.new("GN compatibility test", self.mesh)
        bpy.context.scene.collection.objects.link(self.obj)
        self.sockets = {}
        for kind in ("Geometry", "Bool", "Float", "Int", "Vector", "Color", "String", "Object", "Material", "Collection"):
            self.sockets[kind] = self.newSocket(kind, 'INPUT', 'NodeSocket' + kind)
        self.output = self.newSocket("Geometry", 'OUTPUT', 'NodeSocketGeometry')
        self.fieldOutput = self.newSocket("Height", 'OUTPUT', 'NodeSocketFloat')
        self.mod = self.obj.modifiers.new("GN compatibility test", 'NODES')
        self.mod.node_group = self.group

    def newSocket(self, name, direction, kind):
        if hasattr(self.group, "interface"):
            return self.group.interface.new_socket(name=name, in_out=direction, socket_type=kind)
        sockets = self.group.inputs if direction == 'INPUT' else self.group.outputs
        return sockets.new(kind, name)

    def tearDown(self):
        bpy.data.objects.remove(self.obj, do_unlink=True)
        bpy.data.meshes.remove(self.mesh)
        bpy.data.node_groups.remove(self.group)

    def test_values_and_instance_isolation(self):
        other = self.obj.modifiers.new("Other instance", 'NODES')
        other.node_group = self.group
        material = bpy.data.materials.new("GN test material")
        collection = bpy.data.collections.new("GN test collection")
        try:
            values = {"Bool": False, "Float": 0.0, "Int": 0, "Vector": (1.0, 2.0, 3.0),
                      "Color": (0.25, 0.5, 0.75, 1.0), "String": "", "Object": self.obj,
                      "Material": material, "Collection": collection}
            for kind, value in values.items():
                identifier = self.sockets[kind].identifier
                gn.setGnInput(self.mod, identifier, value)
                actual = gn.getGnInput(self.mod, identifier)
                self.assertEqual(tuple(actual) if isinstance(value, tuple) else actual, value)
            identifier = self.sockets["Float"].identifier
            gn.setGnInput(self.mod, identifier, 7.5)
            self.assertEqual(gn.getGnInput(other, identifier), 0.0)
            for kind in ("Object", "Material", "Collection"):
                gn.setGnInput(self.mod, self.sockets[kind].identifier, None)
                self.assertIsNone(gn.getGnInput(self.mod, self.sockets[kind].identifier))
        finally:
            bpy.data.materials.remove(material)
            bpy.data.collections.remove(collection)

    def test_attributes_and_literal_switch(self):
        identifier = self.sockets["Float"].identifier
        for name in ("building:height", ""):
            gn.useAttributeForGnInput(self.mod, identifier, name)
            self.assertEqual(gn.getGnInputAttribute(self.mod, identifier), name)
            gn.setGnInput(self.mod, identifier, 3.0)
            self.assertIsNone(gn.getGnInputAttribute(self.mod, identifier))
        identifier = self.fieldOutput.identifier
        for name in ("generated_height", ""):
            gn.setGnOutputAttribute(self.mod, identifier, name)
            self.assertEqual(gn.getGnOutputAttribute(self.mod, identifier), name)

    def test_socket_discovery_and_errors(self):
        if hasattr(self.group, "interface"):
            self.group.interface.new_panel("Panel")
        ids = [s.identifier for s in gn.iterGnInputs(self.mod)]
        self.assertEqual(ids, [s.identifier for name, s in self.sockets.items() if name != "Geometry"])
        self.sockets["Float"].name = "Renamed"
        gn.setGnInput(self.mod, self.sockets["Float"].identifier, 2.0)
        with self.assertRaises(KeyError):
            gn.setGnInput(self.mod, "missing_socket", 1.0)
        with self.assertRaises(KeyError):
            gn.setGnInput(self.mod, self.fieldOutput.identifier, 1.0)
        with self.assertRaises(TypeError):
            gn.setGnInput(self.mod, self.sockets["Geometry"].identifier, 1.0)
        with self.assertRaises(TypeError):
            gn.useAttributeForGnInput(self.mod, self.sockets["Object"].identifier, "invalid")
        plain = self.obj.modifiers.new("Plain modifier", 'SOLIDIFY')
        with self.assertRaises(TypeError):
            gn.setGnInput(plain, "missing", 1.0)
        empty = self.obj.modifiers.new("No node group", 'NODES')
        with self.assertRaises(ValueError):
            gn.setGnInput(empty, "missing", 1.0)

    def test_evaluated_geometry_updates(self):
        inputs = self.group.nodes.new('NodeGroupInput')
        output = self.group.nodes.new('NodeGroupOutput')
        cube = self.group.nodes.new('GeometryNodeMeshCube')
        self.group.links.new(inputs.outputs["Vector"], cube.inputs["Size"])
        self.group.links.new(cube.outputs["Mesh"], output.inputs["Geometry"])
        for size in ((2.0, 4.0, 6.0), (3.0, 5.0, 7.0)):
            gn.setGnInput(self.mod, self.sockets["Vector"].identifier, size)
            bpy.context.view_layer.update()
            evaluated = self.obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
            self.assertEqual(tuple(round(v, 5) for v in evaluated.dimensions), size)


if __name__ == '__main__':
    print("BLENDER", bpy.app.version_string, flush=True)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(GeometryNodesTests))
    if not result.wasSuccessful():
        raise RuntimeError("Geometry Nodes compatibility checks failed")
