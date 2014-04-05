import os, math
import bpy, bmesh
import osm_utils

def buildings(way, parser, kwargs):
	tags = way["tags"]
	objects = kwargs["objects"] if "objects" in kwargs else None
	if "building" in tags:
		thickness = kwargs["thickness"] if ("thickness" in kwargs) else 0
		osmId = way["id"]
		# compose object name
		name = osmId
		if "addr:housenumber" in tags and "addr:street" in tags:
			name = tags["addr:street"] + ", " + tags["addr:housenumber"]
		elif "name" in tags:
			name = tags["name"]

		wayNodes = way["nodes"]
		bm = bmesh.new()
		for node in range(len(wayNodes)-1): # we need to skip the last node which is the same as the first ones
			node = parser.nodes[wayNodes[node]]
			v = kwargs["projection"].fromGeographic(node["lat"], node["lon"])
			bm.verts.new((v[0], v[1], 0))

		faces = [bm.faces.new(bm.verts)]

		# extrude
		if thickness>0:
			geom = bmesh.ops.extrude_face_region(bm, geom=faces)
			verts_extruded = [v for v in geom["geom"] if isinstance(v, bmesh.types.BMVert)]
			bmesh.ops.translate(bm, verts=verts_extruded, vec=(0, 0, thickness))

		bm.normal_update()

		me = bpy.data.meshes.new(osmId)
		bm.to_mesh(me)

		obj = bpy.data.objects.new(name, me)
		bpy.context.scene.objects.link(obj)
		bpy.context.scene.update()

		# final adjustments
		obj.select = True
		# assign OSM tags to the blender object
		osm_utils.assignTags(obj, tags)

def highways(way, parser, kwargs):
	pass
