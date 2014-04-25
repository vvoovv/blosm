import os, math
import bpy, bmesh
import osm_utils

class buildings:
	@staticmethod
	def condition(tags, way):
		return "building" in tags
	
	@staticmethod
	def handler(way, parser, kwargs):
		wayNodes = way["nodes"]
		numNodes = len(wayNodes)-1 # we need to skip the last node which is the same as the first ones
		# a polygon must have at least 3 vertices
		if numNodes<3: return
		
		tags = way["tags"]
		thickness = kwargs["thickness"] if ("thickness" in kwargs) else 0
		osmId = way["id"]
		# compose object name
		name = osmId
		if "addr:housenumber" in tags and "addr:street" in tags:
			name = tags["addr:street"] + ", " + tags["addr:housenumber"]
		elif "name" in tags:
			name = tags["name"]
		
		bm = bmesh.new()
		for node in range(numNodes):
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
