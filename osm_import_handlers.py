import os, math
import bpy, bmesh
import utils, osm_utils

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
		
		if not kwargs["bm"]: # not a single mesh
			tags = way["tags"]
			thickness = kwargs["thickness"] if ("thickness" in kwargs) else 0
			osmId = way["id"]
			# compose object name
			name = osmId
			if "addr:housenumber" in tags and "addr:street" in tags:
				name = tags["addr:street"] + ", " + tags["addr:housenumber"]
			elif "name" in tags:
				name = tags["name"]
		
		bm = kwargs["bm"] if kwargs["bm"] else bmesh.new()
		verts = []
		for node in range(numNodes):
			node = parser.nodes[wayNodes[node]]
			v = kwargs["projection"].fromGeographic(node["lat"], node["lon"])
			verts.append( bm.verts.new((v[0], v[1], 0)) )
		
		bm.faces.new(verts)
		
		if not kwargs["bm"]:
			thickness = kwargs["thickness"] if ("thickness" in kwargs) else 0
			# extrude
			if thickness>0:
				utils.extrudeMesh(bm, thickness)
			
			bm.normal_update()
			
			mesh = bpy.data.meshes.new(osmId)
			bm.to_mesh(mesh)
			
			obj = bpy.data.objects.new(name, mesh)
			bpy.context.scene.objects.link(obj)
			bpy.context.scene.update()
			
			# final adjustments
			obj.select = True
			# assign OSM tags to the blender object
			osm_utils.assignTags(obj, tags)


class highways:
	@staticmethod
	def condition(tags, way):
		return "highway" in tags
	
	@staticmethod
	def handler(way, parser, kwargs):
		wayNodes = way["nodes"]
		numNodes = len(wayNodes) # we need to skip the last node which is the same as the first ones
		# a way must have at least 2 vertices
		if numNodes<2: return
		
		if not kwargs["bm"]: # not a single mesh
			tags = way["tags"]
			osmId = way["id"]
			# compose object name
			name = tags["name"] if "name" in tags else osmId
		
		bm = kwargs["bm"] if kwargs["bm"] else bmesh.new()
		verts = []
		prevVertex = None
		for node in range(numNodes):
			node = parser.nodes[wayNodes[node]]
			v = kwargs["projection"].fromGeographic(node["lat"], node["lon"])
			v = bm.verts.new((v[0], v[1], 0))
			if prevVertex:
				bm.edges.new([prevVertex, v])
			prevVertex = v
		
		if not kwargs["bm"]:
			mesh = bpy.data.meshes.new(osmId)
			bm.to_mesh(mesh)
			
			obj = bpy.data.objects.new(name, mesh)
			bpy.context.scene.objects.link(obj)
			bpy.context.scene.update()
			
			# final adjustments
			obj.select = True
			# assign OSM tags to the blender object
			osm_utils.assignTags(obj, tags)