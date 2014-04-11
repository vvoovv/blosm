# This is the release version of the plugin file io_import_scene_osm_dev.py
# If you would like to make edits, make them in the file io_import_scene_osm_dev.py and the other related modules
# To create the release version of io_import_scene_osm_dev.py, executed:
# python plugin_builder.py io_import_scene_osm_dev.py:
bl_info = {
	"name": "Import OpenStreetMap (.osm)",
	"author": "Vladimir Elistratov <vladimir.elistratov@gmail.com>",
	"version": (1, 0, 0),
	"blender": (2, 6, 9),
	"location": "File > Import > OpenStreetMap (.osm)",
	"description": "Import a file in the OpenStreetMap format (.osm)",
	"warning": "",
	"wiki_url": "https://github.com/vvoovv/blender-geo/wiki/Import-OpenStreetMap-(.osm)",
	"tracker_url": "https://github.com/vvoovv/blender-geo/issues",
	"support": "COMMUNITY",
	"category": "Import-Export",
}

import bpy
# ImportHelper is a helper class, defines filename and invoke() function which calls the file selector
from bpy_extras.io_utils import ImportHelper

import sys, os
import math

# see conversion formulas at
# http://en.wikipedia.org/wiki/Transverse_Mercator_projection
# and
# http://mathworld.wolfram.com/MercatorProjection.html
class TransverseMercator:
	radius = 6378137
	lat = 0 # in degrees
	lon = 0 # in degrees
	k = 1 # scale factor

	def __init__(self, **kwargs):
		for attr in kwargs:
			setattr(self, attr, kwargs[attr])
		self.latInRadians = math.radians(self.lat)

	def fromGeographic(self, lat, lon):
		lat = math.radians(lat)
		lon = math.radians(lon-self.lon)
		B = math.sin(lon) * math.cos(lat)
		x = 0.5 * self.k * self.radius * math.log((1+B)/(1-B))
		y = self.k * self.radius * ( math.atan(math.tan(lat)/math.cos(lon)) - self.latInRadians )
		return [x,y]

	def toGeographic(self, x, y):
		x = x/(self.k * self.radius)
		y = y/(self.k * self.radius)
		D = y + self.latInRadians
		lon = math.atan(math.sinh(x)/math.cos(D))
		lat = math.asin(math.sin(D)/math.cosh(x))

		lon = self.lon + math.degrees(lon)
		lat = math.degrees(lat)
		return [lat, lon]
import xml.etree.cElementTree as etree
import inspect, importlib

def prepareHandlers(kwArgs):
	nodeHandlers = []
	wayHandlers = []
	# getting a dictionary with local variables
	_locals = locals()
	for handlers in ("nodeHandlers", "wayHandlers"):
		if handlers in kwArgs:
			for handler in kwArgs[handlers]:
				if isinstance(handler, str):
					# we've got a module name
					handler = importlib.import_module(handler)
				if inspect.ismodule(handler):
					# iterate through all module functions
					for f in inspect.getmembers(handler, inspect.isfunction):
						_locals[handlers].append(f[1])
				elif inspect.isfunction(handler):
					_locals[handlers].append(handler)
		if len(_locals[handlers])==0: _locals[handlers] = None
	return (nodeHandlers if len(nodeHandlers) else None, wayHandlers if len(wayHandlers) else None)

class OsmParser:
	nodes = {}
	ways = {}
	relations = {}
	doc = None
	osm = None
	minLat = 90
	maxLat = -90
	minLon = 180
	maxLon = -180
	
	def __init__(self, filename):
		self.doc = etree.parse(filename)
		self.osm = self.doc.getroot()
		self.prepare()

	def prepare(self):
		for e in self.osm: # e stands for element
			if "action" in e.attrib and e.attrib["action"] == "delete": continue
			if e.tag == "bounds": continue
			attrs = e.attrib
			_id = attrs["id"]
			if e.tag == "node":
				tags = None
				for c in e:
					if c.tag == "tag":
						if not tags: tags = {}
						tags[c.get("k")] = c.get("v")
				lat = float(attrs["lat"])
				lon = float(attrs["lon"])
				# calculating minLat, maxLat, minLon, maxLon
				if lat<self.minLat: self.minLat = lat
				elif lat>self.maxLat: self.maxLat = lat
				if lon<self.minLon: self.minLon = lon
				elif lon>self.maxLon: self.maxLon = lon
				# creating entry
				entry = dict(
					id=_id,
					e=e,
					lat=lat,
					lon=lon
				)
				if tags: entry["tags"] = tags
				self.nodes[_id] = entry
			elif e.tag == "way":
				nodes = []
				tags = None
				for c in e:
					if c.tag == "nd":
						nodes.append(c.get("ref"))
					elif c.tag == "tag":
						if not tags: tags = {}
						tags[c.get("k")] = c.get("v")
				# ignore ways without tags
				if tags:
					self.ways[_id] = dict(
						id=_id,
						e=e,
						nodes=nodes,
						tags=tags
					)

	def parse(self, **kwargs):
		(nodeHandlers, wayHandlers) = prepareHandlers(kwargs)
		for e in self.osm: # e stands for element
			if "action" in e.attrib and e.attrib["action"] == "delete": continue
			if e.tag == "bounds": continue
			attrs = e.attrib
			_id = attrs["id"]
			if wayHandlers and e.tag == "way" and _id in self.ways:
				for handler in wayHandlers:
					handler(self.ways[_id], self, kwargs)
			elif nodeHandlers and e.tag == "node" and _id in self.nodes:
				for handler in nodeHandlers:
					handler(self.nodes[_id], self, kwargs)

import os, math
import bpy, bmesh
def assignTags(obj, tags):
	for key in tags:
		obj[key] = tags[key]

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
		assignTags(obj, tags)

def highways(way, parser, kwargs):
	pass

import bpy

def findGeoObject(context):
	"""
	Find a Blender object with "latitude" and "longitude" as custom properties
	"""
	for o in context.scene.objects:
		if "latitude" in o and "longitude" in o:
			return o
	return None

class ImportOsm(bpy.types.Operator, ImportHelper):
	"""Import a file in the OpenStreetMap format (.osm)"""
	bl_idname = "import_scene.osm"  # important since its how bpy.ops.import_scene.osm is constructed
	bl_label = "Import OpenStreetMap"
	bl_options = {"UNDO"}

	# ImportHelper mixin class uses this
	filename_ext = ".osm"

	filter_glob = bpy.props.StringProperty(
		default="*.osm",
		options={"HIDDEN"},
	)

	thickness = bpy.props.FloatProperty(
		name="Thickness",
		description="Set thickness to make OSM objects extruded",
		default=0,
	)

	def execute(self, context):
		# setting active object if there is no active object
		if not context.scene.objects.active:
			context.scene.objects.active = context.scene.objects[0]
		bpy.ops.object.mode_set(mode="OBJECT")
		
		bpy.ops.object.select_all(action="DESELECT")
		
		# try to find a Blender object with "latitude" and "longitude" as custom properties
		geoObject = findGeoObject(context)
		
		# create an empty object to parent all imported OSM objects
		bpy.ops.object.empty_add(type="PLAIN_AXES", location=(0, 0, 0))
		parentObject = context.active_object
		self.parentObject = parentObject
		parentObject.name = os.path.basename(self.filepath)
		#parentObject.hide = True
		#parentObject.hide_select = True
		parentObject.hide_render = True
		
		self.read_osm_file(geoObject, parentObject)
		
		# perform parenting
		context.scene.objects.active = parentObject
		bpy.ops.object.parent_set()
		bpy.ops.object.select_all(action="DESELECT")
		return {"FINISHED"}

	def read_osm_file(self, geoObject, parentObject):
		osm = OsmParser(self.filepath)
		if geoObject:
			lat = geoObject["latitude"]
			lon = geoObject["longitude"]
		else:
			lat = (osm.minLat + osm.maxLat)/2
			lon = (osm.minLon + osm.maxLon)/2
			parentObject["latitude"] = lat
			parentObject["longitude"] = lon
		projection = TransverseMercator(lat=lat, lon=lon)
		osm.parse(
			projection=projection,
			thickness=self.thickness,
			# possible values for wayHandlers and nodeHandlers list elements:
			#	1) a string name for the module containing functions (all functions from the modules will be used as handlers)
			#	2) a python variable representing the module containing functions (all functions from the modules will be used as handlers)
			#	3) a python variable representing the function
			wayHandlers = [buildings] #[handlers.buildings] #[handlers] #["handlers"]
		)


# Only needed if you want to add into a dynamic menu
def menu_func_import(self, context):
	self.layout.operator(ImportOsm.bl_idname, text="OpenStreetMap (.osm)")

def register():
	bpy.utils.register_class(ImportOsm)
	bpy.types.INFO_MT_file_import.append(menu_func_import)

def unregister():
	bpy.utils.unregister_class(ImportOsm)
	bpy.types.INFO_MT_file_import.remove(menu_func_import)
