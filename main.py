#!/usr/bin/env python3

import os, json
import bpy
from transverse_mercator import TransverseMercator
from osm_parser import OsmParser
from srtm import Srtm

def build_neighborhood(osmFile, kwargs):
	osm = OsmParser(osmFile)
	lat = (osm.minLat + osm.maxLat)/2
	lon = (osm.minLon + osm.maxLon)/2
	#load_terrain(osm, os.path.join( os.path.dirname(basepath), "srtm" ))
	#lat = 54.1950
	#lon = 37.6204
	projection = TransverseMercator(lat=lat, lon=lon)
	osm.parse(
		projection=projection,
		#thickness=10,
		#objects=json.load(open(os.path.join(basepath, "objects.json"))),
		basepath=os.path.dirname(os.path.realpath(__file__))
	)
	# saving geo reference information for the scene
	bpy.longitude = lon
	bpy.latitude = lat

def load_terrain(extent, basepath):
	terrain = Srtm(extent, basepath)
	#terrain.build()
