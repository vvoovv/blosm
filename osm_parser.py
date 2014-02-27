#!/usr/bin/env python3

import xml.etree.cElementTree as etree
import inspect

import handlers

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
		for e in self.osm: # e stands for element
			if "action" in e.attrib and e.attrib["action"] == "delete": continue
			if e.tag == "bounds": continue
			attrs = e.attrib
			_id = attrs["id"]
			if e.tag == "node":
				pass
			elif e.tag == "way":
				if _id in self.ways:
					self.processWay(self.ways[_id], kwargs)

	def processWay(self, way, kwargs):
		for handler in inspect.getmembers(handlers, inspect.isfunction):
			handler[1](way, self, kwargs)
