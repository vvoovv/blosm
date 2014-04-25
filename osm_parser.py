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
					for f in inspect.getmembers(handler, inspect.isclass):
						_locals[handlers].append(f[1])
				elif inspect.isclass(handler):
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
	
	def __init__(self, filename, **kwargs):
		self.kwargs = kwargs
		(self.nodeHandlers, self.wayHandlers) = prepareHandlers(kwargs)
		
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

	def iterate(self, wayFunction, nodeFunction):
		nodeHandlers = self.nodeHandlers
		wayHandlers = self.wayHandlers
		
		for e in self.osm: # e stands for element
			if "action" in e.attrib and e.attrib["action"] == "delete": continue
			if e.tag == "bounds": continue
			attrs = e.attrib
			_id = attrs["id"]
			if wayHandlers and e.tag == "way" and _id in self.ways:
				way = self.ways[_id]
				if "tags" in way:
					for handler in wayHandlers:
						if handler.condition(way["tags"], way):
							wayFunction(handler, way)
							continue
			elif nodeHandlers and e.tag == "node" and _id in self.nodes:
				node = self.nodes[_id]
				if "tags" in node:
					for handler in nodeHandlers:
						if handler.condition(node["tags"], node):
							nodeFunction(handler, node)
							continue

	def parse(self, **kwargs):
		def wayFunction(handler, way):
			handler.handler(way, self, kwargs)
		def nodeFunction(handler, node):
			handler.handler(node, self, kwargs)
		self.iterate(wayFunction, nodeFunction)