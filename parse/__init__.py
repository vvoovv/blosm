import xml.etree.cElementTree as etree
import inspect, importlib

from .node import Node
from .way import Way


class Osm:
    """
    Representation of data in an OSM file
    """
    
    # OSM data type
    node = (1,)
    way = (1,)
    relation = (1,)
    # <types> is set in the constructor
    types = None
    
    # <relationTypes> is set in the constructor
    relationTypes = None
    
    # roles of members of an OSM relation
    outer = (1,)
    inner = (1,)
    outline = (1,)
    part = (1,)
    # <roles> is set in the constructor
    roles = None
    
    def __init__(self, op):
        self.op = op
        
        if not Osm.types:
            # initialize dictionaries
            Osm.types = {'node':Osm.node, 'way':Osm.way, 'relation':Osm.relation}
            Osm.relationTypes = {'multipolygon': Multipolygon, 'building': Building}
            Osm.roles = {'outer':Osm.outer, 'inner':Osm.inner, 'outline':Osm.outline, 'part':Osm.part}
        
        self.nodes = {}
        self.ways = {}
        self.relations = {}
        
        self.conditions = []
        
        self.doc = etree.parse(op.filepath)
        self.osm = self.doc.getroot()
    
    def addCondition(self, condition, layerId=None, manager=None, renderer=None):
        # convert layerId to layerIndex
        layerIndex = self.op.layerIndices.get(layerId)
        self.conditions.append((condition, manager, renderer, layerIndex))
    
    def parse(self, **kwargs):        
        self.projection = kwargs.get("projection")
        if not self.projection:
            self.minLat = 90.
            self.maxLat = -90.
            self.minLon = 180.
            self.maxLon = -180.
        
        relations = self.relations
        
        for e in self.osm: # e stands for element
            attrs = e.attrib
            if "action" in attrs and attrs["action"] == "delete": continue
            if e.tag == "node":
                _id = attrs["id"]
                tags = None
                for c in e:
                    if c.tag == "tag":
                        if not tags:
                            tags = {}
                        tags[c.get("k")] = c.get("v")
                self.nodes[_id] = Node(float(attrs["lat"]), float(attrs["lon"]), tags)
            elif e.tag == "way":
                _id = attrs["id"]
                nodes = []
                tags = None
                for c in e:
                    if c.tag == "nd":
                        nodes.append(c.get("ref"))
                    elif c.tag == "tag":
                        if not tags:
                            tags = {}
                        tags[c.get("k")] = c.get("v")
                way = Way(nodes, tags, self)
                if way.valid:
                    # do we need to skip the OSM <way> from storing in <self.ways>
                    skip = False
                    # <ci> stands for <condition index>
                    if tags:
                        ci = self.checkConditions(tags, way)
                        if not ci is None:
                            skip = self.processCondition(ci, way, _id, Osm.way)
                            if not self.projection and way.valid:
                                self.updateBounds(way)
                    if not skip:
                        self.ways[_id] = way
            elif e.tag == "relation":
                _id = attrs["id"]
                members = []
                tags = None
                relation = None
                for c in e:
                    if c.tag == "member":
                        mType = Osm.types.get( c.get("type") )
                        mId = c.get("ref")
                        mRole = Osm.roles.get( c.get("role") )
                        if not (mType and mId):
                            continue
                        members.append((mType, mId, mRole))
                    elif c.tag == "tag":
                        if not tags:
                            tags = {}
                        k = c.get("k")
                        v = c.get("v")
                        tags[k] = v
                        if k == "type":
                            relation = Osm.relationTypes.get(v)
                # skip the relation without tags
                if relation and tags:
                    createdBefore = _id in relations
                    if createdBefore:
                        # The empty OSM relation was created before,
                        # since it's referenced by another OSM relation
                        relation = relations[_id]
                    else:
                        relation = relation(self)
                    if relation.valid:
                        # <ci> stands for <condition index>
                        ci = self.checkConditions(tags, relation)
                        if not ci is None:
                            relation.process(members, tags, self)
                            if relation.valid:
                                skip = self.processCondition(ci, relation, _id, Osm.relation)
                                if not createdBefore and not skip:
                                    relations[_id] = relation
            elif e.tag == "bounds":
                # If <projectionClass> is present in <kwargs>,
                # it means we need to set <self.projection> here,
                # adding to kwargs the center of <bounds>
                projectionClass = kwargs.get("projectionClass")
                if projectionClass:
                    lat = ( float(attrs["minlat"]) + float(attrs["maxlat"]) )/2.
                    lon = ( float(attrs["minlon"]) + float(attrs["maxlon"]) )/2.
                    self.setProjection(projectionClass, lat, lon, kwargs)
        
        if not self.projection:
            # set projection using the calculated bounds (self.minLat, self.maxLat, self.minLon, self.maxLon)
            lat = (self.minLat + self.maxLat)/2.
            lon = (self.minLon + self.maxLon)/2.
            self.setProjection(kwargs["projectionClass"], lat, lon, kwargs)
    
    def checkConditions(self, tags, element):
        for i,c in enumerate(self.conditions):
            if c[0](tags, element):
                # setting manager
                element.m = c[1]
                return i
    
    def processCondition(self, ci, element, elementId, elementType):
        # do we need to skip the OSM <element> from storing in <self.ways> or <self.relations>
        skip = False
        condition = self.conditions[ci]
        # check if we have a special manager for the element
        manager = condition[1]
        if manager:
            if elementType is Osm.way:
                skip = manager.parseWay(element, elementId)
            else:
                skip = manager.parseRelation(element, elementId)
        # alwayes set <layerIndex>
        # layerIndex = condition[3]
        element.li = condition[3]
        # check if wee need to set a special renderer
        if condition[2]:
            # renderer = condition[2]
            element.rr = condition[2]
        return skip
    
    def updateBounds(self, way):
        # <way> has been already used for bounds calculation
        if way.used:
            return
        for i in range(way.n):
            node = self.nodes[way.nodes[i]]
            lat = node.lat
            lon = node.lon
            if lat<self.minLat:
                self.minLat = lat
            elif lat>self.maxLat:
                self.maxLat = lat
            if lon<self.minLon:
                self.minLon = lon
            elif lon>self.maxLon:
                self.maxLon = lon
        # mark <way> as used for bounds calculation
        way.used = True

    def getRelation(self, _id, relationClass):
        """
        Returns a relation with <_id> if it has been already encountered in the OSM file or
        creates a relation of <relationClass> with no parameters and sets it to <selfs.relations[_id]>
        """
        relations = self.relations
        relation = relations.get(_id)
        if not relation:
            relation = relationClass()
            relations[_id] = relation
        return relation
    
    def setProjection(self, projectionClass, lat, lon, kwargs):
        self.lat = lat
        self.lon = lon
        kwargs["lat"] = lat
        kwargs["lon"] = lon
        self.projection = projectionClass(**kwargs)


from .relation.multipolygon import Multipolygon
from .relation.building import Building