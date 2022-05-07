from math import ceil
import bpy
import parse
from renderer import Renderer2d
from renderer.layer import MeshLayer
from util.osm import parseNumber
from util.random import RandomWeighted


class GnBldg2dLayer(MeshLayer):
    
    def __init__(self, layerId, app):
        super().__init__(layerId, app)
        self.attributeValues = []
    
    def finalizeBlenderObject(self, obj):
        """
        Apply a Geometry Nodes setup
        """
        
        # create an attribute for the number of building levels
        obj.data.attributes.new("building:levels", 'INT', 'FACE')
        
        attributeData = obj.data.attributes["building:levels"].data
        for index, value in enumerate(self.attributeValues):
            attributeData[index].value = value
        
        m = obj.modifiers.new("", "NODES")
        m.node_group = bpy.data.node_groups[self.app.gnSetup2d]
        m["Input_2_use_attribute"] = 1
        m["Input_2_attribute_name"] ="building:levels"


class GnBldg2dManager:
    
    def __init__(self, app):
        self.layerClass = GnBldg2dLayer
        self.renderer = GnBldg2dRenderer(app)
        self.acceptBroken = False
    
    def parseWay(self, element, elementId):
        if element.closed:
            element.t = parse.polygon
            # render it in <BaseManager.render(..)>
            element.r = True
            element.rr = self.renderer
        else:
            element.valid = False

    def parseRelation(self, element, elementId):
        # skip the relation for now
        if element.valid:
            element.makePolygon()
            # render it in <BaseManager.render(..)>
            element.r = True
            element.rr = self.renderer

    def createLayer(self, layerId, app, **kwargs):
        return app.createLayer(layerId, self.layerClass)


class GnBldg2dRenderer(Renderer2d):
    
    def __init__(self, app, **kwargs):
        super().__init__(app, **kwargs)
        self.applyMaterial = False
        
        # initializing the stuff dealing with the default number of levels
        defaultLevels = bpy.context.scene.blosm.defaultLevels
        if not defaultLevels:
            from gui import addDefaultLevels
            addDefaultLevels()
        self.randomLevels = RandomWeighted(tuple((e.levels, e.weight) for e in defaultLevels))
    
    def renderPolygon(self, element, data):
        super().renderPolygon(element, data)
        element.l.attributeValues.append( self.getNumLevels(element) )
    
    def getNumLevels(self, element):
        """
        Returns the number of levels from the ground as defined by the OSM tag <building:levels>
        """
        n = element.tags.get("building:levels")
        if n:
            n = parseNumber(n)
            if n:
                n = int( ceil(n) )
                if n < 1:
                    n = None
        if not n:
            h = element.tags.get("height")
            if h:
                h = parseNumber(h)
                if h:
                    # An estimate for the number of levels. It takes into account
                    # the ground level factor (1.5) and the level height (3.)
                    n = int( ceil(abs(h/3. - 0.5)) )
            if not n:
                n = self.randomLevels.value
        return n