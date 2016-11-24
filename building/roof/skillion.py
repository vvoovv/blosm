import math
from util.polygon import Polygon
from util.osm import parseNumber
from renderer import Renderer
from . import Roof


class RoofSkillion:
    
    defaultHeight = 2.
    
    def init(self, element, osm):
        self.polygon = Polygon(
            element.getData(osm) if element.t is Renderer.polygon else element.getOuterData(osm)
        )
        self.processDirection(element)
    
    def getHeight(self, element):
        tags = element.tags
        h = parseNumber(tags["roof:height"], self.defaultHeight)\
            if "roof:height" in tags\
            else self.defaultHeight

    def processDirection(self, element):
        # <d> stands for direction
        d = element.tags.get("roof:direction")
        # getting a direction vector with the unit length
        if d is None:
            d = self.getDefaultDirection()
        elif d in Roof.directions:
            d = Roof.directions[d]
        else:
            # trying to get a direction angle in degrees
            d = parseNumber(d)
            if d is None:
                d = self.getDefaultDirection()
            else:
                d = math.radians(d)
                d = (math.sin(d), math.cos(d), 0.)
    
    def getDefaultDirection(self):
        pass