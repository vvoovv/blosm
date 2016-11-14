from renderer import Renderer
from material import colors


class Manager:
    
    # <hexdigits> are used to check validity of colors in the hex form
    hexdigits = set("0123456789abcdef")
    
    def __init__(self, osm):
        self.osm = osm
        # don't accept broken multipolygons
        self.acceptBroken = False
    
    def setRenderer(self, renderer):
        self.renderer = renderer
        
    def process(self):
        pass

    def parseRelation(self, element, elementId):
        # render <element> in <BaseManager.render(..)>
        element.r = True
    
    @staticmethod
    def getColor(color):
        """
        Returns Python tuple of 3 float values between 0. and 1. from a string,
        which can be either a hex or a CSS color
        """
        return colors[color] if color in colors else Manager.getColorFromHex(color)
    
    @staticmethod
    def normalizeColor(color):
        """
        Check the validity of the Python string <color> for a color and
        returns the string in the lower case if it's valid or None otherwise.
        If the string is hex color, the resulting string is composed of exactly 6 character
        without leading character like <#>.
        """
        if color is None:
            return None
        color = color.lower()
        if not color in colors:
            numCharacters = len(color)
            if numCharacters == 7:
                color = color[1:]
            elif numCharacters in (3,4):
                # <color> has the form like <fff> or <#fff>
                color = "".join( 2*letter for letter in (color[-3:] if numCharacters==4 else color) )
            elif numCharacters != 6:
                # invalid
                return None
            # check that all characters in <color> are valid for a hex string
            if not all(c in Manager.hexdigits for c in color):
                # invalid
                return None
        return color
    
    @staticmethod
    def getColorFromHex(color):
        return tuple( map(lambda component: component/255, bytes.fromhex(color)) )


class Linestring(Manager):
    
    def parseWay(self, element, elementId):
        if element.tags.get("area")=="yes":
            if element.closed:
                element.t = Renderer.polygon
                # render it in <BaseManager.render(..)>
                element.r = True
            else:
                element.valid = False
        else:
            element.t = Renderer.linestring
            # render it in <BaseManager.render(..)>
            element.r = True


class Polygon(Manager):
    
    def parseWay(self, element, elementId):
        if element.closed:
            element.t = Renderer.polygon
            # render it in <BaseManager.render(..)>
            element.r = True
        else:
            element.valid = False


class PolygonAcceptBroken(Polygon):
    """
    Marks an OSM relation of the type 'multipolygon' that it accepts broken multipolygons
    """
    
    def __init__(self, osm):
        super().__init__(osm)
        # accept broken multipolygons
        self.acceptBroken = True


class BaseManager(Manager):
    
    def __init__(self, osm):
        super().__init__(osm)
    
    def render(self):
        osm = self.osm
        
        for rel in osm.relations:
            rel = osm.relations[rel]
            if rel.valid and rel.r:
                renderer = rel.rr if rel.rr else self.renderer
                renderer.preRender(rel)
                if rel.t is Renderer.polygon:
                    renderer.renderPolygon(rel, osm)
                elif rel.t is Renderer.multipolygon:
                    renderer.renderMultiPolygon(rel, osm)
                elif rel.t is Renderer.linestring:
                    renderer.renderLineString(rel, osm)
                else:
                    renderer.renderMultiLineString(rel, osm)
                renderer.postRender(rel)
        
        for way in osm.ways:
            way = osm.ways[way]
            if way.valid and way.r:
                renderer = way.rr if way.rr else self.renderer
                renderer.preRender(way)
                if way.t is Renderer.polygon:
                    renderer.renderPolygon(way, osm)
                else:
                    renderer.renderLineString(way, osm)
                renderer.postRender(way)