from util.osm import parseNumber


class RoofFlat:
    
    defaultHeight = 0.
    
    def init(self, element, osm):
        pass
    
    def getHeight(self, element):
        tags = element.tags
        return parseNumber(tags["roof:height"], self.defaultHeight)\
            if "roof:height" in tags\
            else self.defaultHeight
        