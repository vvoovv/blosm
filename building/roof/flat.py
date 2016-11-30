from util.osm import parseNumber


class RoofFlat:
    
    defaultHeight = 0.5
    
    def init(self, element, osm):
        pass
    
    def getHeight(self):
        tags = self.element.tags
        return parseNumber(tags["roof:height"], self.defaultHeight)\
            if "roof:height" in tags\
            else self.defaultHeight
        