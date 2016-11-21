from util.osm import parseNumber

class Building:
    """
    A wrapper for a OSM building
    """
    def __init__(self, element):
        self.element = element
        self.parts = []
    
    def addPart(self, part):
        self.parts.append(part)

    def markUsedNodes(self, buildingIndex, osm):
        """
        For each OSM node of <self.element> (OSM way or OSM relation) add the related
        <buildingIndex> (i.e. the index of <self> in Python list <buildings> of an instance
        of <BuildingManager>) to Python set <b> of the node 
        """
        for nodeId in self.element.nodeIds(osm):
            osm.nodes[nodeId].b.add(buildingIndex)
    
    def getHeight(self, element, op):
        tags = element.tags
        if "height" in tags:
            h = parseNumber(tags["height"], op.defaultBuildingHeight)
            if "roof:height" in tags:
                h -= parseNumber(tags["roof:height"], 0.)
        elif "building:levels" in tags:
            numLevels = parseNumber(tags["building:levels"])
            h = op.defaultBuildingHeight if numLevels is None else numLevels * op.levelHeight
        else:
            h = op.defaultBuildingHeight
        return h
    
    def getMinHeight(self, element, op):
        tags = element.tags
        if "min_height" in tags:
            z0 = parseNumber(tags["min_height"], 0.)
        elif "building:min_level" in tags:
            numLevels = parseNumber(tags["building:min_level"])
            z0 = 0. if numLevels is None else numLevels * op.levelHeight
        else:
            z0 = 0.
        return z0