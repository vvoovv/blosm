from renderer import Renderer


class Feature:
    
    @staticmethod
    def updateBounds(bounds, coords):
        for coord in coords:
            lat = coord[1]
            lon = coord[0]
            if lat<bounds.minLat:
                bounds.minLat = lat
            elif lat>bounds.maxLat:
                bounds.maxLat = lat
            if lon<bounds.minLon:
                bounds.minLon = lon
            elif lon>bounds.maxLon:
                bounds.maxLon = lon


class Polygon:
    """
    A closed polygonal chain without self-intersections and holes
    """
    
    def __init__(self, coords, tags):
        self.valid = True
        self.coords = coords
        # preserved projected coordinates
        self._coords = None
        self.tags = tags
        self.r = False
        self.rr = None
        self.t = Renderer.polygon

    def updateBounds(self, bounds):
        Feature.updateBounds(bounds, self.coords)

    def getData(self, geojson):
        """
        Get projected data for the polygon coordinates
        
        Returns a Python generator
        """
        if not self._coords:
            coords = self.coords
            numCoords = len(coords)
            self._coords = tuple(geojson.projection.fromGeographic(coords[index][1], coords[index][0]) for index in range(numCoords-1))
        return (coord for coord in self._coords)


class Multipolygon:
    """
    A polygon with one or more holes
    """
    
    def __init__(self, coords, tags):
        self.valid = True
        self.coords = coords
        self.tags = tags
        self.r = False
        self.rr = None
        self.t = Renderer.multipolygon

    def updateBounds(self, bounds):
        # only the outer ring matters, so we <self.coords[0]>
        Feature.updateBounds(bounds, self.coords[0])


class Node:
    
    def __init__(self, coords, tags):
        self.valid = True
        self.coords = coords
        self.tags = tags