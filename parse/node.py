

class Node:
    """
    A class to represent an OSM node
    
    Some attributes:
        tags (dict): OSM tags
        b (set): Here we store building indices (i.e. the indices of instances of
        the wrapper class <building.manager.Building> in Python list <buildings> of an instance
        of <building.manager.BuildingManager>)
    """
    __slots__ = ("tags", "lat", "lon", "coords", "b")
    
    def __init__(self, lat, lon, tags):
        self.tags = tags
        self.lat = lat
        self.lon = lon
        # projected coordinates
        self.coords = None
    
    def getData(self, osm):
        """
        Get projected coordinates
        """
        if not self.coords:
            # preserve coordinates in the local system of reference for a future use
            self.coords = osm.projection.fromGeographic(self.lat, self.lon)
        return self.coords