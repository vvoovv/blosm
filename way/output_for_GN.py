class WaySection_gn():
    def __init__(self):
        self.centerline = None      # The trimmed centerline in forward direction (first vertex is start.
                                    # and last is end). A Python list of vertices of type mathutils.Vector.
        self.category = None        # The category of the way-section.
        self.startWidths = None     # The left and right width at its start, seen relative to the direction
                                    # of the centerline. A tuple of floats.
        self.endWidths = None       # The left and right width at its end, seen relative to the direction
                                    # of the centerline. A tuple of floats.
        self.tags = None            # The OSM tags of the way-section.

class IntersectionPoly_gn():
    def __init__(self):
        self.polygon = None         # The vertices of the polygon in counter-clockwise order. A Python
                                    # list of vertices of type mathutils.Vector.
        self.connectors = dict()    # The connectors (class Connector_gn) of this polygon to the way-sections.
                                    # A dictionary of tuples, where the key is the ID of the corresponding
                                    # way-section key in the dictionary of WaySection_gn. The tuple has two
                                    # elements, <indx> and <end>. <indx> is the first index in the intersection
                                    # polygon for this connector, the second index is <indx>+1. <end> is
                                    # the type 'S' for the start or 'E' for the end of the WaySection_gn
                                    # connected here.
