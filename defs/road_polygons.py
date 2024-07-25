# Parameter definitions used in and around road_polygons.py

# Way segements with these tags are excluded from networkGraph and sectionNetwork
ExcludedWayTags = ['steps']

# Large object polygons may have no vertices in or near graph cycles. To make them
# detectable, a grid of detection vertices is added to the KD-Tree.
# <MinDetectionSize> is the minimum width or height of the bounding box of
# a polygon to consider it to be large
# <DetectionGridWidth> is the distance between the grid points
MinDetectionSize = 100
DetectionGridWidth = 30

# Single common vertices between polygons have to be separated to avoid the
# construction of degenerated polygons. These vertices of holes and polyline
# object are moved by <SharedVertDist> away from the vertices of the way-segments.
SharedVertDist = 0.001  # 1 mm

# Large graph-cycles are subdivided, if their bonding box exeeds the size of
# <MaxCycleSize>. The subdivision is made along the smaller bounding box dimension,
# as long as all tiles are smaller than <MaxCycleSize>.
MaxCycleSize = 10000000

