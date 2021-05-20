from math import tan, pi

from .way import Category, facadeVisibilityWayCategories


class FacadeClass:
    unknown = 0
    front = 1
    side = 2
    back = 3
    shared = 4
    passage = 5
    deadend = 6

CrossedFacades = (FacadeClass.deadend, FacadeClass.passage)
FrontLikeFacades = (FacadeClass.front, FacadeClass.passage)

searchRange = (10., 100.)                               # (searchWidthMargin, searchHeight)

FrontFacadeSight = 0.6                                  # edge sight required to classify as front facade
FacetVisibilityLimit = 0.75                             # visibility limit for corner facade facet special case (manhattan_01.osm)
VisibilityAngle = 50                                    # maximum angle in ° between way-segment and facade to be accepted as visible
VisibilityAngleFactor = tan(pi*VisibilityAngle/180.)    # Factor used in angle condition: VisibilityAngleFactor*dx > dy
maxDistanceRatio = 7.5                                  # maximum allowed ratio of edge diatnce (Y1+Y2) and maximum building dimension

WayLevel = dict((category,1) for category in facadeVisibilityWayCategories)
WayLevel[Category.service] = 2
MaxWayLevel = 2