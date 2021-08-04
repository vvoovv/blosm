from math import sin, pi


class BldgPolygonFeature:
    unclassified = 0
    straightAngle = 1
    curved = 2
    triangle_convex = 3
    triangle_concave = 4
    quadrangle_convex = 5
    quadrangle_concave = 6
    complex_convex = 7
    complex_concave = 8
    

#
# values for <BldgVector.straightAngle> and <StraightAngle.type>
#
class StraightAngleType:
    # A general case of the straight angle
    other = 1
    # A more specific case of the straight angle: both edges attached to a node in question
    # do not have a shared building
    noSharedBldg = 2
    # A more specific case of the straight angle: both edges attached to a node in question
    # do have a shared building and only one
    sharedBldgBothEdges = 3
    # A more specific case of the straight angle: the straight angle is formed after
    # skipping a small feature (e.g. a quadrangular one)
    smallFeatureSkipped = 4


sin_lo = abs(sin(pi/180.*4.5))
sin_me = abs(sin(pi/180.*30))
sin_hi = abs(sin(pi/180.*80))
curvedLengthFactor = 2.
longEdgeFactor = 0.3
midEdgeFactor = 0.1


