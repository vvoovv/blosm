from math import sin, pi


class BldgPolygonFeature:
    unclassified = 0
    straightAngle = 1
    curved = 2
    triangle = 3
    quadrangle = 4
    complex = 5


sin_lo = abs(sin(pi/180.*4.5))
sin_me = abs(sin(pi/180.*30))
sin_hi = abs(sin(pi/180.*80))
curvedLengthFactor = 2.
longEdgeFactor = 0.3
midEdgeFactor = 0.08