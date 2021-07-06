from math import sin, pi


class BldgPolygonFeature:
    unclassified = 0
    straightAngle = 1
    curved = 1
    rectangle = 2
    triangle = 3


sin_lo = abs(sin(pi/180.*5.))
sin_me = abs(sin(pi/180.*30))
sin_hi = abs(sin(pi/180.*80))
curvyLengthFactor = 2.
lengthThresh = 5.