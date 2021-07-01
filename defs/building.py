from math import sin, pi


class BldgPolygonFeature:
    Unclassified = 0
    StraightAngle = 1
    Curvy = 1
    Rectangle = 2
    Triangle = 3


sin_lo = abs(sin(pi/180.*5.))
sin_me = abs(sin(pi/180.*30))
sin_hi = abs(sin(pi/180.*80))
curvyLengthFactor = 2.
lengthThresh = 5.