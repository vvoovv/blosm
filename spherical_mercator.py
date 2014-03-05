#!/usr/bin/env python3

import math

class SphericalMercator:
	radius = 6378137

	def fromGeographic(self, coords):
		x = self.radius * math.radians(coords[1])
		y = self.radius * math.log(math.tan(math.pi/4 + math.radians(coords[0])/2))
		return [x,y]

	def toGeographic(self, x, y):
		lon = math.degrees(x/self.radius)
		lat = 2*math.degrees(math.atan(math.exp(y/self.radius))) - 90
		return [lat, lon]