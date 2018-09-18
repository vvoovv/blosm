"""
This file is part of blender-osm (OpenStreetMap importer for Blender).
Copyright (C) 2014-2018 Vladimir Elistratov
prokitektura+support@gmail.com

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

from random import normalvariate
from builtins import property


class RandomNormal:
    
    sigmaRatio = 0.05
    
    def __init__(self, mean, numValues=100):
        self.numValues = numValues
        sigma = abs(self.sigmaRatio*mean)
        self.values = tuple(normalvariate(mean, sigma) for i in range(numValues))
        # the current index
        self.index = -1
        
    @property
    def value(self):
        self.index += 1
        if self.index == self.numValues:
            self.index = 0
        return self.values[self.index]