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

from random import normalvariate, randrange
from builtins import property


class RandomNormal:
    
    def __init__(self, mean, sigmaRatio=0.05, numValues=100):
        self.numValues = numValues
        sigma = abs(sigmaRatio*mean)
        self.values = tuple(normalvariate(mean, sigma) for _ in range(numValues))
        # the current index
        self.index = -1
    
    @property
    def value(self):
        self.index += 1
        if self.index == self.numValues:
            self.index = 0

        return self.values[self.index]


class RandomWeighted:
    
    # the number of random indices for <self.distrList>
    numIndices = 1000

    def __init__(self, distribution):
        """
        Args:
            distribution (tuple): A tuple of tuples (value, its relative weight between integer 1 and 100)
        """
        self.singleValue = None
        distrList = []
        
        if len(distribution) == 1:
            self.singleValue = distribution[0][0]
        else: 
            for n,w in distribution:
                distrList.extend(n for _ in range(w))
            self.distrList = distrList
            lenDistrList = len(distrList)
            self.indices = tuple(randrange(lenDistrList) for _ in range(self.numIndices))
            # the current index in <self.indices> which in turn points to <self.distrList>
            self.index = -1
    
    @property
    def value(self):
        if self.singleValue is None:
            self.index += 1
            if self.index == self.numIndices:
                self.index = 0
            return self.distrList[ self.indices[self.index] ]
        else:
            return self.singleValue