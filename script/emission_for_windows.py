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

import random
import numpy
import bpy

numWindows = 200
numProbabilities = 100
imageName = "emission_for_windows"

imageData = numpy.zeros(4*numWindows*numProbabilities)

for y in range(numProbabilities):
    for x in range(numWindows):
        offset = 4*(y*numWindows+x)
        randomNumber = random.randrange(numProbabilities-1)
        value = 1. if randomNumber < y else 0.
        imageData[offset:offset+4] = value, value, value, 1.


images = bpy.data.images

if imageName in images:
    images.remove(images[imageName], True)


image = bpy.data.images.new(
    imageName,
    width = numWindows,
    height = numProbabilities,
    alpha=False,
    float_buffer=False
)
image.pixels = imageData
image.use_fake_user = True
# pack the image into .blend file
image.pack(as_png=True)