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