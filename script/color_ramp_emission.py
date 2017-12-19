"""
A script to assign colors to the Color Ramp node responsible for colors and strengths of
building window emission.
"""
import bpy
import random

nodeGroupName = "WindowEmission"

# the number of elements in the color ramp (32 is the limit imposed by Blender)
numElements = 32

colors = (
    (0.847, 0.418, 0.087),
    (1., 0.931, 0.455),
    (0.847, 0.671, 0.325),
    (1., 0.527, 0.297)
)
# emission strengths
strengths = (0.5, 0.4, 0.42, 0.32)

numColors = len(colors)
numStrengths = len(strengths)

colorRampStep = 1./numElements

nodes = bpy.data.node_groups[nodeGroupName].nodes
nodes["Modulo"].inputs[1].default_value = numElements
nodes["Divide"].inputs[1].default_value = numElements
nodes["Add"].inputs[1].default_value = colorRampStep/2.

# elements of the color ramp
elements = nodes["ColorRamp"].color_ramp.elements

# remove all but the very first one elements
for i in range(len(elements)-1, 0, -1):
    elements.remove(elements[i])

elements[0].position = 0.

# create <numElements-1> elements
for i in range(1, numElements):
    elements.new(i*colorRampStep)

# Set color with alpha for each element of the color ramp.
# The alpha serves as emission strength; it will be multiplied by 10 in the Cycles material
for i in range(numElements):
    c = elements[i].color
    # a random integer between 0 and numColors
    colorIndex = random.randrange(0, numColors)
    # a random integer between 0 and numStrengths
    strengthIndex = random.randrange(0, numStrengths)
    c[0] = colors[colorIndex][0]
    c[1] = colors[colorIndex][1]
    c[2] = colors[colorIndex][2]
    c[3] = strengths[strengthIndex]
