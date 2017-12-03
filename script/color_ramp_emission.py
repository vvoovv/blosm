"""
A script to assign colors to the Color Ramp node responsible for colors and strengths of
building window emission.
"""
import bpy

nodeGroupName = "WindowEmission"

# the last element of each tuple is emission strength divided by 10
colors = (
    (0.847, 0.418, 0.087, 0.5),
    (1., 0.931, 0.455, 0.4),
    (0.847, 0.671, 0.325, 0.42),
    (1., 0.527, 0.297, 0.32)
)
numColors = len(colors)
colorRampStep = 1./numColors

nodes = bpy.data.node_groups[nodeGroupName].nodes
nodes["Modulo"].inputs[1].default_value = numColors
nodes["Divide"].inputs[1].default_value = numColors
nodes["Add"].inputs[1].default_value = colorRampStep/2.

# elements of the color ramp
elements = nodes["ColorRamp"].color_ramp.elements

# remove all but the very first one elements
for i in range(numColors-1, 0, -1):
    elements.remove(elements[i])

elements[0].position = 0.

# create <numColors-1> elements
for i in range(1, numColors):
    elements.new(i*colorRampStep)

# set colors for each element
for i in range(numColors):
    c = elements[i].color
    c[0] = colors[i][0]
    c[1] = colors[i][1]
    c[2] = colors[i][2]
    c[3] = colors[i][3]
