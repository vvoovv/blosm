import os
from string import Template
import bpy

from app.blender import app


def getTemplate():
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), bpy.context.scene["outputTemplate"]), 'r') as file:
        template = file.read()
    return template


def getDebugHippedRoofPath():
    return bpy.context.scene["outputDir"]


class TemplateBpypolyskel(Template):
    
    def substitute(self, *args, **kws):
        # iterate through <kws> to find Python lists or tuples to get a string out of their elements
        for k in kws:
            value = kws[k]
            if isinstance(value, (list, tuple)):
                # replace <value>
                value = "[\n    %s\n]" % ",\n    ".join(repr(element) for element in value)
                kws[k] = value
        
        return super().substitute(*args, **kws)


def dumpInputHippedRoof(verts, firstVertIndex, numPolygonVerts, holesInfo, unitVectors):
    """
    Creates a Python script with the automated tests out of the template <test_bpypolyskel.py.template>.
    The resulting file is saved to <bpypolyskel/debug> directory
    """
    with open(
        os.path.join(
            getDebugHippedRoofPath(),
            "%s%s.py" % (
                bpy.context.scene["outputFileNamePrefix"],
                os.path.splitext(os.path.basename(app.osmFilepath))[0]
            )
        ),
    'w') as file:
        file.write(
            TemplateBpypolyskel( getTemplate() ).substitute(
                verts = verts,
                unitVectors = unitVectors,
                numPolygonVerts = numPolygonVerts,
                firstVertIndex = firstVertIndex,
                holesInfo = holesInfo
            )
        )