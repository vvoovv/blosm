import bpy
import os


def initUvAlongPolygonEdge(bldgVector):
    """
    Returns:
        tuple: uVec, uv0, uv1
            uVec: A unit vector for the polygon edge defined by the indices <index0> and <index1>
            uv0: UV-coordinates for the point at the polygon vertex <index0>
            uv1: UV-coordinates for the point at the polygon vertex <index1>
    """
    return bldgVector.unitVector, (0., 0.), (bldgVector.length, 0.)


def setTextureSize(assetInfo, image):
    assetInfo["textureSize"] = tuple(image.size)

def setTextureSize2(assetInfo, materialName, imageName):
    if not "textureSize" in assetInfo:
        assetInfo["textureSize"] = tuple(
            bpy.data.materials[materialName].node_tree.nodes.get(imageName).image.size
        )


def getPath(globalRenderer, path):
    if path[0] == '/':
        return os.path.join(globalRenderer.assetsDir, path[1:])
    else:
        return os.path.join(globalRenderer.assetPackageDir, path)


def getFilepath(globalRenderer, assetInfo):
    return os.path.join(
        getPath(globalRenderer, assetInfo["path"]),
        assetInfo["name"]
    )