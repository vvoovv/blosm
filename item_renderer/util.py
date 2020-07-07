import bpy


def initUvAlongPolygonEdge(polygon, index0, index1):
    """
    Returns:
        tuple: uVec, uv0, uv1
            uVec: A unit vector the polygon edge defined by the indices <index0> and <index1>
            uv0: UV-coordinates for the point at the polygon vertex <index0>
            uv1: UV-coordinates for the point at the polygon vertex <index1>
    """
    uVec = polygon.allVerts[polygon.indices[index1]] - polygon.allVerts[polygon.indices[index0]]
    uv0 = (0., 0.)
    uv1 = (uVec.length, 0.)
    uVec /= uv1[0]
    return uVec, uv0, uv1


def setTextureSize(assetInfo, image):
    assetInfo["textureSize"] = tuple(image.size)

def setTextureSize2(assetInfo, materialName, imageName):
    if not "textureSize" in assetInfo:
        assetInfo["textureSize"] = tuple(
            bpy.data.materials[materialName].node_tree.nodes.get(imageName).image.size
        )