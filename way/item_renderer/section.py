import bpy

from util.blender import addGeometryNodesModifier, useAttributeForGnInput, createPolylineMesh, loadMaterialsFromFile
from item_renderer.util import getFilepath
from ..asset_store import AssetType, AssetPart


class Section:
    
    def __init__(self):
        pass

    def init(self, globalRenderer):
        self.globalRenderer = globalRenderer
        self.assetStore = globalRenderer.assetStore
    
    def render(self, section, itemIndex, obj, pointIndexOffset):
        createPolylineMesh(obj, None, section.centerline)
        self.setModifierRoadway(obj, section, itemIndex, 0., 0.)
        self.setOffsetWeights(obj, section, pointIndexOffset)

    def setModifierRoadway(self, obj, section, itemIndex, trimLengthStart, trimLengthEnd):
        m = addGeometryNodesModifier(obj, self.gnRoadway, "Roadway")
        m["Input_2"] = section.offset
        m["Input_3"] = section.width
        useAttributeForGnInput(m, "Input_4", "offset_weight")
        self.setMaterial(m, "Input_5", AssetType.material, "demo", AssetPart.roadway, section.getStyleBlockAttr("cl"))
        # set trim lengths
        m["Input_6"] = trimLengthStart
        m["Input_7"] = trimLengthEnd
        if itemIndex:
            m["Input_9"] = itemIndex
    
    def requestNodeGroups(self, nodeGroupNames):
        nodeGroupNames.add("blosm_roadway")
    
    def setNodeGroups(self, nodeGroups):
        self.gnRoadway = nodeGroups["blosm_roadway"]
    
    def setMaterial(self, modifier, modifierAttr, assetType, group, streetPart, cl):
        # get asset info for the material
        assetInfo = self.assetStore.getAssetInfo(
            assetType, group, streetPart, cl
        )
        if assetInfo:
            # set material
            material = self.getMaterial(assetInfo)
            if material:
                modifier[modifierAttr] = material
    
    def getMaterial(self, assetInfo):
        materialName = assetInfo["material"]
        material = bpy.data.materials.get(materialName)
        
        if not material:
            material = loadMaterialsFromFile(
                getFilepath(self.globalRenderer, assetInfo),
                False,
                materialName
            )
            material = material[0] if material else None
            
        return material
    
    def setOffsetWeights(self, obj, section, pointIndexOffset):
        # Set offset weights. An offset weight is equal to
        # 1/sin(angle/2), where <angle> is the angle between <vec1> and <vec2> (see below the code)
        attributes = obj.data.attributes["offset_weight"].data
        centerline = section.centerline
        numPoints = len(centerline)
        attributes[pointIndexOffset].value = attributes[pointIndexOffset+numPoints-1].value = 1.
        
        if numPoints > 2:
            vec1 = centerline[0] - centerline[1]
            vec1.normalize()
            for centerlineIndex, pointIndex in zip(range(1, numPoints-1), range(pointIndexOffset+1, pointIndexOffset+numPoints-1)):
                vec2 = centerline[centerlineIndex+1] - centerline[centerlineIndex]
                vec2.normalize()
                vec = vec1 + vec2
                vec.normalize()
                attributes[pointIndex].value = abs(1/vec.cross(vec2))
                vec1 = -vec2