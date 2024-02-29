from . import ItemRenderer
from util.blender import addGeometryNodesModifier, useAttributeForGnInput, createPolylineMesh
from ..asset_store import AssetType, AssetPart


class Section(ItemRenderer):
    
    def renderItem(self, section):
        createPolylineMesh(None, section.street.bm, section.centerline)
    
    def finalizeItem(self, section, itemIndex):
        self.setModifierSection(section, itemIndex, 0., 0.)
        
        #
        # set the index of the street section
        #
        obj = section.street.obj
        for pointIndex in range(self.pointIndexOffset, self.pointIndexOffset + len(section.centerline)):
            obj.data.attributes['section_index'].data[pointIndex].value = itemIndex
        
        self.setOffsetWeights(section)
        self.pointIndexOffset += len(section.centerline)
        
    def reset(self):
        self.pointIndexOffset = 0
        self.itemIndex = 0

    def setModifierSection(self, section, itemIndex, trimLengthStart, trimLengthEnd):
        m = addGeometryNodesModifier(section.street.obj, self.gnSection, "Street Section")
        m["Input_2"] = section.offset
        m["Input_3"] = section.width
        useAttributeForGnInput(m, "Input_4", "offset_weight")
        self.setMaterial(m, "Input_5", AssetType.material, "demo", AssetPart.section, section.getStyleBlockAttr("cl"))
        # set trim lengths
        m["Input_6"] = trimLengthStart
        m["Input_7"] = trimLengthEnd
        if itemIndex:
            m["Input_9"] = itemIndex
    
    def requestNodeGroups(self, nodeGroupNames):
        nodeGroupNames.add("blosm_section")
    
    def setNodeGroups(self, nodeGroups):
        self.gnSection = nodeGroups["blosm_section"]
    
    def setOffsetWeights(self, section):
        # Set offset weights. An offset weight is equal to
        # 1/sin(angle/2), where <angle> is the angle between <vec1> and <vec2> (see below the code)
        attributes = section.street.obj.data.attributes["offset_weight"].data
        centerline = section.centerline
        numPoints = len(centerline)
        pointIndexOffset = self.pointIndexOffset
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