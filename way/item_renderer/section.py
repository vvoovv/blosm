from . import ItemRenderer
from util.blender import addGeometryNodesModifier, useAttributeForGnInput, createPolylineMesh
from ..asset_store import AssetType, AssetPart


class Section(ItemRenderer):
    
    def initItemCenterline1(self, section, singleItem):
        street = section.street
        if singleItem:
            # create a polyline mesh
            createPolylineMesh(None, street.bm, section.centerline, None)
        else:
            # create a polyline mesh and set a BMesh vertex for the next section
            street.bmVert = createPolylineMesh(None, street.bm, section.centerline, street.bmVert)
    
    def initItemCenterline2(self, section, itemIndex):
        #
        # set the index of the street section
        #
        street = section.street
        obj = street.obj
        numEdges = len(section.centerline) - 1
        for pointIndex in range(street.edgeIndexOffset, street.edgeIndexOffset + numEdges):
            obj.data.attributes['section_index'].data[pointIndex].value = itemIndex
        
        street.edgeIndexOffset += numEdges
    
    def finalizeItem(self, section, itemIndex):
        self.setModifierSection(section, itemIndex)

    def setModifierSection(self, section, itemIndex):
        m = addGeometryNodesModifier(section.street.obj3d, self.gnSection, "Street Section")
        m["Input_2"] = section.offset
        m["Input_3"] = section.width
        # Offset weights are now calculated in the Geometry Nodes
        #useAttributeForGnInput(m, "Input_4", "offset_weight")
        self.setMaterial(m, "Input_5", AssetType.material, "demo", AssetPart.section, section.getClass())
        if itemIndex:
            m["Input_9"] = itemIndex
    
    def requestNodeGroups(self, nodeGroupNames):
        nodeGroupNames.add("Blosm Street Section")
    
    def setNodeGroups(self, nodeGroups):
        self.gnSection = nodeGroups["Blosm Street Section"]
    
    def setOffsetWeights(self, section):
        # This method is not used anymore. Offset weights (a reversed sine of half angle between the edges) are set in the Geometry Nodes.
        
        # Set offset weights. An offset weight is equal to
        # 1/sin(angle/2), where <angle> is the angle between <vec1> and <vec2> (see below the code)
        attributes = section.street.obj.data.attributes["offset_weight"].data
        centerline = section.centerline
        numPoints = len(centerline)
        edgeIndexOffset = section.street.edgeIndexOffset
        attributes[edgeIndexOffset].value = attributes[edgeIndexOffset+numPoints-1].value = 1.
        
        if numPoints > 2:
            vec1 = centerline[0] - centerline[1]
            vec1.normalize()
            for centerlineIndex, pointIndex in zip(range(1, numPoints-1), range(edgeIndexOffset+1, edgeIndexOffset+numPoints-1)):
                vec2 = centerline[centerlineIndex+1] - centerline[centerlineIndex]
                vec2.normalize()
                vec = vec1 + vec2
                vec.normalize()
                attributes[pointIndex].value = abs(1./vec.cross(vec2))
                vec1 = -vec2