from . import Geometry
from util import zero


class PolygonHB(Geometry):
    """
    A polygon with the horizontal base
    """
    
    def __init__(self, geometryTriangle):
        self.geometryTriangle = geometryTriangle
    
    def initRenderStateForLevels(self, rs, parentItem):
        super().initRenderStateForLevels(rs, parentItem)
        rs.uvBL, rs.uvBR = parentItem.uvs[0], parentItem.uvs[1]
        rs.startIndexL = -1
        rs.startIndexR = 2
    
    def renderLevelGroup(self, parentItem, levelGroup, levelRenderer, rs):
        if rs.remainingGeometry and not rs.remainingGeometry is self:
            rs.remainingGeometry.renderLevelGroup(parentItem, levelGroup, levelRenderer, rs)
            return
        
        parentIndices = parentItem.indices
        parentUvs = parentItem.uvs
        
        verts = parentItem.building.renderInfo.verts
        startIndexL, startIndexR = rs.startIndexL, rs.startIndexR
        # initialize the variables <indexTL> and <indexTR>
        indexTL = indexTR = 0
        
        height = levelGroup.levelHeight\
            if levelGroup.singleLevel else\
            levelGroup.levelHeight * (levelGroup.index2 - levelGroup.index1 + 1)
        texVt = rs.texVb + height
        
        # Check the condition for the left side
        indicesL = []
        uvsL = []
        while True:
            if texVt <= parentUvs[startIndexL][1] + zero:
                if texVt >= parentUvs[startIndexL][1] - zero:
                    # use existing vertex
                    indexTL = parentIndices[startIndexL]
                    uvsL.append(parentUvs[startIndexL])
                    startIndexL -= 1
                else:
                    indexTL = len(verts)
                    # factor
                    k = (parentUvs[startIndexL][1] - texVt) / (texVt - parentUvs[startIndexL+1][1])
                    verts.append(
                        (k*verts[parentIndices[startIndexL+1]] + verts[parentIndices[startIndexL]])/(1.+k)
                    )
                    uvsL.append((
                        (k*parentUvs[startIndexL+1][0] + parentUvs[startIndexL][0])/(1.+k),
                        (k*parentUvs[startIndexL+1][1] + parentUvs[startIndexL][1])/(1.+k)
                    ))
                indicesL.append(indexTL)
                break
            indicesL.append(parentIndices[startIndexL])
            uvsL.append(parentUvs[startIndexL])
            startIndexL -= 1

        # Check the condition for the left side
        # the following variable is used if <rs.remainingGeometry> is set
        _uvIndex = 0
        indicesR = [rs.indexBL, rs.indexBR]
        uvsR = [ rs.uvBL, rs.uvBR ]
        while True:
            if texVt <= parentUvs[startIndexR][1] + zero:
                if texVt >= parentUvs[startIndexR][1] - zero:
                    # use existing vertex
                    indexTR = parentIndices[startIndexR]
                    uvsR.append(parentUvs[startIndexR])
                    startIndexR += 1
                else:
                    indexTR = len(verts)
                    # factor
                    k = (parentUvs[startIndexR][1] - texVt) / (texVt - parentUvs[startIndexR-1][1])
                    verts.append(
                        (k*verts[parentIndices[startIndexR-1]] + verts[parentIndices[startIndexR]])/(1.+k)
                    )
                    uvsR.append((
                        (k*parentUvs[startIndexR-1][0] + parentUvs[startIndexR][0])/(1.+k),
                        (k*parentUvs[startIndexR-1][1] + parentUvs[startIndexR][1])/(1.+k)
                    ))
                indicesR.append(indexTR)
                break
            indicesR.append(parentIndices[startIndexR])
            uvsR.append(parentUvs[startIndexR])
            startIndexR += 1
        # set <uvsR> to the last index of <uvsR>
        _uvIndex = len(indicesR)-1
        indicesR.extend(reversed(indicesL))
        uvsR.extend(reversed(uvsL))
        
        item = levelGroup.item
        if item:
            # Set the geometry for the <levelGroup.item>;
            # the lower part of <self> after a devision can be
            # either a rectange (processed in the if-clause) or <self>
            item.geometry = self
        
        if item and item.markup:
            item.indices = indicesR
            item.uvs = uvsR
            levelRenderer.renderDivs(item, levelGroup)
        else:
            levelRenderer.renderLevelGroup(
                item or parentItem,
                levelGroup,
                indicesR,
                uvsR
            )
        rs.indexBL = indexTL
        rs.indexBR = indexTR
        rs.texVb = texVt
        rs.startIndexL = startIndexL
        rs.startIndexR = startIndexR
        rs.uvBL, rs.uvBR = uvsR[_uvIndex+1], uvsR[_uvIndex]
        
        # the condition for a triangle as the remaining geometry
        if len(parentIndices)+startIndexL == startIndexR:
            rs.remainingGeometry = self.geometryTriangle
            rs.indices = (indexTL, indexTR, parentIndices[startIndexR])
            rs.uvs = (
                uvsR[_uvIndex+1], uvsR[_uvIndex], parentUvs[startIndexR]
            )
    
    def getFinalUvs(self, numItemsInFace, numLevelsInFace, numTilesU, numTilesV, itemUvs):
        offsetU, offsetV = itemUvs[0]
        # Calculate the height of <item> as the difference between the largest V-coordinate and
        # the smallest one
        h = max(itemUvs[i][1] for i in range(2, len(itemUvs))) - offsetV
        # Calculate the width of <item> as the difference between the largest U-coordinate and
        # the smallest one
        w = itemUvs[1][0] - itemUvs[0][0]
        
        u = numItemsInFace/numTilesU
        v = numLevelsInFace/numTilesV
        
        factorU, factorV = u/w, v/h
        
        uvs = [
            (0., 0.,),
            (u, 0.)
        ]
        uvs.extend(
            ( (itemUvs[i][0]-offsetU)*factorU, (itemUvs[i][1]-offsetV)*factorV )\
                for i in range(2, len(itemUvs))
        )
        
        return uvs
    
    def renderCladdingAtTop(self, parentItem, parentRenderer):
        rs = parentRenderer.renderState
        
        if rs.remainingGeometry and not rs.remainingGeometry is self:
            rs.remainingGeometry.renderCladdingAtTop(parentItem, parentRenderer)
        else:
            parentIndices = parentItem.indices
            parentUvs = parentItem.uvs
            
            indices = [rs.indexBL, rs.indexBR]
            indices.extend( parentIndices[i] for i in range(rs.startIndexR, len(parentIndices)+rs.startIndexL+1) )
            uvs = [ rs.uvBL, rs.uvBR ]
            uvs.extend( parentUvs[i] for i in range(rs.startIndexR, len(parentIndices)+rs.startIndexL+1) )
            
            self._renderCladding(parentItem, parentRenderer, indices, uvs)
    
    def offsetFromLeft(self, renderer, item, parentIndices, parentUvs, offset):
        self._offsetFromLeft(
            renderer, item, parentIndices, parentUvs, offset,
            parentUvs[0][0] == parentUvs[-1][0],
            parentUvs[1][0] == parentUvs[2][0]
        )
    
    def _offsetFromLeft(self, renderer, item, parentIndices, parentUvs, offset, raL, raR):
        """
        Args:
            raL (bool): There is a right angle to the left
            raR (bool): There is a right angle to the right
        """
        verts = item.building.renderInfo.verts
        
        # the new vertex at the bottom
        indexB = len(verts)
        verts.append(
            verts[parentIndices[0]] + \
                offset/(parentUvs[1][0]-parentUvs[0][0]) * (verts[parentIndices[1]]-verts[parentIndices[0]])
        )
        # add offset
        offset += parentUvs[0][0]
        uvB = (offset, parentUvs[0][1])
        
        # initialize the variables to be used below in the code
        indexT = 0
        uvT = indicesOffset = uvsOffset = indicesGeometry = uvsGeometry = None
        
        numVerts = len(parentIndices)
        startIndex = -1 if raL else 0
        endIndex = (2-numVerts) if raR else (1-numVerts)
        # the value of <endIndexExtra> is used in the <for>-cycle below
        endIndexExtra = endIndex + 1
        for i1, i2 in zip(range(startIndex, endIndex, -1), range(startIndex-1, endIndex-1, -1)):
            if parentUvs[i1][0] < offset <= parentUvs[i2][0] + zero:
                if offset >= parentUvs[i2][0] - zero:
                    # use existing vertex
                    indexT = parentIndices[i2]
                    uvT = parentUvs[i2]
                    if i2==endIndexExtra:
                        if raR:
                            # change <item.geometry> to <TrapezoidRV>
                            item.geometry = self.geometryTrapezoidR if parentUvs[endIndexExtra][1] > parentUvs[endIndex][1] else self.geometryTrapezoidL
                        else:
                            # change <item.geometry> to <Triangle>
                            item.geometry = self.geometryTriangle
                else:
                    indexT = len(verts)
                    k = (offset-parentUvs[i1][0]) / (parentUvs[i2][0]-parentUvs[i1][0])
                    verts.append(
                        verts[parentIndices[i1]] + k * (verts[parentIndices[i2]] - verts[parentIndices[i1]])
                    )
                    uvT = (
                        offset,
                        parentUvs[i1][1] + k * (parentUvs[i2][1] - parentUvs[i1][1])
                    )
                    if i2==endIndex:
                        if raR:
                            # change <item.geometry> to <TrapezoidRV>
                            item.geometry = self.geometryTrapezoidR if parentUvs[endIndexExtra][1] > parentUvs[endIndex][1] else self.geometryTrapezoidL
                        else:
                            # change <item.geometry> to <Triangle>
                            item.geometry = self.geometryTriangle
                break
        
        # absolute value of the index <i1>
        abs_i1 = numVerts+i1
        #
        # offset area
        #
        if i1 == startIndex:
            if raL:
                # the offset area has the geometry of <TrapezoidRV>
                indicesOffset = (parentIndices[0], indexB, indexT, parentIndices[-1])
                uvsOffset = (parentUvs[0], uvB, uvT, parentUvs[-1])
            else:
                # the offset area has the geometry of <Triangle>
                indicesOffset = (parentIndices[0], indexB, indexT)
                uvsOffset = (parentUvs[0], uvB, uvT)
        else:
            # the general case
            indicesOffset = [parentIndices[0], indexB, indexT]
            indicesOffset.extend(parentIndices[i] for i in range(abs_i1, numVerts))
            uvsOffset = [parentUvs[0], uvB, uvT]
            uvsOffset.extend(parentUvs[i] for i in range(abs_i1, numVerts))
        
        #
        # the remaining geometry after applying the offset
        #
        if item.geometry is self:
            indicesGeometry = [indexB]
            indicesGeometry.extend(parentIndices[i] for i in range(1, abs_i1))
            
            uvsGeometry = [uvB]
            uvsGeometry.extend(parentUvs[i] for i in range(1, abs_i1))
            
            if indexT != parentIndices[i2]:
                indicesGeometry.append(indexT)
                uvsGeometry.append(uvT)
        else:
            if raR:
                # <TrapezoidRV>
                indicesGeometry = (indexB, parentIndices[1], parentIndices[2], indexT)
                uvsGeometry = (uvB, parentUvs[1], parentUvs[2], uvT)
            else:
                # <Triangle>
                indicesGeometry = (indexB, parentIndices[1], indexT)
                uvsGeometry = (uvB, parentUvs[1], uvT)
        
        item.indices = indicesGeometry
        item.uvs = uvsGeometry
        
        # render offset area
        self._renderCladding(item, renderer, indicesOffset, uvsOffset)
    
    def offsetFromRight(self, renderer, item, parentIndices, parentUvs, offset):
        self._offsetFromRight(
            renderer, item, parentIndices, parentUvs, offset,
            parentUvs[0][0] == parentUvs[-1][0],
            parentUvs[1][0] == parentUvs[2][0]
        )
    
    def _offsetFromRight(self, renderer, item, parentIndices, parentUvs, offset, raL, raR):
        """
        Args:
            raL (bool): There is a right angle to the left
            raR (bool): There is a right angle to the right
        """
        verts = item.building.renderInfo.verts
        
        # convert <offset> to the distance from the leftmost vertex
        offset = parentUvs[1][0] - parentUvs[0][0] - offset
        
        # the new vertex at the bottom
        indexB = len(verts)
        verts.append(
            verts[parentIndices[0]] + \
                offset/(parentUvs[1][0]-parentUvs[0][0]) * (verts[parentIndices[1]]-verts[parentIndices[0]])
        )
        # add offset
        offset += parentUvs[0][0]
        uvB = (offset, parentUvs[0][1])
        
        # initialize the variables to be used below in the code
        indexT = 0
        uvT = indicesOffset = uvsOffset = indicesGeometry = uvsGeometry = None
        
        numVerts = len(parentIndices)
        startIndex = (2-numVerts) if raR else (1-numVerts)
        endIndex = -1 if raL else 0
        # the value of <endIndexExtra> is used in the <for>-cycle below
        endIndexExtra = endIndex - 1
        for i2, i1 in zip(range(startIndex+1, endIndex+1), range(startIndex, endIndex)):
            if parentUvs[i2][0] - zero <= offset < parentUvs[i1][0]:
                if offset <= parentUvs[i2][0] + zero:
                    # use existing vertex
                    indexT = parentIndices[i2]
                    uvT = parentUvs[i2]
                    if i2 == endIndexExtra:
                        if raL:
                            # change <item.geometry> to <TrapezoidRV>
                            item.geometry = self.geometryTrapezoidR if parentUvs[endIndex][1] > parentUvs[endIndexExtra][1] else self.geometryTrapezoidL
                        else:
                            # change <item.geometry> to <Triangle>
                            item.geometry = self.geometryTriangle
                else:
                    indexT = len(verts)
                    k = (offset-parentUvs[i2][0]) / (parentUvs[i1][0]-parentUvs[i2][0])
                    verts.append(
                        verts[parentIndices[i2]] + k * (verts[parentIndices[i1]] - verts[parentIndices[i2]])
                    )
                    uvT = (
                        offset,
                        parentUvs[i2][1] + k * (parentUvs[i1][1] - parentUvs[i2][1])
                    )
                    if i2==endIndex:
                        if raL:
                            # change <item.geometry> to <TrapezoidRV>
                            item.geometry = self.geometryTrapezoidR if parentUvs[endIndex][1] > parentUvs[endIndexExtra][1] else self.geometryTrapezoidL
                        else:
                            # change <item.geometry> to <Triangle>
                            item.geometry = self.geometryTriangle
                break
        
        # absolute value of the index <i1>
        abs_i2 = numVerts+i2
        #
        # offset area
        #
        if i1 == startIndex:
            if raR:
                # the offset area has the geometry of TrapezoidRV
                indicesOffset = (indexB, parentIndices[1], parentIndices[2], indexT)
                uvsOffset = (uvB, parentUvs[1], parentUvs[2], uvT)
            else:
                # the offset area has the geometry of <Triangle>
                indicesOffset = (indexB, parentIndices[1], indexT)
                uvsOffset = (uvB, parentUvs[1], uvT)
        else:
            # the general case
            indicesOffset = [indexB]
            indicesOffset.extend(parentIndices[i] for i in range(1, abs_i2))
            indicesOffset.append(indexT)
            
            uvsOffset = [uvB]
            uvsOffset.extend(parentUvs[i] for i in range(1, abs_i2))
            uvsOffset.append(uvT)
        
        #
        # the remaining geometry after applying the offset
        #
        if item.geometry is self:
            if indexT == parentIndices[i2]:
                indicesGeometry = [parentIndices[0], indexB]
                uvsGeometry = [parentUvs[0], uvB]
            else:
                indicesGeometry = [parentIndices[0], indexB, indexT]
                uvsGeometry = [parentUvs[0], uvB, uvT]
            
            indicesGeometry.extend(parentIndices[i] for i in range(abs_i2, numVerts))
            
            uvsGeometry.extend(parentUvs[i] for i in range(abs_i2, numVerts))
        else:
            if raL:
                # <TrapezoidRV>
                indicesGeometry = (parentIndices[0], indexB, indexT, parentIndices[-1])
                uvsGeometry = (parentUvs[0], uvB, uvT, parentUvs[-1])
            else:
                # <Triangle>
                indicesGeometry = (parentIndices[0], indexB, indexT)
                uvsGeometry = (parentUvs[0], uvB, uvT)
        
        item.indices = indicesGeometry
        item.uvs = uvsGeometry
        
        # render offset area
        self._renderCladding(item, renderer, indicesOffset, uvsOffset)