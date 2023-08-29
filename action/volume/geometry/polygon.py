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