from math import floor

from util import zero

minLevelHeightFactor = 0.7
threshold = 0.05


class Geometry:

    def initRenderStateForDivs(self, rs, item):
        rs.indexLB = item.indices[0]
        rs.indexLT = item.indices[-1]
        rs.texUl = item.uvs[0][0]
        rs.texVlt = item.uvs[-1][1]

    def initRenderStateForLevels(self, rs, parentItem):
        parentIndices = parentItem.indices
        # <indexBL> and <indexBR> are indices of the bottom vertices of an level item to be created
        # The prefix <BL> means "bottom left"
        rs.indexBL = parentIndices[0]
        # The prefix <BR> means "bottom rights"
        rs.indexBR = parentIndices[1]
        # <texVb> is the current V-coordinate for texturing the bottom vertices of
        # level items to be created out of <parentItem>
        rs.texVb = parentItem.uvs[0][1]
        
        if rs.remainingGeometry:
            rs.remainingGeometry = None
    
    def renderLevelGroups(self, parentItem, parentRenderer):
        rs = parentRenderer.renderState
        self.initRenderStateForLevels(rs, parentItem)
        self.renderBottom(parentItem, parentRenderer, rs)
        
        levelGroups = parentItem.levelGroups
        
        if not levelGroups.begin:
            # the case when only bottom is available (no levels)
            self.renderCladdingAtTop(parentItem, parentRenderer)
            return
        
        top = levelGroups.top
        
        # Remember that <self> is exactly equeal to <parentItem.geometry>
        maxV = self.getMaxV(parentItem.uvs)
        if top:
            maxV -= top.levelHeight
        
        if levelGroups.begin.next:
            # There are at least 2 level groups or one level group and the top
            # Detach the end level group. It can be either the top or the last level group
            lastLevelGroup = levelGroups.end
            lastLevelGroup.prev.next = None
            
            levelGroup = levelGroups.begin
            while levelGroup:
                
                if not parentItem.footprint.rectangularWalls:
                    # available height in <parentItem>
                    availableHeight = maxV - rs.texVb
                    
                    groupLevels = levelGroup.index2 - levelGroup.index1 + 1
                    # how many levels can fit in <availableHeight>
                    levels = floor(availableHeight/levelGroup.levelHeight)
                    
                    if levels < groupLevels:
                        extraHeight = (availableHeight - levels * levelGroup.levelHeight)\
                            if levels else availableHeight
                        if levels:
                            # patch <levelGroup>
                            if levels == 1:
                                levelGroup.makeSingleLevel()
                            else:
                                levelGroup.index2 = levelGroup.index1 + levels - 1
                            if extraHeight > threshold:
                                # render <levelGroup> with the number of levels equal to <levels>
                                self.renderLevelGroup(
                                    parentItem, 
                                    levelGroup,
                                    levelGroup.item.getLevelRenderer(levelGroup, parentRenderer.itemRenderers),
                                    rs
                                )
                            else:
                                # no space is left besides the one for the levels in <levelGroup>
                                self.renderLastLevelGroup(
                                    parentItem,
                                    levelGroup,
                                    levelGroup.item.getLevelRenderer(levelGroup, parentRenderer.itemRenderers)\
                                        if levelGroup.item else parentRenderer,
                                    rs
                                )
                                return
                        # check if there is a space for one more level
                        if extraHeight/levelGroup.levelHeight > minLevelHeightFactor:
                            # There is a space for a one more level with a smaller height.
                            # Patch <levelGroup again>.
                            if levels:
                                levelGroup.index2 += 1
                                levelGroup.index1 = levelGroup.index2
                                if not levelGroup.singleLevel:
                                    levelGroup.singleLevel = True
                            elif not levelGroup.singleLevel:
                                levelGroup.makeSingleLevel()
                            
                            if top:
                                levelGroup.levelHeight = extraHeight
                                self.renderLevelGroup(
                                    parentItem, 
                                    levelGroup,
                                    levelGroup.item.getLevelRenderer(levelGroup, parentRenderer.itemRenderers),
                                    rs
                                )
                                # Exit the while cycle and then render <top>
                                # in the call of <self.renderLastLevelGroup(..)>
                                break
                            else:
                                self.renderLastLevelGroup(
                                    parentItem,
                                    levelGroup,
                                    levelGroup.item.getLevelRenderer(levelGroup, parentRenderer.itemRenderers)\
                                        if levelGroup.item else parentRenderer,
                                    rs
                                )
                        else:
                            # No space for an additional level. Render cladding.
                            self.renderCladdingAtTop(parentItem, parentRenderer)
                        
                        return
                    else:
                        extraHeight = availableHeight - groupLevels * levelGroup.levelHeight
                        if extraHeight <= threshold:
                            if top:
                                self.renderLevelGroup(
                                    parentItem, 
                                    levelGroup,
                                    levelGroup.item.getLevelRenderer(levelGroup, parentRenderer.itemRenderers),
                                    rs
                                )
                                # Exit the while cycle and then render <top>
                                # in the call of <self.renderLastLevelGroup(..)>
                                break
                            else:
                                # the space is only available for the levels in <levelGroup>
                                self.renderLastLevelGroup(
                                    parentItem,
                                    levelGroup,
                                    levelGroup.item.getLevelRenderer(levelGroup, parentRenderer.itemRenderers)\
                                        if levelGroup.item else parentRenderer,
                                    rs
                                )
                                return
                        # else: self.renderLevelGroup(..) below will be called
                
                self.renderLevelGroup(
                    parentItem, 
                    levelGroup,
                    levelGroup.item.getLevelRenderer(levelGroup, parentRenderer.itemRenderers),
                    rs
                )
                levelGroup = levelGroup.next
        else:
            lastLevelGroup = levelGroups.begin
        
        if not parentItem.footprint.rectangularWalls:
            # available height in <parentItem>
            availableHeight = maxV - rs.texVb
            
            if top:
                if top.item:
                    if availableHeight > threshold:
                        self.renderCladdingStripAtMiddle(availableHeight, rs)
                    # then self.renderLastLevelGroup(..) below will be called
                else:
                    # cladding for all available space
                    self.renderCladdingAtTop(parentItem, parentRenderer)
                    return
            else:
                # <lastLevelGroup> is not <top>, i.e. <lastLevelGroup> represents normal layers
                
                groupLevels = lastLevelGroup.index2 - lastLevelGroup.index1 + 1
                # how many levels can fit in <availableHeight>
                levels = floor(availableHeight/lastLevelGroup.levelHeight)
                
                if levels < groupLevels:
                    extraHeight = (availableHeight - levels * lastLevelGroup.levelHeight)\
                        if levels else availableHeight
                    if levels:
                        # patch <lastLevelGroup>
                        if levels == 1:
                            lastLevelGroup.makeSingleLevel()
                        else:
                            lastLevelGroup.index2 = lastLevelGroup.index1 + levels - 1
                        if extraHeight > threshold:
                            # render <lastLevelGroup> with the number of levels equal to <levels>
                            self.renderLevelGroup(
                                parentItem, 
                                lastLevelGroup,
                                lastLevelGroup.item.getLevelRenderer(lastLevelGroup, parentRenderer.itemRenderers),
                                rs
                            )
                        else:
                            # no space is left besides the one for the levels in <lastLevelGroup>
                            self.renderLastLevelGroup(
                                parentItem,
                                lastLevelGroup,
                                lastLevelGroup.item.getLevelRenderer(lastLevelGroup, parentRenderer.itemRenderers)\
                                    if lastLevelGroup.item else parentRenderer,
                                rs
                            )
                            return
                    # check if there is a space for one more level
                    if extraHeight/lastLevelGroup.levelHeight > minLevelHeightFactor:
                        # There is a space for a one more level with a smaller height.
                        # Patch <lastLevelGroup again>.
                        if levels:
                            lastLevelGroup.index2 += 1
                            lastLevelGroup.index1 = lastLevelGroup.index2
                            if not lastLevelGroup.singleLevel:
                                lastLevelGroup.singleLevel = True
                        elif not lastLevelGroup.singleLevel:
                            lastLevelGroup.makeSingleLevel()
                        
                        self.renderLastLevelGroup(
                            parentItem,
                            lastLevelGroup,
                            lastLevelGroup.item.getLevelRenderer(lastLevelGroup, parentRenderer.itemRenderers)\
                                if lastLevelGroup.item else parentRenderer,
                            rs
                        )
                    else:
                        # No space for an additional level. Render cladding.
                        self.renderCladdingAtTop(parentItem, parentRenderer)
                    
                    return
                else:
                    extraHeight = availableHeight - groupLevels * lastLevelGroup.levelHeight
                    if extraHeight > threshold:
                        self.renderLevelGroup(
                            parentItem, 
                            lastLevelGroup,
                            lastLevelGroup.item.getLevelRenderer(lastLevelGroup, parentRenderer.itemRenderers),
                            rs
                        )
                        self.renderCladdingAtTop(parentItem, parentRenderer)
                        return
                    # else: self.renderLastLevelGroup(..) below will be called
        
        self.renderLastLevelGroup(
            parentItem,
            lastLevelGroup,
            lastLevelGroup.item.getLevelRenderer(lastLevelGroup, parentRenderer.itemRenderers)\
                if lastLevelGroup.item else parentRenderer,
            rs
        )
    
    def renderBottom(self, parentItem, parentRenderer, rs):
        bottom = parentItem.levelGroups.bottom
        if bottom:
            self.renderLevelGroup(
                parentItem, bottom, parentRenderer.bottomRenderer, rs
            )
    
    def _renderCladding(self, item, renderer, indices, uvs):
        """
        A helper function: a wrapper for <renderer.renderCladding(..)>
        """
        renderer.renderCladding(
            item,
            renderer.r.createFace(
                item.footprint,
                indices
            ),
            uvs
        )

    @staticmethod
    def _getIndicesTriangleAtLeft(indexB, indexT, parentIndices):
        return (parentIndices[0], indexB, indexT)
    
    @staticmethod
    def _getUvsTriangleAtLeft(uvB, uvT, parentUvs):
        return (parentUvs[0], uvB, uvT)
    
    @staticmethod
    def _getIndicesTrapezoidAtLeft(indexB, indexT, parentIndices):
        return (parentIndices[0], indexB, indexT, parentIndices[3])
    
    @staticmethod
    def _getUvsTrapezoidAtLeft(uvB, uvT, parentUvs):
        return (parentUvs[0], uvB, uvT, parentUvs[3])
    
    @staticmethod
    def _getIndicesTriangleAtRight(indexB, indexT, parentIndices):
        return (indexB, parentIndices[1], indexT)
    
    @staticmethod
    def _getUvsTriangleAtRight(uvB, uvT, parentUvs):
        return (uvB, parentUvs[1], uvT)
    
    @staticmethod
    def _getIndicesTrapezoidAtRight(indexB, indexT, parentIndices):
        return (indexB, parentIndices[1], parentIndices[2], indexT)
    
    @staticmethod
    def _getUvsTrapezoidAtRight(uvB, uvT, parentUvs):
        return (uvB, parentUvs[1], parentUvs[2], uvT)
    
    @staticmethod
    def _appendVertAtBottom(verts, parentIndices, parentUvs, offset):
        return verts.append(
            verts[parentIndices[0]] + \
                offset/(parentUvs[1][0]-parentUvs[0][0]) * (verts[parentIndices[1]]-verts[parentIndices[0]])
        )
    
    @staticmethod
    def _getIndexAndUvAtLeft(verts, parentIndices, parentUvs, offsetU):
        """
        A helper function.
        
        Returns a vertex index and UV-coordinate of a point at the intersection of
        the line segment (<verts[parentIndices[0]]>, <verts[parentIndices[-1]]>) and
        # the vertical line with the U-coordinate <offsetU>
        """
        indexT = len(verts)
        k = (offsetU-parentUvs[0][0]) / (parentUvs[-1][0]-parentUvs[0][0])
        verts.append(
            verts[parentIndices[0]] + k * (verts[parentIndices[-1]] - verts[parentIndices[0]])
        )
        return\
            indexT,\
            (#uvT
                offsetU,
                parentUvs[0][1] + k * (parentUvs[-1][1] - parentUvs[0][1])
            )
    
    @staticmethod
    def _getIndexAndUvAtRight(verts, parentIndices, parentUvs, offsetU):
        """
        A helper function.
        
        Returns a vertex index and UV-coordinate of a point at the intersection of
        the line segment (<verts[parentIndices[2]]>, <verts[parentIndices[1]]>) and
        # the vertical line with the U-coordinate <offsetU>
        """
        indexT = len(verts)
        k = (offsetU-parentUvs[2][0]) / (parentUvs[1][0]-parentUvs[2][0])
        verts.append(
            verts[parentIndices[2]] + k * (verts[parentIndices[1]] - verts[parentIndices[2]])
        )
        return\
            indexT,\
            (
                offsetU,
                parentUvs[2][1] + k * (parentUvs[1][1] - parentUvs[2][1])
            )
    
    @staticmethod
    def _getIndexAndUvGeneral(verts, parentIndices, parentUvs, offsetU, index1, index2):
        """
        A helper function for the general case of the intersection of the line segment and
        the vertical line. 
        
        Returns a vertex index and UV-coordinate of a point at the intersection of
        the line segment (<verts[parentIndices[index1]]>, <verts[parentIndices[index2]]>) and
        # the vertical line with the U-coordinate <offsetU>
        """
        indexT = len(verts)
        k = (offsetU-parentUvs[index1][0]) / (parentUvs[index2][0]-parentUvs[index1][0])
        verts.append(
            verts[parentIndices[index1]] + k * (verts[parentIndices[index2]] - verts[parentIndices[index1]])
        )
        return\
            indexT,\
            (
                offsetU,
                parentUvs[index1][1] + k * (parentUvs[index2][1] - parentUvs[index1][1])
            )