

class LevelHeights:
    """
    A class used to calculate and store level heights (including roof levels).
    
    A simple case is when <levelHeights> are not given in the style block and
    <levelHeight> is given instead.
    <groundLevelHeight>, <levelHeight> and <lastLevelHeight> are stored in that case as separate
    variables.
    
    A complex is when <levelHeight> is not given in the style block and
    <levelHeights> are given instead.
    All level heights are stored in <levelHeights> for that case.
    
    The same applies for the roof levels.
    
    The bottom height is always stored in the variable <self.bottomHeight>.
    """
    
    def __init__(self, footprint):
        self.footprint = footprint
        self.init()
    
    def init(self):
        self.topHeight = None
        self.lastLevelHeight = None
        self.levelHeight = None
        self.groundLevelHeight = None
        self.bottomHeight = None
        
        self.levelHeights = None
        
        self.lastRoofLevelHeight = None
        self.roofLevelHeight = None
        self.roofLevelHeight0 = None
        
        self.roofLevelHeights = None
        
        # Do we have at least one of the following heights provided:
        #   <self.lastLevelHeight>
        #   <self.lastRoofLevelHeight>
        #   <self.roofLevelHeight>
        #   <self.roofLevelHeight0>
        self.multipleHeights = False

    def calculateHeight(self, volumeGenerator):
        footprint = self.footprint
        
        # either <levelHeights> of <levelHeight> is given
        self.levelHeights = levelHeights = footprint.getStyleBlockAttr("levelHeights")
        
        if not levelHeights:
            levelHeight = footprint.getStyleBlockAttr("levelHeight")
            if not levelHeight:
                levelHeight = volumeGenerator.levelHeight
            self.levelHeight = levelHeight
        
        h = footprint.getStyleBlockAttr("height")
        if h:
            self.calculateBottomHeight(volumeGenerator)
            volumeGenerator.calculateRoofHeight(footprint)
            numLevels = footprint.getStyleBlockAttr("numLevels")
            # get the number of wall levels
            if footprint.getStyleBlockAttr("hasNumLevelsAttr"):
                footprint.numLevels = numLevels
                if numLevels:
                    # first we calculate levels height
                    levelsHeight = self.calculateLevelsHeight(volumeGenerator)
                    # then we adjust levels height based on <h> and <footprint.numLevels>
                    self.adjustLevelHeights(levelsHeight, h)
            elif numLevels == 0:
                footprint.numLevels = 0
            else: # None or randomly generated in the Footprint style block
                self.calculateNumLevelsFromHeight(h)
        else:
            h = self.calculateBottomHeight(volumeGenerator)
            # get the number of wall levels
            numLevels = footprint.getStyleBlockAttr("numLevels")
            footprint.numLevels = numLevels
            if numLevels:
                h += self.calculateLevelsHeight(volumeGenerator)
            roofHeight = volumeGenerator.calculateRoofHeight(footprint)
            h += roofHeight
            if footprint.roofHeight is None:
                footprint.roofHeight = roofHeight + footprint.lastLevelOffset\
                    if footprint.lastLevelOffset\
                    else roofHeight
        footprint.height = h
        return h
    
    def calculateBottomHeight(self, volumeGenerator):
        if self.levelHeight:
            h = self.footprint.getStyleBlockAttr("bottomHeight")
            if h is None:
                h = 0.
        else:
            h = self.levelHeights.getBottomHeight()
        self.bottomHeight = h
        return h
    
    def calculateLevelsHeight(self, volumeGenerator):
        footprint = self.footprint
        numLevels = footprint.numLevels
        
        if self.levelHeights:
            h = self.levelHeights.getHeight(0, numLevels-1)
            lastLevelHeight = self.levelHeights.getLevelHeight(numLevels-1)
        else:
            levelHeight = self.levelHeight
            #
            # ground level
            #
            h = footprint.getStyleBlockAttr("groundLevelHeight")
            if h:
                self.groundLevelHeight = h
            else:
                h = levelHeight
            #
            # the levels above the ground level
            #
            if numLevels == 1:
                # the special case
                lastLevelHeight = h
            else:
                #
                # the last level
                #
                lastLevelHeight = footprint.getStyleBlockAttr("lastLevelHeight")
                if lastLevelHeight:
                    self.lastLevelHeight = lastLevelHeight
                    self.multipleHeights = True
                else:
                    lastLevelHeight = levelHeight
                h += lastLevelHeight
                #
                # the levels between the ground and the last ones
                #
                if numLevels > 2:
                    # the height of the middle levels
                    h += (numLevels-2)*levelHeight
        
        if footprint.lastLevelOffset:
            footprint.lastLevelOffset *= lastLevelHeight
        
        return h
    
    def calculateMinHeight(self):
        footprint = self.footprint
        minLevel = footprint.getStyleBlockAttr("minLevel") if footprint.numLevels else 0
        if minLevel:
            footprint.minLevel = minLevel
            if self.levelHeights:
                h = self.levelHeights.getHeight(0, minLevel-1)
            else:
                levelHeight = self.levelHeight
                # Calculate the height for <minLevel>
                # The heights of the bottom and the ground level have been already
                # calculated in <self.calculateHeight()>
                h = self.bottomHeight + self.groundLevelHeight if self.groundLevelHeight else levelHeight
                if minLevel > 1:
                    # the height of the levels above the ground level
                    h += (minLevel-1)*levelHeight
        else:
            h = footprint.getStyleBlockAttr("minHeight")
            if h is None:
                h = 0.
        footprint.minHeight = h
        return h
    
    def getLevelHeight(self, index):
        if self.levelHeight:
            if not index:
                return self.groundLevelHeight
            elif index == self.footprint.numLevels-1:
                return self.lastLevelHeight
            else:
                return self.levelHeight
        else:
            return self.levelHeights.getLevelHeight(index)
    
    def getHeight(self, index1, index2):
        """
        Get the height of the building part starting from the level <index1> till the level <index2>
        """
        if self.levelHeights:
            return self.levelHeights.getHeight(index1, index2)
        else:
            h = 0.
            if not index1:
                h += self.groundLevelHeight
                index1 += 1
            if index2 == self.footprint.numLevels-1:
                h += self.lastLevelHeight
                index2 -= 1
            if index2 >= index1:
                h += (index2-index1+1)*self.levelHeight
            return h
    
    def adjustLevelHeights(self, levelsHeight, totalHeight):
        """
        Adjust the height of levels
        """
        footprint = self.footprint
        totalHeight -= self.bottomHeight
        # calculate adjustment <factor>
        if footprint.roofHeight:
            # <roofHeight> is given or we have no roof levels
            factor = (totalHeight - footprint.roofHeight) /\
                (levelsHeight - footprint.lastLevelOffset if footprint.lastLevelOffset else levelsHeight)
        elif footprint.roofHeight is None:
            # <roofHeight> is not given but is greater than zero and
            # we must have <footprint.roofLevelsHeight> for that case
            factor = (totalHeight - self.topHeight) / (levelsHeight + footprint.roofLevelsHeight)
        else:
            # the case of a flat roof: <footprint.roofHeight == 0>
            factor = (totalHeight - self.topHeight) / levelsHeight
        
        if footprint.lastLevelOffset:
            footprint.lastLevelOffset *= factor
        
        # now adjust the heights of separate levels
        if self.levelHeights:
            self.levelHeights.adjustLevelHeights(factor)
        else:
            self.levelHeight *= factor
            if self.groundLevelHeight:
                self.groundLevelHeight *= factor
            if self.lastLevelHeight:
                self.lastLevelHeight *= factor
            
            # now deal with the roof
            if footprint.roofHeight:
                if footprint.roofLevelsHeight:
                    self.adjustRoofLevelHeights(
                        (totalHeight - factor*levelsHeight - self.topHeight)/footprint.roofLevelsHeight
                    )
            elif footprint.roofHeight is None:
                # we are ready to calculate <footprint.roofHeight>
                self.roofHeight = totalHeight - factor*levelsHeight + footprint.lastLevelOffset
                if footprint.roofLevelsHeight:
                    # adjust roof level height by <factor>
                    self.adjustRoofLevelHeights(factor)

    def adjustRoofLevelHeights(self, factor):
        """
        Adjust the height of roof levels by <factor>
        """
        if self.roofLevelHeight:
            self.roofLevelHeight *= factor
        if self.roofLevelHeight0:
            self.roofLevelHeight0 *= factor
        if self.lastRoofLevelHeight:
            self.lastRoofLevelHeight *= factor
    
    def calculateNumLevelsFromHeight(self, totalHeight):
        if self.levelHeights:
            numLevels, levelsHeight = self.levelHeights.calculateNumLevelsFromHeight(totalHeight)
        else:
            footprint = self.footprint
            # calculate the height of wall levels
            levelsHeight = totalHeight - self.bottomHeight
            if footprint.roofHeight:
                levelsHeight -= footprint.roofHeight
                if levelsHeight:
                    if footprint.lastLevelOffset:
                        levelsHeight += footprint.lastLevelOffset
                else:
                    footprint.numLevels = 0
                    return
            elif footprint.roofHeight is None:
                # <roofHeight> is not given but is greater than zero and
                # we must have <footprint.roofLevelsHeight> for that case
                levelsHeight -= footprint.roofLevelsHeight + self.topHeight
            else:
                # the case of a flat roof: <footprint.roofHeight == 0>
                levelsHeight -= self.topHeight
            
            levelHeight = self.levelHeight
            groundLevelHeight = self.groundLevelHeight or levelHeight
            if levelsHeight > groundLevelHeight:
                lastLevelHeight = self.lastLevelHeight or levelHeight
                groundAndLastLevelHeight = groundLevelHeight + lastLevelHeight
                if levelsHeight > groundAndLastLevelHeight:
                    numLevels = round( (levelsHeight-groundAndLastLevelHeight)/levelHeight ) + 2
                else:
                    numLevels = round( (levelsHeight-groundLevelHeight)/lastLevelHeight ) + 1
            else:
                numLevels = 1
        
        footprint.numLevels = numLevels
        self.adjustLevelHeights(levelsHeight, totalHeight)