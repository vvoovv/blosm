

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
        
        # either <levelHeight> or <levelHeights> is given
        levelHeight = footprint.getStyleBlockAttr("levelHeight")
        levelHeights = None
        if levelHeight:
            self.levelHeight = levelHeight
        else:
            self.levelHeights = levelHeights = footprint.getStyleBlockAttr("levelHeights")
            if not levelHeights:
                self.levelHeight = levelHeight = volumeGenerator.levelHeight
        
        # get wall levels
        numLevels = footprint.getStyleBlockAttr("numLevels")
        footprint.numLevels = numLevels
        
        h = footprint.getStyleBlockAttr("height")
        if h is None:
            h = self.calculateBottomHeight(volumeGenerator)
            if footprint.numLevels:
                h += self.calculateLevelsHeight(volumeGenerator)
            h += volumeGenerator.calculateRoofHeight(footprint)
        else:
            self.calculateBottomHeight(volumeGenerator)
            if footprint.numLevels:
                self.calculateLevelsHeight(volumeGenerator)
            # calculate roof height
            # calculate last level offset in the function below
            volumeGenerator.calculateRoofHeight(footprint)
        footprint.height = h
        return h
    
    def calculateBottomHeight(self, volumeGenerator):
        if self.levelHeight:
            h = self.footprint.getStyleBlockAttr("bottomHeight")
            if h is None:
                h = volumeGenerator.bottomHeight
        else:
            h = self.levelHeights.getBottomHeight()
        self.bottomHeight = h
        return h
    
    def calculateLevelsHeight(self, volumeGenerator):
        footprint = self.footprint
        numLevels = footprint.numLevels
        
        levelHeight = self.levelHeight
        if levelHeight:
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
        else:
            h = self.levelHeights.getHeight(0, numLevels-1)
            lastLevelHeight = self.levelHeights.getLevelHeight(numLevels-1)
        
        if footprint.lastLevelOffset:
            footprint.lastLevelOffset *= lastLevelHeight
        
        return h
    
    def calculateMinHeight(self):
        footprint = self.footprint
        minLevel = footprint.getStyleBlockAttr("minLevel")
        if minLevel:
            footprint.minLevel = minLevel
            levelHeight = self.levelHeight
            if levelHeight:
                # Calculate the height for <minLevel>
                # The heights of the bottom and the ground level have been already
                # calculated in <self.calculateHeight()>
                h = self.bottomHeight + self.groundLevelHeight if self.groundLevelHeight else levelHeight
                if minLevel > 1:
                    # the height of the levels above the ground level
                    h += (minLevel-1)*levelHeight
            else:
                h = self.levelHeights.getHeight(0, minLevel-1)
        else:
            h = footprint.getStyleBlockAttr("minHeight")
            if h is None:
                h = 0.
        footprint.minHeight = h
        return h
    
    def getLevelHeight(self, index):
        numLevels = self.footprint.levels
        if self.levelHeight:
            if not index:
                return self.groundLevelHeight
            elif index == numLevels-1:
                return self.lastLevelHeight
            else:
                return self.levelHeight
        else:
            return self.levelHeights.getLevelHeight(index)
    
    def getHeight(self, index1, index2):
        """
        Get the height of the building part starting from the level <index1> till the level <index2>
        """
        numLevels = self.footprint.levels
        if self.levelHeight:
            h = 0.
            if not index1:
                h += self.groundLevelHeight
                index1 += 1
            if index2 == numLevels-1:
                h += self.lastLevelHeight
                index2 -= 1
            if index2 >= index1:
                h += (index2-index1+1)*self.levelHeight
            return h
        else:
            return self.levelHeights.getHeight(index1, index2)