

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
    
    The basement height is always stored in the variable <self.basementHeight>.
    """
    
    def __init__(self, footprint):
        self.footprint = footprint
        self.init()
    
    def init(self):
        self.lastLevelHeight = None
        self.levelHeight = None
        self.groundLevelHeight = None
        self.basementHeight = None
        self.levelHeights = None
        
        self.lastRoofLevelHeight = None
        self.roofLevelHeight = None
        self.roofLevelHeight0 = None
        self.roofLevelHeights = None

    def calculateHeight(self, defaults):
        footprint = self.footprint
        
        self.levelHeight = footprint.getStyleBlockAttr("levelHeight")
        # either <levelHeight> or <levelHeights> is given
        self.levelHeights = footprint.getStyleBlockAttr("levelHeights") if self.levelHeight is None else None
        
        h = footprint.getStyleBlockAttr("height")
        if h is None:
            h = self.calculateLevelsHeight(defaults)
            h += self.calculateRoofHeight(defaults)
        else:
            # calculate roof height
            # calculate last level offset in the function below
            self.calculateLevelsHeight(defaults)
            self.calculateRoofHeight(defaults)
        footprint.height = h
        return h
    
    def calculateBasementHeight(self, defaults):
        levelHeights = self.levelHeights
        h = self.footprint.getStyleBlockAttr("basementHeight")
        if h is None:
            h = levelHeights.getBasementHeight() if levelHeights else defaults.basementHeight
        self.basementHeight = h
        return h
    
    def calculateGroundLevelHeight(self, defaults):
        levelHeight = self.levelHeight
        levelHeights = self.levelHeights
        h = self.footprint.getStyleBlockAttr("groundLevelHeight")
        if h is None:
            h = levelHeight if levelHeight else (levelHeights.getLevelHeight(0) if levelHeights else defaults.groundLevelHeight)
        else:
            if levelHeights:
                # override or set the height of the ground level
                levelHeights.setLevelHeight(0, h)
            else:
                self.groundLevelHeight = h
        return h
    
    def calculateLastLevelHeight(self, defaults):
        levelHeight = self.levelHeight
        levelHeights = self.levelHeights
        h = self.footprint.getStyleBlockAttr("lastLevelHeight")
        if h is None:
            h = levelHeight if levelHeight else (levelHeights.getLevelHeight(-1) if levelHeights else defaults.lastLevelHeight)
        else:
            if levelHeights:
                # override or set the height of the last level
                levelHeights.setLevelHeight(-1, h)
            else:
                self.lastLevelHeight = h
        return h
    
    def calculateRoofHeight(self, defaults):
        footprint = self.footprint
        h = footprint.getStyleBlockAttr("roofHeight")
        if h is None:
            if self.hasRoofLevels and "roofLevels" in footprint.styleBlock:
                h = self.calculateRoofLevelsHeight(defaults)
                if footprint.lastLevelOffset:
                    h += footprint.lastLevelOffset
            else:
                # default height of the roof
                h = defaults.height
        footprint.roofHeight = h
        return h

    def calculateRoofLevelsHeight(self, defaults):
        footprint = self.footprint
        levels = footprint.getStyleBlockAttr("roofLevels")
        footprint.roofLevels = levels
        if not levels:
            return defaults.height
        
        levelHeight = footprint.getStyleBlockAttr("roofLevelHeight")
        self.roofLevelHeight = levelHeight
        # either <roofLevelHeight> or <roofLevelHeights> is given
        levelHeights = footprint.getStyleBlockAttr("rooflevelHeights") if levelHeight is None else None
        self.roofLevelHeights = levelHeights
        
        h = footprint.getStyleBlockAttr("roofLevelHeight0")
        if h is None:
            h = levelHeight if levelHeight else (levelHeights.getLevelHeight(0) if levelHeights else defaults.roofLevelHeight0)
        else:
            if levelHeights:
                # override or set the height of the ground level
                levelHeights.setLevelHeight(0, h)
            else:
                self.roofLevelHeight0 = h
        
        if levels > 1.:
            # the height of the last level
            lastLevelHeight = footprint.getStyleBlockAttr("lastRoofLevelHeight")
            if h is None:
                lastLevelHeight = levelHeight if levelHeight else (levelHeights.getLevelHeight(-1) if levelHeights else defaults.lastRoofLevelHeight)
            else:
                if levelHeights:
                    # override or set the height of the last level
                    levelHeights.setLevelHeight(-1, h)
                else:
                    self.lastRoofLevelHeight = lastLevelHeight
            h += lastLevelHeight
            
            if levels > 2.:
                # the height of the middle levels
                if levelHeights:
                    h += levelHeights.getHeight(1, -2)
                else:
                    h += (levels-2.)*levelHeight
        return h
    
    def calculateLevelsHeight(self, defaults):
        footprint = self.footprint
        levels = footprint.getStyleBlockAttr("levels")
        footprint.levels = levels
        if not levels:
            return 0.
        
        h = self.calculateBasementHeight(defaults)
        groundLevelHeight = self.calculateGroundLevelHeight(defaults)
        h += groundLevelHeight
        
        if levels == 1.:
            # the special case
            lastLevelHeight = groundLevelHeight
        else:
            # the height of the last level
            lastLevelHeight = self.calculateLastLevelHeight(defaults)
            h += lastLevelHeight
            if levels > 2.:
                # the height of the middle levels
                if self.levelHeights:
                    h += self.levelHeights.getHeight(1, -2)
                else:
                    h += (levels-2.)*self.levelHeight
        if footprint.lastLevelOffset:
            footprint.lastLevelOffset *= lastLevelHeight
        return h
    
    def calculateMinHeight(self):
        footprint = self.footprint
        minLevel = footprint.getStyleBlockAttr("minLevel")
        if minLevel:
            footprint.minLevel = minLevel
            # either <levelHeight> or <levelHeights> is given
            levelHeight = self.levelHeight
            levelHeights = self.levelHeights
            # Calculate the height for <minLevel>
            # The heights of the basement and the ground level have been already
            # calculated in <self.calculateHeight()>
            h = self.basementHeight + (self.groundLevelHeight if levelHeight else levelHeights.getLevelHeight(0))
            
            if minLevel > 1.:
                # the height of the middle levels
                if levelHeight:
                    h += (minLevel-1.)*levelHeight
                else:
                    h += levelHeights.getHeight(1, minLevel)
        else:
            h = footprint.getStyleBlockAttr("minHeight")
            if h is None:
                h = 0.
        footprint.minHeight = h
        return h