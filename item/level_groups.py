

class LevelGroups:
    
    def __init__(self, item):
        self.item = item
        totalNumLevels = item.footprint.totalNumLevels
        self.numGroups = totalNumLevels if totalNumLevels < 5 else (
            5 if totalNumLevels < 24 else totalNumLevels // 4
        )
        self.groups = [
            LevelGroup() for _ in range(self.numGroups)
        ]
        # a wrapper level group for the bottom
        self.bottomGroup = LevelGroup()
        # setting <singleLevel> to <True> is needed for correct height calculation of the level group
        self.bottomGroup.singleLevel = True
        self.bottom = None
        # a level group just above the bottom; it may not start from the ground level
        self.begin = None
        # a wrapper level group for the top
        self.topGroup = LevelGroup()
        self.topGroup.singleLevel = True
        # setting <singleLevel> to <True> is needed for correct height calculation of the level group
        self.top = None
        self.numUsedGroups = 0
    
    def init(self):
        item = self.item
        footprint = item.footprint
        lh = footprint.levelHeights
        numLevels = footprint.numLevels
        totalNumLevels = footprint.totalNumLevels
        minLevel = item.minLevel
        top = None
        bottom = None
        begin = None
        end = None
        groupCounter = 0
        
        # check if have the top
        topHeight = item.getStyleBlockAttr("topHeight")
        if topHeight is None:
            topHeight = lh.topHeight
        if topHeight:
            top = self.topGroup
            top.levelHeight = topHeight
            self.top = top
        # check if have the bottom
        if not item.minHeight:
            bottomHeight = item.getStyleBlockAttr("bottomHeight")
            if bottomHeight is None:
                bottomHeight = lh.bottomHeight
            if item.building.renderInfo.altitudeDifference:
                self.bottomGroup.levelHeight = item.building.renderInfo.altitudeDifference
            if bottomHeight:
                self.bottomGroup.levelHeight += bottomHeight
            if self.bottomGroup.levelHeight:
                self.bottom = bottom = self.bottomGroup
        
        # We need to convert the negative level indices to the absolute (and positive) ones.
        # If there is at least one element in the markup with the attribute <roof> equal to <True>,
        # then <numLevels> will be added to the negative level index.
        # Otherwise <totalNumLevels> will be to the negative level index.
        # Note, that if an element in the markup has the attribute <roof> equal to <True> and
        # a negative level index, than <totalNumLevels> will be added to that negative level index.
        negativeLevelIndexBase = numLevels\
            if sum(_item.styleBlock.roof for _item in item.markup) else\
            totalNumLevels
        
        for _item in reversed(item.markup):
            styleBlock = _item.styleBlock
            if styleBlock.isBottom:
                if bottom:
                    bottom.item = _item
            elif styleBlock.isTop:
                if top:
                    top.item = _item
            else:
                group = self.groups[groupCounter]
                group.item = _item
                index1, index2 = styleBlock.indices
                
                if index1 == index2:
                    if styleBlock.roof:
                        index1 += totalNumLevels if index1 < 0 else numLevels
                    elif index1 < 0:
                        index1 += negativeLevelIndexBase
                    # sanity check
                    if index1 < minLevel or index1 >= totalNumLevels:
                        continue
                    group.index1 = group.index2 = index1
                    group.singleLevel = True
                else:
                    if styleBlock.roof:
                        index1 += totalNumLevels if index1 < 0 else numLevels
                        index2 += totalNumLevels if index2 < 0 else numLevels
                    else:
                        if index1 < 0:
                            index1 += negativeLevelIndexBase
                        if index2 < 0:
                            index2 += negativeLevelIndexBase
                    # sanity check
                    if index2 < minLevel:
                        continue
                    elif index1 < minLevel:
                        index1 = minLevel
                    elif index1 >= totalNumLevels:
                        continue
                    elif index2 >= totalNumLevels:
                        index2 = totalNumLevels-1
                    
                    if index1 == index2:
                        group.index1 = group.index2 = index1
                        group.singleLevel = True
                    else:
                        group.index1 = index1
                        group.index2 = index2
                
                if begin:
                    if index1 > end.index2:
                        end.next = group
                        group.prev = end
                        end = group
                    else:
                        pass
                else:
                    begin = end = group
                
                groupCounter += 1
                if groupCounter >= self.numGroups:
                    break
        
        # Check if we need to split some level groups due to the different level height inside a level group
        if lh.levelHeight:
            if not lh.multipleHeights:
                # Treat the most common case when we have not more than two heights provided:
                # <lh.levelHeight> and optionally <lh.groundLevelHeight>
                if lh.groundLevelHeight and not minLevel:
                    if begin.singleLevel:
                        # override <buildingPart> for an item
                        #begin.buildingPart = "groundlevel"
                        group = begin
                    else:
                        # split <begin>
                        group = self.groups[groupCounter]
                        groupCounter += 1
                        group.item = begin.item
                        group.index1 = 0
                        group.makeSingleLevel()
                        group.next = begin
                        begin.prev = group
                        begin.index1 += 1
                        if begin.index1 == begin.index2:
                            begin.singleLevel = True
                        begin = group
                    group.levelHeight = lh.groundLevelHeight
                    group = group.next
                else:
                    group = begin
                # set the height for rest of the levels
                while group:
                    group.levelHeight = lh.levelHeight
                    group = group.next
        
        # attach <top> to the linked list
        if top:
            end.next = top
            top.prev = end
            end = top
        self.end = end
        self.begin = begin
        
        self.numUsedGroups = groupCounter
    
    def clear(self):
        if self.bottom:
            self.bottom = None
        if self.top:
            self.top = None
        for i in range(self.numUsedGroups):
            self.groups[i].clear()
        self.numUsedGroups = 0


class LevelGroup:
    
    def __init__(self):
        self.buildingPart = None
        # the related Level item
        self.item = None
        self.index1 = 0
        self.index2 = 0
        self.singleLevel = False
        # the level number where the previous interval begins
        self.prev = None
        # the level number where the next interval begins
        self.next = None
        # a height of a single level for the level group
        self.levelHeight = 0.
    
    def clear(self):
        self.item = None
        self.prev = None
        self.next = None
        if self.singleLevel:
            self.singleLevel = False
        if self.buildingPart:
            self.buildingPart = None
    
    def makeSingleLevel(self):
        self.index2 = self.index1
        self.singleLevel = True