

_numGroups = 170


class LevelGroups:
    
    def __init__(self, item):
        self.item = item
        self.groups = tuple(LevelGroup() for _ in range(_numGroups))
        self.basement = None
        # a wrapper for the basement
        self.basementGroup = LevelGroup()
        self.numActiveGroups = 0
    
    def init(self):
        footprint = self.item.footprint
        numLevels = footprint.levels
        minLevel = footprint.minLevel
        groupCounter = 0
        for item in self.item.markup:
            styleBlock = item.styleBlock
            if styleBlock.isBasement:
                self.basement = self.basementGroup
                self.basement.item = item
            else:
                group = self.groups[groupCounter]
                group.item = item
                index1, index2 = styleBlock.indices
                if index1 < minLevel:
                    index1 = minLevel
                if index1 == index2:
                    group.singleLevel = True
                    if index1 < 0:
                        index1 += numLevels
                    group.index1 = index1
                else:
                    if index1 < 0:
                        index1 += numLevels
                    if index2 < 0:
                        index2 += numLevels
                    group.index1 = index1
                    if index1 == index2:
                        group.singleLevel = True
                    else:
                        group.index2 = index2
                groupCounter += 1
        self.numActiveGroups = groupCounter
    
    def clear(self):
        self.basement = None
        for i in range(self.numActiveGroups):
            self.groups[i].clear()
        self.numActiveGroups = 0


class LevelGroup:
    
    def __init__(self):
        # the relate Level item
        self.item = None
        self.index1 = 0
        self.index2 = 0
        self.singleLevel = False
        # the level number where the level interval of level styles begins
        self.begin = 0
        # the level number where the level interval of level styles ends
        self.end = 0
        # the level number where the previous interval begins
        self.prev = 0
        # the level number where the next interval begins
        self.next = 0
    
    def clear(self):
        self.item = None
        if self.singleLevel:
            self.singleLevel = False