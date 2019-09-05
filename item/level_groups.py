

_numGroups = 170


class LevelGroups:
    
    def __init__(self, item):
        self.item = item
        self.groups = tuple(LevelGroup() for _ in range(_numGroups))
        self.basement = None
        self.numActiveGroups = 0
    
    def init(self):
        groupCounter = 0
        for item in self.item.markup:
            if item.styleBlock.isBasement:
                self.basement = item
            else:
                self.groups[groupCounter].item = item
        self.numActiveGroups = groupCounter
    
    def clear(self):
        self.basement = None
        for i in range(self.numActiveGroups):
            self.groups[i].item = None
        self.numActiveGroups = 0


class LevelGroup:
    
    def __init__(self):
        # the relate Level item
        self.item = None
        # the level number where the level interval of level styles begins
        self.begin = 0
        # the level number where the level interval of level styles ends
        self.end = 0
        # the level number where the previous interval begins
        self.prev = 0
        # the level number where the next interval begins
        self.next = 0