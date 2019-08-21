

_numIntervals = 170


class LevelStyles:
    
    def __init__(self):
        self.intervals = tuple(LevelStyleInterval() for _ in range(_numIntervals))


class LevelStyleInterval:
    
    def __init__(self):
        # the relate level style
        self.style = None
        # the level number where the level interval of level styles begins
        self.begin = 0
        # the level number where the level interval of level styles ends
        self.end = 0
        # the level number where the previous interval begins
        self.prev = 0
        # the level number where the next interval begins
        self.next = 0