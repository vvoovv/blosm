

class Entrance:
    """
    The Entrance renderer is the special case of the <item_renderer.level.Level> when
    an entrance in the only element in the level markup
    
    A mixin class for Entrance texture based item renderers
    """
    
    def __init__(self):
        self.facadePatternInfo = dict(Entrance=1)