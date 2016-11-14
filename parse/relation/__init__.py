

class Relation:
    """
    A class to represent an OSM relation
    
    Some attributes:
        li (int): layer index used to place the related geometry to a specific Blender object
        tags (dict): OSM tags
        m: A manager used during the rendering, if None <manager.BaseManager> applies defaults
            during the rendering
        r (bool): Defines if we need to render (True) or not (False) the OSM relation
            in the function <manager.BaseManager.render(..)>
        rr: A special renderer for the OSM relation
    """
    
    # use __slots__ for memory optimization
    __slots__ = ("li", "tags", "m", "r", "rr", "valid")
    
    def __init__(self):
        self.r = False
        self.rr = None
        self.valid = True