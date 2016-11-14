from parse import Osm
from . import Relation
from .multipolygon import Multipolygon


class Building(Relation):
    
    # use __slots__ for memory optimization
    __slots__ = ("outline",)
    
    def __init__(self, osm):
        if osm.op.mode == "2D":
            # OSM relation of the type 'building' makes sense only in the 3D mode
            self.valid = False
            return
        super().__init__()
    
    def preprocess(self, members, osm):
        # the first pass: looking for a relation member with the role 'outline'
        for mType, mId, mRole in members:
            if mRole is Osm.outline:
                if mType is Osm.way:
                    # all OSM ways are already available
                    if mId in osm.ways:
                        outline = (mId, Osm.way)
                        break
                elif mType is Osm.relation:
                    # get either an existing relation encountered before or create an empty relation
                    outline = osm.getRelation(mId, Multipolygon)
                    # Ensure that the OSM relation serving as outline has <b> attribute,
                    # which will be needed later;
                    # the attribute <b> is used to store an instance of <building.manager.Building>
                    if not outline.tags:
                        # <not outline.tags> means that outline is an empty relation,
                        # i.e. not encountered in the OSM file yet
                        outline.b = None
                    outline = (mId, Osm.relation)
                    break
        else:
            # a relation member with the role 'outline' not found
            self.valid = False
            return
        self.outline = outline
    
    def process(self, members, tags, osm):
        # the first pass
        self.preprocess(members, osm)
        if not self.valid:
            return
        # The second pass:
        # For all building parts from <self.members> set attribute <o> to <self.outline>
        for mType, mId, mRole in members:
            if mRole is Osm.part:
                if mType is Osm.way:
                    # all OSM ways are already available
                    if mId in osm.ways:
                        osm.ways[mId].o = self.outline
                elif mType is Osm.relation:
                    osm.getRelation(mId, Multipolygon).o = self.outline