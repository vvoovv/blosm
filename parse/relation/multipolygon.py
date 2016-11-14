from parse import Osm
from . import Relation
from renderer import Renderer


class Linestring:
    """
    An auxiliary class to represent a linestring (with open ends or closed)
    
    The linestring can be composed of a number connected OSM ways in the following way.
    
    OSM node id <self.start> defines the start of the linestring.
    <self.end is None> if the linestring is closed,
    otherwise it defines the end of the linestring
    <--->
    Let nodeId = self.start
    The id of the current OSM way <wayId> of the linestring is found then:
    wayId, direct = self.parts[nodeId]
    If <direct> is True, then <nodeId> is the first node of the OSM way <wayId>,
    if <direct> is False, then <nodeId> is the last node of the OSM way <wayId>.
    
    To go to the next OSM way in the linestring, find the id of the OSM node
    opposite to <nodeId> as follow:
    if <direct> is True, take the last node of the OSM way <wayId>,
    if <direct> is False, take the first node of the OSM way <wayId>
    
    Assign that id to <nodeId> and continue from <--->
    
    Continue the process until:
    <nodeId == self.start> for the closed linestring;
    <not nodeId in self.parts> for linestring with open ends
    
    If the linstring is closed and composed of exactly one OSM way with the id <wayId>,
    then <self.parts = None> and <self.wayId = wayId>
    """
    
    # use __slots__ for memory optimization
    __slots__ = ("parts", "role", "wayId", "start", "end")
    
    def __init__(self, role, wayId, start=None, end=None):
        """
        If <start> and <end> aren't provided, it means, that the OSM way <wayId> is closed and
        therefore the only one for the linestring to be created.
        
        Args:
            role: Role of OSM way <wayId> in the corresponding OSM relation
            wayId: Id of the initial OSM way
            start: Id of the OSM node at the start of the OSM way <wayId>
            end: Id of the OSM node at the end of the OSM way <wayId>
        """
        self.role = role
        if start is None: # <end is None> also in this case
            self.wayId = wayId
            self.parts = None
        else:
            parts = {}
            self.parts = parts
            parts[start] = (wayId, True)
            self.parts = parts
            self.start = start
            self.end = end
    
    def nodeIds(self, osm):
        """
        A generator to get id of OSM nodes of the linestring
        """
        parts = self.parts
        
        if parts:
            nodeId = self.start
            # <self.end is None> means that the linestring is closed
            breakNodeId = self.start if self.end is None else self.end
            while True:
                wayId, direct = parts[nodeId]
                way = osm.ways[wayId]
                nodes = way.nodes
                for i in range(way.n - 1) if direct else range(way.n - 1, 0, -1):
                    yield nodes[i]
                nodeId = nodes[-1] if direct else nodes[0]
                if nodeId == breakNodeId:
                    # for the open linestring also yield the last node
                    if not self.end is None:
                        yield nodeId
                    break
        else:
            way = osm.ways[self.wayId]
            for i in range(way.n):
                yield way.nodes[i]
    
    def extend(self, wayId, start, end, connectWayStart):
        """
        Extend the linestring with the OSM way <wayId>
        
        Args:
            wayId: Id of the OSM way that is to extend the linestring
            start: Id of the OSM node at the start of the OSM way <wayId>
            end: Id of the OSM node at the end of the OSM way <wayId>
            connectWayStart (bool): connect the start (True) or the end (False) of the OSM way <wayId> to <self>
        """
        # consider below all possible cases
        if connectWayStart:
            if start == self.start:
                # the way goes into the start of the linestring in the reverse order
                key = end
                direct = False
                self.start = end
                extendedStart = True
            else: #start == self.end:
                # the way goes into the end of the linestring in the direct order
                key = start
                direct = True
                self.end = end
                extendedStart = False
        else:
            if end == self.start:
                # the way goes into the start of the linestring in the direct order
                key = start
                direct = True
                self.start = start
                extendedStart = True
            else: #end == self.end:
                # the way goes into the end of the linestring in the reverse order
                key = end
                direct = False
                self.end = start
                extendedStart = False
        self.parts[key] = (wayId, direct)
        return extendedStart
    
    def close(self, wayId, start, end):
        """
        Close the linestring with the OSM way <wayId>
        
        Args:
            wayId: Id of the closing OSM way
            start: Id of the OSM node at the start of the OSM way <wayId>
            end: Id of the OSM node at the end of the OSM way <wayId>
        """
        direct = start == self.end
        self.parts[start if direct else end] = (wayId, direct)
        self.end = None
    
    def connect(self, l, connectToStart, osm):
        """
        Connect another linestring <l> to <self>
        
        Args:
            l (Linestring): Linestring to connect to <self>
            connectToStart (bool): connect <l> to the start (True) or to the end (False) of <self>
            osm (parse.Osm): OSM data
        """
        def copyPartsKeepDirection():
            for key in l.parts:
                self.parts[key] = l.parts[key]
        def copyPartsReverseDirection():
            for key in l.parts:
                wayId, direct = l.parts[key]
                self.parts[ osm.ways[wayId].nodes[-1 if direct else 0] ] = (wayId, not direct)
        
        if connectToStart:
            if l.end == self.start:
                copyPartsKeepDirection()
                start = l.start
            else: # l.start == self.start
                copyPartsReverseDirection()
                start = l.end
            self.start = start   
        else:
            if l.start == self.end:
                copyPartsKeepDirection()
                end = l.end
            else: #l.end = self.end
                copyPartsReverseDirection()
                end = l.start
            self.end = end


class Multipolygon(Relation):
    """
    A class to represent OSM relation of the type 'multipolygon'
    
    Attributes in addition to the parent class <Relation>:
        t: type for rendering (Render.polygon, Render.multipolygon, Render.linestring, Render.multipolygon)
        b (building.manager.Building): A related 3D Building; set only for 3D buildings
        o (tuple): Defines the related outline for a building part (building:part=*);
            has the form (osmId, osmElement); set only in the case of 3D buildings
        l (Linestring or tuple of Linestring's): linestring(s) the relation is composed of
    """
    
    # use __slots__ for memory optimization
    __slots__ = ("t", "l", "b", "o")
    
    def __init__(self, osm=None):
        super().__init__()
        # mark the relation as empty
        self.tags = None
    
    def process(self, members, tags, osm):
        # store <tags>
        self.tags = tags
        # For each linestring under processing we store in <linestrings> two entries:
        # 1) <start of the linestring (nodeId)> -> <linestring object>
        # 2) <end of the linestring (nodeId)> -> <linestring object>
        linestrings = {}
        polygons = []
        ways = osm.ways
        for mType, mId, mRole in members:
            if not (mType is Osm.way and mId in ways):
                continue
            way = osm.ways[mId]
            if not way.valid:
                continue
            if way.closed:
                # no special processing is need, just take the way as is
                polygons.append(Linestring(mRole, mId))
            else:
                start = way.nodes[0]
                end = way.nodes[-1]
                if start in linestrings and end in linestrings:
                    l1 = linestrings[start]
                    l2 = linestrings[end]
                    if l1 is l2:
                        # event: close the linestring with the <way>
                        # ensure that the resulting polygon will have at least 3 nodes
                        if len(l1.parts) > 1 or (ways[ l1.parts[l1.start][0] ].n + ways[mId].n) > 4:
                            l1.close(mId, start, end)
                            polygons.append(l1)
                    else:
                        # event: connect two different linestrings <l1> and <l2> with the <way>
                        connectToL1 = len(l1.parts) >= len(l2.parts)
                        if connectToL1:
                            extendedStart = l1.extend(mId, start, end, True)
                            l1.connect(l2, extendedStart, osm)
                            linestrings[l1.start if extendedStart else l1.end] = l1
                        else:
                            extendedStart = l2.extend(mId, start, end, False)
                            l2.connect(l1, extendedStart, osm)
                            linestrings[l2.start if extendedStart else l2.end] = l2
                    del linestrings[start], linestrings[end]
                elif start in linestrings:
                    # event: extend the linestring with the <way>
                    l = linestrings[start]
                    l.extend(mId, start, end, True)
                    del linestrings[start]
                    linestrings[end] = l
                elif end in linestrings:
                    # event: extend the linestring with the <way>
                    l = linestrings[end]
                    l.extend(mId, start, end, False)
                    del linestrings[end]
                    linestrings[start] = l
                else:
                    l = Linestring(mRole, mId, start, end)
                    # create entries in <linestrings> for the new linestring
                    linestrings[start] = l
                    linestrings[end] = l
        
        acceptBroken = self.m and self.m.acceptBroken
            
        if polygons and not linestrings:
            if len(polygons) == 1:
                self.t = Renderer.polygon
                # the only linestring is the valid polygon
                self.l = polygons[0]
            else:
                self.t = Renderer.multipolygon
                # all linestrings are valid polygon
                self.l = polygons
            # update bounds of the OSM data with the valid elements of the relation
            if not osm.projection:
                for p in polygons:
                    parts = p.parts
                    if parts:
                        for _ in parts:
                            way = osm.ways[ parts[_][0] ]
                            osm.updateBounds(way)
                    else:
                        way = osm.ways[p.wayId]
                        osm.updateBounds(way)
        elif acceptBroken and linestrings:
            # The number of entries in <linestrings> is divisible by two,
            # so the condition <len(linestrings) == 2> actually means the only broken linestring
            if not polygons and len(linestrings) == 2:
                self.t = Renderer.linestring
                self.l = next( iter(linestrings.values()) )
            else:
                self.t = Renderer.multilinestring
                l = polygons
                # Each linestring is stored twice in <linestrings> for its start and end,
                # so use Python set <nodeIds> to mark ids of OSM nodes that must be skipped
                # in the following cycle
                nodeIds = set()
                for nodeId in linestrings:
                    if nodeId in nodeIds:
                        continue
                    else:
                        linestring = linestrings[nodeId]
                        # add to <nodeIds> the id of the opposite open OSM node of <linestring>
                        nodeIds.add(linestring.end if nodeId == linestring.start else linestring.start)
                        l.append(linestring)
                self.l = l
        else:
            self.valid = False
    
    def getData(self, osm):
        """
        Get projected data for the relation if it is composed of the only linestring
        """
        return self.getLinestringData(self.l, osm)
    
    def getDataMulti(self, osm):
        """
        Get projected data for the relation if it is composed of several linestrings
        """
        return [self.getLinestringData(_l, osm) for _l in self.l]
    
    def getLinestringData(self, linestring, osm):
        return [osm.nodes[nodeId].getData(osm) for nodeId in linestring.nodeIds(osm)]
    
    def getOuterData(self, osm):
        """
        Get projected data for the outer polygon of the relation
        """
        # the method is applicable only for <self.t is Render.multipolygon>
        
        # iterate through the linestrings in the list <self.l>
        for _l in self.l:
            if _l.role is Osm.outer:
                break
        else:
            return
        return self.getLinestringData(_l, osm)
    
    def isClosed(self, linestringIndex=None):
        """
        Checks if the linestring <self.l[linestringIndex]> is closed
        """
        # the only linestring can't be closed, otherwise we would have a polygon
        if linestringIndex is None:
            return False
        else:
            l = self.l[linestringIndex]
            return not l.parts or self.l[linestringIndex].end is None
    
    def nodeIds(self, osm):
        """
        A generator to get id of OSM nodes of all linestrings of the relation
        """
        l = self.l
        if isinstance(l, list):
            # iterate through the linestrings in the list <l>
            for _l in l:
                for nodeId in _l.nodeIds(osm):
                    yield nodeId
        else:
            for nodeId in l.nodeIds(osm):
                yield nodeId