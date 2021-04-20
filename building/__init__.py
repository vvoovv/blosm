"""
This file is part of blender-osm (OpenStreetMap importer for Blender).
Copyright (C) 2014-2018 Vladimir Elistratov
prokitektura+support@gmail.com

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

from numpy import zeros
import parse
from mathutils import Vector
from util.polygon import Polygon

#
# values for <BldgVector.skip>
#
# Both edges attached to a node in question do not have a shared building
NoSharedBldg = 1
# Both edges attached to a node in question do not have a shared building
SharedBldgBothEdges = 2


class BldgPolygon:
    
    __slots__ = ("vectors", "numEdges", "reversed", "refVector")
    
    def __init__(self, building, manager):
        self.reversed = False
        # If <self.refVector> is not None, it indicates that the polygon has at least one
        # straight angle. In that case is stores a reference vector WITHOUT a straight angle
        self.refVector = None
        # vectors
        self.vectors = vectors = tuple(
            self.getVector(nodeId1, nodeId2, manager) \
                for nodeId1,nodeId2 in building.outline.outerVectorNodeIds(manager.data) \
                    if not manager.data.haveSamePosition(nodeId1, nodeId2)
        )
        self.numEdges = len(self.vectors)
        # set the previous and the next vector for each vector from <self.vectors>
        for i in range(self.numEdges-1):
            vectors[i].prev = vectors[i-1]
            vectors[i].next = vectors[i+1]
            vectors[i].index = i
        vectors[-1].prev = vectors[i]
        vectors[-1].next = vectors[0]
        vectors[-1].index = self.numEdges-1
        
        self.forceCcwDirection()
        for vector in self.vectors:
            vector.markUsedNode(self, manager)

    def forceCcwDirection(self):
        """
        Check direction of the building outer polygon and
        force their direction to be counterclockwise
        """
        vectors = self.vectors
        # Get index of the vertex with the minimum Y-coordinate and maximum X-coordinate,
        # i.e. the index of the rightmost lowest vertex
        
        index = min(
            range(self.numEdges),
            key = lambda index: ( vectors[index].v1[1], -vectors[index].v1[0] )
        )
        
        # <vector1=vectors[index].prev>: the vector entering the vertex with the <index>
        # <vector2=vectors[index]>: the vector leaving the vertex with the <index>
        # Check if the <vector2> is to the left from the vector <vector1>;
        # in that case the direction of vertices is counterclockwise,
        # it's clockwise in the opposite case.
        if self.directionCondition(vectors[index].prev.vector, vectors[index].vector):
            self.reverse()

    def processStraightAngles(self, manager):
        """
        Given <verts> constituting a polygon, removes vertices forming a straight angle
        """
        hasStraightAngle = 0
        # <refVector> is used if <hasStraightAngle> to mark a vector without the straight angle
        refVector = None
        
        vec_ = self.vectors[0 if self.reversed else -1].vector
        for vector in (reversed(self.vectors) if self.reversed else self.vectors):
            vec = vector.vector
            dot = vec_[0]*vec[0] + vec_[1]*vec[1]
            if dot and abs( (vec_[0]*vec[1]-vec_[1]*vec[0])/dot ) < Polygon.straightAngleTan:
                # got a straight angle
                vector.straightAngle = True
                if len(manager.data.nodes[vector.id1].bldgs) == 1:
                    vector.skip = NoSharedBldg
                    if hasStraightAngle != NoSharedBldg:
                        # The value <NoSharedBldg> for <hasStraightAngle>
                        # takes precedence over <SharedBldgBothEdges>
                        hasStraightAngle = NoSharedBldg
                elif not hasStraightAngle:
                    hasStraightAngle = SharedBldgBothEdges
            elif not refVector:
                refVector = vector
            vec_ = vec
        
        if hasStraightAngle:
            # Set a reference vector without the straight angle.
            # It also indicates that the polygon has at least one straight angle
            self.refVector = refVector
        
        if hasStraightAngle == NoSharedBldg:
            # Skip only straight angles for the vectors with the value of the attribute <skip>
            # equal to <NoSharedBldg>
            self.skipStraightAngles(manager, NoSharedBldg)
    
    def processStraightAnglesExtra(self, manager):
        if self.refVector:
            for vector in (reversed(self.vectors) if self.reversed else self.vectors):
                if not vector.skip and vector.straightAngle:
                    # The condition <vector.neighbor.polygon is vector.prev.neighbor.polygon> means:
                    # check if the <vector> and its previous vector (<vector.prev>) share
                    # the same building polygon (BldgPolygon)
                    if vector.edge.hasSharedBuildings() and\
                            vector.prev.edge.hasSharedBuildings() and\
                            vector.neighbor.polygon is vector.prev.neighbor.polygon:
                        vector.skip = SharedBldgBothEdges
            self.skipStraightAngles(manager, SharedBldgBothEdges)
    
    def skipStraightAngles(self, manager, skipValue):
        refVector = self.refVector
        vector = prevNonStraightVector = refVector
        isPrevVectorStraight = False
        while True:
            if vector.skip == skipValue:
                self.numEdges -= 1
                isPrevVectorStraight = True
            else:
                if isPrevVectorStraight:
                    prevNonStraightVector.skipNodes(vector, manager)
                    isPrevVectorStraight = False
                # remember the last vector with a non straight angle
                prevNonStraightVector = vector
                
            vector = vector.next
            if vector is refVector:
                if isPrevVectorStraight:
                    prevNonStraightVector.skipNodes(vector, manager)
                break
    
    def directionCondition(self, vectorIn, vectorOut):
        return vectorIn[0] * vectorOut[1] - vectorIn[1] * vectorOut[0] < 0.
    
    def getVector(self, nodeId1, nodeId2, manager):
        edge = manager.getEdge(nodeId1, nodeId2)
        vector = BldgVector(edge, edge.id1 == nodeId1, self)
        edge.addVector(vector)
        return vector
    
    def reverse(self):
        self.reversed = True
        for vector in self.vectors:
            vector.reverse()
    
    @property
    def verts(self):
        return (
            vector.v1 for vector in (reversed(self.vectors) if self.reversed else self.vectors) if not vector.skip
        )
    
    def getEdges(self):
        return (vector.edge for vector in reversed(self.vectors) if not vector.skip) \
            if self.reversed else\
            (vector.edge for vector in self.vectors if not vector.skip)


class BldgEdge:
    
    __slots__ = ("id1", "v1", "id2", "v2", "visibility", "visibilityTmp", "vectors")
    
    def __init__(self, id1, v1, id2, v2):
        #
        # Important: always id1 < id2 
        #
        self.id1 = id1
        self.v1 = v1
        self.id2 = id2
        self.v2 = v2
        
        self.visibility = self.visibilityTmp = 0.
        # instances of the class <BldgVector> shared by the edge are stored in <self.vectorss>
        self.vectors = None
    
    def addVector(self, vector):
        if self.vectors:
            self.vectors = (self.vectors[0], vector)
        else:
            # a Python tuple with one element
            self.vectors = (vector,)
    
    def hasSharedBuildings(self):
        return len(self.vectors) == 2

    def updateVisibility(self):
        self.visibility = max(self.visibility, self.visibilityTmp)


class BldgVector:
    """
    A wrapper for the class BldgEdge
    """
    
    __slots__ = ("edge", "direct", "prev", "next", "polygon", "straightAngle", "skip", "index")
    
    def __init__(self, edge, direct, polygon):
        self.edge = edge
        # <self.direct> defines the direction given the <edge> defined by node1 and node2
        # True: the direction of the vector is from node1 to node2
        self.direct = direct
        self.polygon = polygon
        self.straightAngle = False
        self.skip = 0
    
    def reverse(self):
        self.direct = not self.direct
        self.prev, self.next = self.next, self.prev
    
    @property
    def v1(self):
        return self.edge.v1 if self.direct else self.edge.v2
    
    @property
    def v2(self):
        return self.edge.v2 if self.direct else self.edge.v1
    
    @property
    def id1(self):
        return self.edge.id1 if self.direct else self.edge.id2
    
    @property
    def id2(self):
        return self.edge.id2 if self.direct else self.edge.id1
    
    @property
    def neighbor(self):
        return self.edge.vectors[1] if (self.edge.vectors[0] is self) else self.edge.vectors[0]
    
    @property
    def vector(self):
        if self.direct:
            v1 = self.edge.v1
            v2 = self.edge.v2
        else:
            v1 = self.edge.v2
            v2 = self.edge.v1
        return (v2[0] - v1[0], v2[1] - v1[1])
    
    def skipNodes(self, nextVector, manager):
        """
        Skip nodes between <self> and <nextVector>
        Update <prev> and <next> attributes. Set a new <BldgEdge>
        """
        self.next = nextVector
        nextVector.prev = self
        # set an instance of <BldgEdge> between <self> and <nextVector>
        nodeId1 = self.id1
        nodeId2 = nextVector.id1
        self.edge = manager.getEdge(nodeId1, nodeId2)
        self.edge.addVector(self)
        self.direct = nodeId1 == self.edge.id1
    
    def markUsedNode(self, building, manager):
        manager.data.nodes[self.id1].bldgs.append(building)


class Building:
    """
    A wrapper for a OSM building
    """
    
    __slots__ = ("outline", "parts", "polygon", "auxIndex", "crossedEdges")
    
    def __init__(self, element, buildingIndex, osm):
        self.outline = element
        self.parts = []
        # a polygon for the outline, used for facade classification only
        self.polygon = None
        # an auxiliary variable used to store the first index of the building vertices in an external list or array
        self.auxIndex = 0
        # A dictionary with edge indices as keys and crossing ratio as value,
        # used for buildings that get crossed by way-segments.
        self.crossedEdges = []
        self.markUsedNodes(buildingIndex, osm)
    
    def initPolygon(self, manager):
        self.polygon = BldgPolygon(self, manager)
    
    def addPart(self, part):
        self.parts.append(part)

    def markUsedNodes(self, buildingIndex, osm):
        """
        For each OSM node of <self.element> (OSM way or OSM relation) add the related
        <buildingIndex> (i.e. the index of <self> in Python list <buildings> of an instance
        of <BuildingManager>) to Python set <b> of the node 
        """
        for nodeId in self.outline.nodeIds(osm):
            osm.nodes[nodeId].b[buildingIndex] = 1
    
    def resetTmpVisibility(self):
        for vector in self.polygon.vectors:
            vector.edge.visibilityTmp = 0.

    def resetCrossedEdges(self):
        self.crossedEdges.clear()

    def addCrossedEdge(self, edge, intsectX):
        self.crossedEdges.append( (edge, intsectX) )
    
    def edgeInfo(self, queryBldgVerts, firstVertIndex):
        """
        A generator that yields edge info (the first edge vertex, the second edge vertex)
        out of numpy array <queryBldgVerts> and the index of the first vertex <firstVertIndex> of
        the building polygon at <queryBldgVerts>
        """
        n_1 = self.polygon.numEdges - 1
        for vertIndex in range(firstVertIndex, firstVertIndex + n_1):
            yield queryBldgVerts[vertIndex], queryBldgVerts[vertIndex+1]
        # the last edge
        yield queryBldgVerts[firstVertIndex + n_1], queryBldgVerts[firstVertIndex]