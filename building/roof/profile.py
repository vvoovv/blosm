"""
This file is part of blender-osm (OpenStreetMap importer for Blender).
Copyright (C) 2014-2017 Vladimir Elistratov
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

import math
from mathutils import Vector
from . import Roof
from util import zero
from util.osm import parseNumber


# Use https://raw.githubusercontent.com/wiki/vvoovv/blender-osm/assets/roof_profiles.blend
# to generate values for a specific profile
gabledRoof = (
    (
        (0., 0.),
        (0.5, 1.),
        (1., 0.)
    ),
    {
        "numSamples": 10,
        "angleToHeight": 0.5
    }
)

roundRoof = (
    (
        (0., 0.),
        (0.01, 0.195),
        (0.038, 0.383),
        (0.084, 0.556),
        (0.146, 0.707),
        (0.222, 0.831),
        (0.309, 0.924),
        (0.402, 0.981),
        (0.5, 1.),
        (0.598, 0.981),
        (0.691, 0.924),
        (0.778, 0.831),
        (0.854, 0.707),
        (0.916, 0.556),
        (0.962, 0.383),
        (0.99, 0.195),
        (1., 0.)
    ),
    {
        "numSamples": 1000,
        "angleToHeight": None
    }
)

gambrelRoof = (
    (
        (0., 0.),
        (0.2, 0.6),
        (0.5, 1.),
        (0.8, 0.6),
        (1., 0.)
    ),
    {
        "numSamples": 10,
        "angleToHeight": None
    }
)

saltboxRoof = (
    (
        (0., 0.),
        (0.35, 1.),
        (0.65, 1.),
        (1., 0.)
    ),
    {
        "numSamples": 100,
        "angleToHeight": 0.35
    }
)


class ProfiledVert:
    """
    A class represents a vertex belonging to RoofProfile.polygon projected on the profile
    """
    def __init__(self, roof, i, roofMinHeight, noWalls):
        """
        Args:
            roof (RoofProfile): an instance of the class <RoofProfile>
            i (int): index (between 0 and <roof.polygon.n-1>) of the polygon vertex
            roofMinHeight (float): Supplied by BuildingRenderer.renderElement(..)
            noWalls (bool): Does the building has wallls?
        """
        self.i = i
        verts = roof.verts
        indices = roof.polygon.indices
        proj = roof.projections
        p = roof.profile
        d = roof.direction
        v = verts[indices[i]]
        # is the polygon vertex <i> located on a profile slot?
        onProfile = False
        createVert = True
        # X-coordinate in the profile coordinate system (i.e. along the roof direction);
        # the roof direction is calculated in roof.processDirection(..);
        # the coordinate can possess the value between 0. and 1.
        x = (proj[i] - proj[roof.minProjIndex]) / roof.polygonWidth
        # Y-coordinate in the profile coordinate system;
        # it's a coordinate (with an offset) across roof profile.
        # Note, that a perpendicular to <roof.direction> is equal to <Vector((-d[1], d[0], 0.))
        self.y = -v[0]*d[1] + v[1]*d[0]
        # index in <roof.profileQ>
        index = roof.profileQ[
            math.floor(x * roof.numSamples)
        ]
        distance = x - p[index][0]
        
        if distance < zero:
            # the vertex <i> is located on the profile slot <roof.slots[index]>
            onProfile = True
            if roof.lEndZero and noWalls and not index:
                # The vertex <i> is located on the first profile slot <roof.slots[0]>,
                # also the building doesn't have walls and the profile value is equal to zero.
                # Therefore, no need to create a vertex, just use the polygon vertex with the index <indices[i]>
                createVert = False
                index = 0
                x = 0.
                vertIndex = indices[i]
            else:
                # <x> and <z> coordinates for the profile point with the <index>
                x, h = p[index]
        elif abs(p[index + 1][0] - x) < zero:
            # the vertex <i> is located on the profile slot <roof.slots[index+1]>
            onProfile = True
            # increase <index> by one
            index += 1
            if roof.rEndZero and noWalls and index == roof.lastProfileIndex:
                # The vertex <i> is located on the last profile slot <roof.slots[-1]>,
                # also the building doesn't have walls and the profile value is equal to zero.
                # Therefore, no need to create a vertex, just use the polygon vertex with the index <indices[i]>
                createVert = False
                index = roof.lastProfileIndex
                x = 1.
                vertIndex = indices[i]
            else:
                # <x> and <z> coordinates for the profile point with the <index>
                x, h = p[index]
        else:
            # Polygon vertex <i> has X-coordinate in the profile coordinate system,
            # located between the profile slots <roof.slots[index]> and <roof.slots[index+1]>
            # Z-coordinate for the profile point with the <index>
            h1 = p[index][1]
            # Z-coordinate for the profile point with the <index+1>
            h2 = p[index+1][1]
            # given <h1> and <h2>, calculate Z-coordinate for the polygon vertex <i>
            h = h1 + (h2 - h1) / (p[index+1][0] - p[index][0]) * distance
        if createVert:
            vertIndex = len(verts)
            # note, that <h> is multiplied by the roof height <roof.h>
            verts.append(Vector((v.x, v.y, roofMinHeight + roof.h * h)))
        # The meaning of <self.index> is that the polygon vertex <i> projected on the profile
        # has X-coordinate in the profile coordinate system,
        # located between the profile slots <roof.slots[index]> and <roof.slots[index+1]>
        self.index = index
        # If the polygon vertex <i> is located on a profile slot (i.e. <self.onProfile> is <True>) and
        # has <self.index>, it can be located only on the slot <roof.slots[index]>,
        # not (!) on the slot <roof.slots[index+1]>
        self.onProfile = onProfile
        # X-coordinate in the profile coordinate system, it can possess the value between 0. and 1.
        self.x = x
        # vertex index of the polygon vertex <i> projected on the profile
        self.vertIndex = vertIndex


class Slot:
    """
    An instance of the class is created for each profile point.
    The class is used to form faces for the profiled roof.
    
    See https://github.com/vvoovv/blender-osm/wiki/Profiled-roofs for description and illustration
    of concepts and algorithms used in the code. Specifically, the image <Main> from that webpage is
    used a number of times to illustrate the code.
    """
    
    def __init__(self):
        # Each element of <self.parts> is a Python tuple:
        # (y, part, reflection, index in <self.parts>)
        # A part is sequence of vertices that start at a slot and ends at the same slot or at a neighboring one.
        # Y-coordinate of the first vertex of the part in the coordinate system of the profile is
        # stored for each part.
        # Example of parts on the image <Main> for slots[2]:
        #     [16, 17]: starts at slots[2], ends at slots[3]
        #     [20, 2, 21]: starts at slots[2], ends at slots[2]
        #     [21, 3, 22]: starts at slots[2], ends at slots[2]
        #     [25, 26]: starts at slots[2], ends at slots[1]
        self.parts = []
        # <self.partsR> is used to store incomplete roof faces started in <self.trackUp(..)>
        # Each of those incomplete roof faces is completed in subsequent <self.trackDown(..)>
        self.partsR = []
        # <self.endAtSelf[index]> indicates if a <part> from <self.parts> with <part[3] == index>
        # ends at self (True) or at neighbor slot (False).
        # We use a separate Python list for <self.endAtSelf> to get some memory usage gain.
        # We could use <self.parts> to store that information, however in that case we would need
        # to use a Python list instead of Python tuple for each element of <self.parts>,
        # since the information where a part ends is set post factum, not during the creation of
        # an element of <self.parts>
        self.endAtSelf = []
    
    def reset(self):
        self.parts.clear()
        self.partsR.clear()
        self.endAtSelf.clear()
        # the current index in <self.parts>
        self.index = 0
    
    def prepare(self):
        """
        Prepare the slot to form roof faces.
        
        Specifically, sort <self.parts> by <y> coordinate in the profile coordinate system 
        """
        # remember that <y == part[0]>, where <part> is an element of <self.parts>
        self.parts.sort(key = lambda p: p[0])
    
    def append(self, vertIndex, y=None, originSlot=None, reflection=None):
        """
        Append <vertIndex> to the slot.
        
        If <y>, <originSlot> and <reflection> are given, create a new part starting with <vertIndex> and
        its <y> coordinate in the profile coordinate system.
        Otherwise append <vertIndex> to the last part of the slot (i.e. to <self.parts[-1]>)
        
        Args:
            vertIndex (int): Vertex index to append to the slot
            y (float): Y-coordinate in the profile coordinate system of the vertex with the index <vertIndex>
                wherer the new part starts
            originSlot (Slot): The last part of <originSlot> (i.e. <originSlot.parts[-1]>) ends at <self>.
                We need <originSlot> here to set <originSlot.endAtSelf> for the last part of <originSlot>.
                So after the execution of the method, <originSlot.endAtSelf[-1]> will correspond to
                <originSlot.parts[-1]>
            reflection (bool or None): Is there a reflection at <vertIndex>>
                <reflection> can have 3 values:
                <None>: no reflection
                <True>: reflection to the right (the example is give on the image <Main>; there is reflection
                    at the vertex <5> on the slot <slots[3]>)
                <False>: reflection to the left
        """
        parts = self.parts
        if y is None:
            # append <vertIndex> to the last part of the slot
            parts[-1][1].append(vertIndex)
        else:
            # create a new part starting with <vertIndex> and its <y> coordinate in the profile coordinate system
            parts.append((y, [vertIndex], reflection, self.index))
            # set <originSlot.endAtSelf> for the last part of <originSlot>
            originSlot.endAtSelf.append(originSlot is self)
            self.index += 1
    
    def trackDown(self, roofIndices, index=None, destVertIndex=None):
        """
        Track the slot downwards to form roof faces.
        
        Args:
            roofIndices (list): A Python list for all roof faces, so each entry of <roofIndices> is
                a Python list of vertex indices defining a single roof face
            index (int or None): If <index> is not <None>, it indicates a part in <self.parts> where to begin,
                otherwise start from the very top of the slot
            destVertIndex (int or None): If <index> is not <None>, <destVertIndex> indicates where to stop.
                There is no example on the image <Main>. However the idea is completely similar to
                the related example in <self.trackUp(..)>
        """
        parts = self.parts
        # Incomplete roof faces for that slot are stored in <self.n.partsR>.
        # <indexPartR> specifies the index of the element in <self.n.partsR>
        # to complete the current part.
        indexPartR = -1
        # We start from the very top if <index> is <None>.
        # The example of that case on the image <Main> for <slots[1]>:
        # <len(part) == 4>
        # <index == 2>
        index = (len(parts) if index is None else index) - 2
        # <vertIndex0> is set to the first vertex index of the next part, i.e. to <parts[index+1][1][0]>
        vertIndex0 = None
        while index >= 0:
            # the current part
            _, part, reflection, _index = parts[index]
            if vertIndex0 is None:
                # set <vertIndex0> to the first vertex index of the next part
                vertIndex0 = parts[index+1][1][0]
                # start a new roof face
                roofFace = []
            # <False> for the reflection means reflection to the left
            if reflection is False:
                # No example for that case on the image <Main>. However the idea is completely similar to
                # the related example in <self.trackUp(..)>.
                index -= 1
                continue
            # extend <roofFace> with vertex indices from <part>
            roofFace.extend(part)
            if part[-1] == vertIndex0:
                # Сame up
                # The example of that case on the image <Main> for <slots[3]>:
                # <index == 0>
                # <part == [17, 13, 14, 18]>
                # <vertIndex0 == 18>
                # The roof face is completed
                roofIndices.append(roofFace)
                # Setting <vertIndex0> to <None> means that we start a new roof face in
                # the next iteration of the <while> cycle
                vertIndex0 = None
            elif not self.endAtSelf[_index]:
                # Came to the neighbor from the right.
                # The example of that case on the image <Main> for <slots[1]>:
                # <index == 0>
                # <part == [15, 16]>
                # <roofFace == [27, 9, 10, 28, 15, 16]>
                # <self.n == slots[2]>
                # <self.n.partsR == [[19, 0, 1, 20, 25, 26]]>
                # Complete <roofFace> with the incomplete roof face <self.n.partsR[indexPartR]>
                roofFace.extend(self.n.partsR[indexPartR])
                # Change <indexPartR> to use the next incomplete roof face of <self.n.partsR>
                # in the next iterations of the <while> cycle
                indexPartR -= 1
                # The roof face is complete
                roofIndices.append(roofFace)
                # Setting <vertIndex0> to <None> means that we start a new roof face in
                # the next iteration of the <while> cycle
                vertIndex0 = None
            elif part[-1] != parts[index-1][1][0]:
                # No example for that case on the image <Main>. However the idea is completely similar to
                # the related example in <self.trackUp(..)>.
                # Basically, that case means that there is an island between
                # the vertices <part[0]> and <part[-1]>
                # We need to process that island in line below
                index = self.trackDown(roofIndices, index, part[-1])
            if not destVertIndex is None and parts[index-1][1][0] == destVertIndex:
                # No example for that case on the image <Main>. However the idea is completely similar to
                # the related example in <self.trackUp(..)>.
                return index
            # Proceed to the previous part downwards by decreasing <index>.
            # <True> for the reflection means reflection to the right
            # The example of case with <reflection is True> on the image <Main> for <slots[3]>:
            # <reflection is True>
            # <index == 3>
            # <part == [5, 6, 24]>
            # For that case in the next iteration of the <while> cycle
            # we proceed to <index == 2> and <part == [23, 4, 5]>,instead of <index == 1>,
            # where there is no part for tracking it downwards.
            index -= 1 if reflection is True else 2

    def trackUp(self, roofIndices, index=None, destVertIndex=None):
        """
        Track the slot upwards to form roof faces.
        
        Args:
            roofIndices (list): A Python list for all roof faces, so each entry of <roofIndices> is
                a Python list of vertex indices defining a single roof face
            index (int or None): If <index> is not <None>, it indicates a part in <self.parts> where to begin,
                otherwise start from the very bottom of the slot
            destVertIndex (int or None): If <index> is not <None>, <destVertIndex> indicates where to stop.
                See the example on the image <Main> for the slot <slots[2]>.
                <self.trackUp(..)> is called with <index == 1>, <destVertIndex == 20>.
                The example is elaborated further in the code below.
        """
        parts = self.parts
        numParts = len(parts)
        # Continue the example with the input attribute <index == 1>.
        # <index> will be set to <3> in the line below if <index == 1>.
        # Otherwise we start from the very bottom of the slots and set <index> to <1>
        index = 1 if index is None else index+2
        # <vertIndex0> is set to the first vertex index of the previous part, i.e. to <parts[index-1][1][0]>
        vertIndex0 = None
        while index < numParts:
            # the current part
            _, part, reflection, _index = parts[index]
            if vertIndex0 is None:
                # set <vertIndex0> to the first vertex index of the previous part
                vertIndex0 = parts[index-1][1][0]
                # start a new roof face
                roofFace = []
            # <True> for the reflection means reflection to the right
            if reflection is True:
                # The example of that case on the image <Main>:
                # there is reflection at the vertex <5> on the slot <slots[3]>
                # <index == 3>
                # We increase <index> by <1> in the line below; so we will come to the part [24, 25]
                # in the next iteration of the <while> cycle
                index += 1
                continue
            # extend <roofFace> with vertex indices from <part>
            roofFace.extend(part)
            if part[-1] == vertIndex0:
                # Сame down
                # The example of that case on the image <Main> for <slots[1]>:
                # <index == 1>
                # <part == [28, 11, 12, 15]>
                # <vertIndex0 == 15>
                # The roof face is complete
                roofIndices.append(roofFace)
                # Setting <vertIndex0> to <None> means that we start a new roof face in
                # the next iteration of the <while> cycle
                vertIndex0 = None
            elif not self.endAtSelf[_index]:
                # Came to the neighbor from the left.
                # The example of that case on the image <Main> for <slots[2]>:
                # <index == 5>
                # <part == [25, 26]>
                # <roofFace == [19, 0, 1, 20, 25, 26]>
                # Store the incomplete roof face <roofFace> in <self.partsR>. It will be completed in
                # the subsequent call of <self.trackDown(..)>
                self.partsR.append(roofFace)
                # Setting <vertIndex0> to <None> means that we start a new roof face in
                # the next iteration of the <while> cycle
                vertIndex0 = None
            elif part[-1] != parts[index+1][1][0]:
                # The example of that case on the image <Main> for <slots[2]>:
                # <index == 1>
                # <part == [19, 0, 1, 20]>
                # <part[-1] == 20>
                # <parts[index+1][1] == [22, 3, 21]>
                # <parts[index+1][1][0] == 22>
                # Basically, that case means that there is an island between the vertices <19> and <20>
                # We need to process that island in line below
                index = self.trackUp(roofIndices, index, part[-1])
            if not destVertIndex is None and parts[index+1][1][0] == destVertIndex:
                # The example of that case on the image <Main> for <slots[2]>:
                # <destVertIndex == 20>
                # <index == 3>
                # <part == [21, 3, 22]>
                # <parts[index+1][1] == [20, 2, 21]>
                # <parts[index+1][1][0] == 20>
                return index
            # Proceed to the next part upwards by increasing <index>.
            # <False> for the reflection means reflection to the left
            index += 1 if reflection is False else 2
    
    def processWallFace(self, indices, pv1, pv2):
        """
        A child class may provide realization for this methods
        
        Args:
            indices (list): Vertex indices for the wall face
            pv1 (ProfiledVert): the first profiled vertex of the two between which the slot vertex is located
            pv2 (ProfiledVert): the second profiled vertex of the two between which the slot vertex is located
        """
        pass


class RoofProfile(Roof):
    """
    The class deals with so called profiled roofs (i.e. roofs defined be a profile):
    gabled, round, gambrel, saltbox
    
    See https://github.com/vvoovv/blender-osm/wiki/Profiled-roofs for description and illustration
    of concepts and algorithms used in the code. Specifically, the image <Main> from that webpage is
    used a number of times to illustrate the code.
    """
    
    defaultHeight = 3.
    
    def __init__(self, data):
        """
        Args:
            data (tuple): profile values and some attributes to define a profiled roof,
                e.g. gabledRoof, roundRoof, gambrelRoof, saltboxRoof
        """
        super().__init__()
        self.hasRidge = True
        self.projections = []
        
        # actual profile values as a Python tuple of (x, y)
        profile = data[0]
        self.profile = profile
        numProfilesPoints = len(profile)
        self.lastProfileIndex = numProfilesPoints - 1
        # create profile slots
        slots = tuple(Slot() for i in range(numProfilesPoints) )
        # set the next slot, it will be need in further calculations
        for i in range(self.lastProfileIndex):
            slots[i].n = slots[i+1]
        self.slots = slots
        
        for attr in data[1]:
            setattr(self, attr, data[1][attr])
        
        # is the y-coordinate at <x=0.0> (the left end of the profile) is equal to zero?
        self.lEndZero = not profile[0][1]
        # is the y-coordinate at <x=1.0> (the right end of the profile) is equal to zero?
        self.rEndZero = not profile[-1][1]
        
        # Quantize <profile> with <numSamples> to get performance gain
        # Quantization is needed to perform the following action very fast.
        # Given x-coordinate <x> between 0. and 1. in the profile coordinate system,
        # find two neighboring slots between which <x> is located
        _profile = tuple(math.ceil(p[0]*self.numSamples) for p in profile)
        profileQ = []
        index = 0
        for i in range(self.numSamples):
            if i >= _profile[index+1]:
                index += 1  
            profileQ.append(index)
        profileQ.append(index)
        self.profileQ = profileQ

    def init(self, element, data, minHeight, osm):
        super().init(element, data, minHeight, osm)
        self.initProfile()
    
    def initProfile(self):
        self.projections.clear()
        # The last slot with the index <self.lastProfileIndex> isn't touched,
        # so no need to reset it
        for i in range(self.lastProfileIndex):
            self.slots[i].reset()
    
    def make(self, bldgMaxHeight, roofMinHeight, bldgMinHeight, osm):
        polygon = self.polygon
        roofIndices = self.roofIndices
        noWalls = bldgMinHeight is None
        slots = self.slots
        # the current slot: start from the leftmost slot
        self.slot = slots[0]
        # the slot from which the last part in <slot.parts> originates
        self.originSlot = slots[0]
        
        if not self.projections:
            self.processDirection()
        
        # Start with the vertex from <polygon> with <x=0.> in the profile coordinate system;
        # the variable <i0> is needed to break the cycle below
        i = i0 = self.minProjIndex
        # Create a profiled vertex out of <self.verts[polygon.indices[i]]>;
        # <pv> stands for profiled vertex
        pv1 = pv0 = ProfiledVert(self, i, roofMinHeight, noWalls)
        _pv = None
        while True:
            i = polygon.next(i)
            if i == i0:
                # came to the starting vertex, so break the cycle
                break
            # create a profiled vertex out of <self.verts[polygon.indices[i]]>
            pv2 = ProfiledVert(self, i, roofMinHeight, noWalls)
            # The order of profiled vertices is <_pv>, <pv1>, <pv2>
            # Create in-between vertices located on the slots for the segment between <pv1> and <pv2>),
            # also form a wall face under the segment between <pv1> and <pv2>
            self.createProfileVertices(pv1, pv2, _pv, roofMinHeight, noWalls)
            _pv = pv1     
            pv1 = pv2
        # Create in-between vertices located on the slots for the closing segment between <pv1> and <pv0>,
        # also form a wall face under the segment between <pv1> and <pv0>
        self.createProfileVertices(pv1, pv0, _pv, roofMinHeight, noWalls)
        
        # Append the vertices from the first part of the <slot[0]> (i.e. slots[0].parts[0])
        # to the last part of <slots[1]> (i.e. slots[1].parts[-1])
        # Example on the image <Main>:
        # slots[0].parts[0][1] is [11, 12, 15]
        # slots[1].parts[-1][1] is [28, 11]
        # after the execution of the line of below:
        # slots[1].parts[-1][1] is [28, 11, 12, 15]
        slots[1].parts[-1][1].extend(slots[0].parts[0][1][i] for i in range(1, len(slots[0].parts[0][1])))
        # the last part of <slots[1]> ends at self
        slots[1].endAtSelf.append(True)
        
        # Prepare the slots to form roof faces;
        # note that slots[0] and slots[-1] aren't used in the calculations
        for i in range(1, self.lastProfileIndex):
            slots[i].prepare()
        
        # Below is the cycle to form roof faces
        # Each time a band between the neighboring slots
        # <slotL> (a slot from the left) and <slotR> (a slot from the right) is considered.
        # We track <slotR> upwards by executing <slotR.trackUp(roofIndices)>,
        # then we track <slotL> downwards by executing slotL.trackDown(roofIndices)
        slotR = slots[1]
        slotR.trackUp(roofIndices)
        for i in range(1, self.lastProfileIndex):
            slotL = slotR
            slotR = slots[i+1]
            slotR.trackUp(roofIndices)
            slotL.trackDown(roofIndices)
        return True
    
    def createProfileVertices(self, pv1, pv2, _pv, roofMinHeight, noWalls):
        """
        Create in-between vertices located on the slots for the segment between <pv1> and <pv2>,
        also form a wall face under the segment between <pv1> and <pv2>.
        
        For example (see the image <Main>), if <pv1> was created for the polygon vertex with the index <3>,
        <pv2> was created for the polygon vertex with the index <4>, then two vertices with the indices
        <22> and <23> will be created for the intersection of the segment between <pv1> and <pv2> with
        the slots <slots[2]> and <slots[3]> respectively.
         
        Args:
            pv1 (ProfiledVert): Defines the first vertex of the segment of <self.polygon> projected on the profile
            pv2 (ProfiledVert): Defines the second vertex of the segment of <self.polygon> projected on the profile
            _pv (ProfiledVert): Precedes <pv1>
            roofMinHeight (float): Supplied by BuildingRenderer.renderElement(..)
            noWalls (bool): Does the building has wallls?
        """
        verts = self.verts
        indices = self.polygon.indices
        p = self.profile
        wallIndices = self.wallIndices
        slots = self.slots
        # the current slot
        slot = self.slot
        
        # index of the slot for <pv1>
        index1 = pv1.index
        # index of the slot for <pv1>
        index2 = pv2.index
        
        # skip the polygon vertex with the index indices[pv1.i] from including it to the wall face?
        skip1 = noWalls and pv1.onProfile and\
            ((self.lEndZero and not index1) or\
            (self.rEndZero and index1 == self.lastProfileIndex))
        # skip the polygon vertex with the index indices[pv2.i] from including it to the wall face?
        skip2 = noWalls and pv2.onProfile and\
            ((self.lEndZero and not index2) or\
            (self.rEndZero and index2 == self.lastProfileIndex))

        if skip1 and skip2 and index1 == index2:
            # In the case the building doesn't have walls and both <pv1> and <pv2>
            # are located either on <slots[0]> or <slots[-1]>
            if _pv is None:
                # We are at <slots[0]> and just started, so create the very first part for the <slot>;
                # <slot> is <slots[0]>
                slot.append(pv1.vertIndex, pv1.y, self.originSlot)
            # append <pv2.vertIndex> to the last part of the <slot> (i.e. to <slot.parts[-1]>)
            slot.append(pv2.vertIndex)
            # we are done
            return
        # start a wall face under the segment between <pv1> and <pv2>
        _wallIndices = [pv1.vertIndex]
        if not skip1:
            _wallIndices.append(indices[pv1.i])
        if not skip2:
            _wallIndices.append(indices[pv2.i])
        _wallIndices.append(pv2.vertIndex)
        
        # polygon vertices
        v1 = verts[indices[pv1.i]]
        v2 = verts[indices[pv2.i]]
        if not _pv is None:
            _v = verts[indices[_pv.i]]
        
        if _pv is None:
            # We are at <slots[0]> and just started, so create the very first part for the <slot>;
            # <slot> is <slots[0]>
            slot.append(pv1.vertIndex, pv1.y, self.originSlot)
        elif pv1.onProfile:
            # <pv1> is located on a profile slot
            
            # <reflection> can have 3 values:
            # <None>: no reflection
            # <True>: reflection to the right (see below an example)
            # <False>: reflection to the left
            reflection = None
            # If <appendToSlot> is <True> we change the current slot and create a new part for that slot.
            # If <appendToSlot> is <False> we continue with the <slot> and its last part <slot.parts[-1]>.
            appendToSlot = False
            if pv2.onProfile and index1 == index2:
                # <pv2> is located on the same profile slot as <pv1>
                # Example on the image <Main>:
                # <_pv> ~ <6>
                # <pv1> ~ <7>
                # <pv2> ~ <8>
                if (index1 != self.lastProfileIndex and _pv.x < pv1.x and pv1.y > pv2.y)\
                    or (index1 and _pv.x > pv1.x and pv1.y < pv2.y):
                    # The conditions <index1 != self.lastProfileIndex> and <index1>
                    # are to prevent erroneous behavior due to 180 degrees angle as the result of mapping error or
                    # precision error caused by the nature of <zero> variable
                    
                    # <6>, <7>, <8> on the image <Main> doesn't satisfy that condition
                    appendToSlot = True
            elif pv1.x < pv2.x:
                # going from the left to the right
                if _pv.x < pv1.x:
                    appendToSlot = True
                elif index1: # i.e. index1 != 0
                    # The condition <index1> is to prevent
                    # erroneous reflection due to 180 degrees angle as the result of mapping error or
                    # precision error caused by the nature of <zero> variable
                    if _pv.onProfile and _pv.index == pv1.index:
                        # <_pv> is located on the same profile slot as <pv1>
                        if _pv.y < pv1.y:
                            appendToSlot = True
                            # no reflection in this case!
                    elif (pv2.x-pv1.x)*(_pv.y-pv1.y) - (pv2.y-pv1.y)*(_pv.x-pv1.x) < 0.:
                        # <_pv.x > pv1.x> and <pv1.x < pv2.x>
                        appendToSlot = True
                        # <True> for the reflection means reflection to the right
                        reflection = True
                        # Example of a reflection to the right on the image <Main>:
                        # <_pv> ~ <4>
                        # <pv1> ~ <6>
                        # <pv2> ~ <6>
            else:
                # going from the right to the left
                if _pv.x > pv1.x:
                    appendToSlot = True
                elif index1 != self.lastProfileIndex:
                    # The condition <index1 != self.lastProfileIndex> is to prevent
                    # erroneous reflection due to 180 degrees angle as the result of mapping error or
                    # precision error caused by the nature of <zero> variable
                    if _pv.onProfile and _pv.index == pv1.index:
                        # <_pv> is located on the same profile slot as <pv1>
                        if _pv.y > pv1.y:
                            appendToSlot = True
                            # no reflection in this case!
                    elif (pv2.x-pv1.x)*(_pv.y-pv1.y) - (pv2.y-pv1.y)*(_pv.x-pv1.x) < 0.:
                        # <_pv.x < pv1.x> and <pv1.x > pv2.x>
                        appendToSlot = True
                        # <False> for the reflection means reflection to the left
                        reflection = False
                        # No example of a refelection to the left on the image <Main>,
                        # the example above with the reflection to right should give
                        # understanding what a reflection is about 
            if appendToSlot:
                # change the current slot and <self.originSlot>
                self.originSlot = slot
                slot = slots[index1]
                # Create a new part for the new slot
                # Note that the last part of <self.originSlot> (i.e. <self.originSlot.parts[-1]>)
                # ends at the new current <slot>
                slot.append(pv1.vertIndex, pv1.y, self.originSlot, reflection)
        
        def common_code(slot, vertsRange, slotRange):
            """
            A helper function
            
            Actually create in-between vertices located on the slots for the segment between <pv1> and <pv2>,
            add the indices of the newly created vertices to the wall face <_wallIndices>.
            Also append those indices to the related slots and change the current slot in the cycle
            
            Args:
                slot (Slot): the current slot
                vertsRange (range): range of <slots> to create in-between vertices
                slotRange (range): range of <slots> to append the indices of the newly created vertices
                    to the related slots
            """
            vertIndex = len(verts) - 1
            # <factorX> and <factorY> are used in calculations in the cycle below
            factorX = (v2.x - v1.x) / (pv2.x - pv1.x)
            factorY = (v2.y - v1.y) / (pv2.x - pv1.x)
            for _i in vertsRange:
                vertIndex += 1
                factor = p[_i][0] - pv1.x
                verts.append(Vector((
                    v1.x + factor * factorX,
                    v1.y + factor * factorY,
                    roofMinHeight + self.h * p[_i][1]
                )))
                _wallIndices.append(vertIndex)
            # Fill <slots>
            # <factor> is used to calculate Y-coordinate in the profile coordinate system
            factor = (pv2.y - pv1.y) / (pv2.x - pv1.x)
            for _i in slotRange:
                # append <vertIndex> to the last part of the current slot (i.e. to slot.parts[-1])
                slot.append(vertIndex)
                # change the current slot and <self.originSlot>
                self.originSlot = slot
                slot = slots[_i]
                # Create a new part for the new slot
                # Note that the last part of <self.originSlot> (i.e. <self.originSlot.parts[-1]>)
                # ends at the new current <slot>
                slot.append(vertIndex, pv1.y + factor * (p[_i][0] - pv1.x), self.originSlot)
                # Child classes of <Slot> may use the following function call <slot.processWallFace(..)>
                # to do some stuff
                slot.processWallFace(_wallIndices, pv1, pv2)
                vertIndex -= 1
            # return the current slot
            return slot
        
        if index1 != index2:
            if index2 > index1:
                # Going from the left to the right
                # If the condition below isn't valid, there is no need to call <common_code(..)>
                if not pv2.onProfile or index1 != index2-1:
                    slot = common_code(
                        slot,
                        range(index2-1 if pv2.onProfile else index2, index1, -1),
                        range(index1+1, index2 if pv2.onProfile else index2+1)
                    )
            else:
                # Going from the right to the left
                # If the condition below isn't valid, there is no need to call <common_code(..)>
                if not pv1.onProfile or index2 != index1-1:
                    slot = common_code(
                        slot,
                        range(index2+1, index1 if pv1.onProfile else index1+1),
                        range(index1-1 if pv1.onProfile else index1, index2, -1)
                    )
        # the wall face <_wallIndices> is ready, append it to <wallIndices>
        wallIndices.append(_wallIndices)
        # append <pv2.vertIndex> to the last part of the current slot (i.e. to <slot.parts[-1]>)
        slot.append(pv2.vertIndex)
        # remember the current slot
        self.slot = slot
    
    def getHeight(self, app):
        tags = self.element.tags
        
        h = parseNumber(tags["roof:height"]) if "roof:height" in tags else None
        if h is None:
            if not self.angleToHeight is None and "roof:angle" in tags:
                angle = parseNumber(tags["roof:angle"])
                if not angle is None:
                    self.processDirection()
                    h = self.angleToHeight * self.polygonWidth * math.tan(math.radians(angle))
            if h is None:
                # get the number of levels
                if "roof:levels" in tags:
                    h = parseNumber(tags["roof:levels"])
                h = self.defaultHeight if h is None else h * app.levelHeight
        self.h = h
        return h