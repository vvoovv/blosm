
"""
This file is part of blender-osm (OpenStreetMap importer for Blender).
this file: Copyright (C) 2018 Kirill Bondarenko aka Zkir, GNU GPL
contact Vladimir Elistratov prokitektura+support@gmail.com

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
from . import Roof

import bpy
import math
from math import cos
from math import sin
from math import pi
from mathutils import Vector
from util.polygon import PolygonOLD

# Utility functions
# dummy function to create vertex, wrapped into particular class required by blender-osm
def vert(x, y,z):
    """ Create a single vert """
    return Vector((x, y, z))
    #return (x, y, z)

"""
Profiles for several roof types.
"""
#generatrix for a pyramid
def pyramidal_profile(rows):
    profile=[]
    for j in range(rows+1):
        x = 1-j/rows
        z = 1-x
        #just equation of line.
        profile.append((x,z))
    return profile

#generatrix for a dome
#genratix for dome is circle equation in the parameterized form x=x(t), y=y(t)
def dome_profile(rows):
    profile=[]
    for j in range(rows+1):
        x = cos(j/rows*pi/2)
        z = sin(j/rows*pi/2)
        profile.append((x,z))
    return profile

#generatrix for an onion roof.
#there is no explicit formula for an onion, that's why we will just write it as a set of vertex cordinates. 
#Note, that there are actually more than 5 typical onion types even for orthodox churches only. Other forms will be added later.
# (r, z) 
onion_profile=(
(1.0000,	0.0000),
(1.2971,	0.0999),
(1.2971,	0.2462),
(1.1273,	0.3608),
(0.6219,	0.4785),
(0.2131,	0.5984),
(0.1003,	0.7243),
(0.0000,	1.0000)
)

#generatrix for a flat roof. not really used, for testing purposes only.
flat_profile=(
(1.00,0),
(0.75,0),
(0.50,0),
(0.25,0),
(0.00,0),
)



class RoofGeneratrix(Roof):
    """
    A Blender object to deal with buildings or building part with a pseudo-conical roof
    the roof mesh is created via directrix and generatrix.
    building outline is used as a directrix, and roof profiles defined above are used as a generatrix.
    practically any form can be created this way. dome, onion, pyramid
    see https://en.wikipedia.org/wiki/Generatrix 

    """
    
    defaultHeight = 4.

    def __init__(self, strRoofType):

        super().__init__()
        self.roofType=strRoofType 
 
        if strRoofType=="pyramidal":        
            self.roof_profile=pyramidal_profile(2)

        elif strRoofType=="dome":
            self.roof_profile=dome_profile (7)

        elif strRoofType=="half-dome":
            self.roof_profile=dome_profile (7)

        elif strRoofType=="onion":  
            self.roof_profile=onion_profile

        else:
            raise Exception("unknown roof profile: " + strRoofType)
        
    def make(self, osm):

        # Maybe it's not exactly the way vvoovv expected it to be, but we will do it this way:
        # We will start from the scratch!   
        verts = []
        roofFaces = [] #self.roofIndices 
        wallFaces = [] #self.wallIndices
        
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #  Над этим местом я сломал себе весь мозг.
        #  self.verts -- очень странный массив. В него мы добавляем созданную геометрию. 
        #  Я бы ожидал, что сюда придет либо пустое множество, либо контур основания здания, уже подготовленный(!).
        #  но проблема в том, что  self.verts содержит не только нужные вершины, но и те, 
        #  которые были отсеяны как лишние removeStraitAngles()
        #  т.е. он содержит как вершины входяшие в меш, так и те которые в него не входят!
        #  polygon.indices массив, который содержит номера "нужных" вершин.
        #  Что-то тут надо переделать, потому что так жить невозможно: тут играем, тут не играем, тут рыбу заворачивали. 
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!    
        # we will remove superfluous vertexes forever. According to my tests they are not really needed anyway.
        for i in range(self.polygon.n):
          verts.append(self.verts[self.polygon.indices[i]])  
        self.verts = verts

        self.polygon = PolygonOLD(verts)    

        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 

        n=len(verts)

        #z-coords. we will use the following model.
        z0 = verts[0][2]  # bottom of the building part. Should be zero if building stands on the ground.  self.z1 ???
        z1 = self.roofVerticalPosition  # roof cornice
        z2 = self.z2 # top of the roof.

        if self.roofType=="half-dome":
            # for half-dome we need another algorithm.
            center = self.getHalfDomeCenter()
        else: 
            center = self.getCenter()
        
         
        #number of face loops, which we will create
        rows=len(self.roof_profile)-1 # it is defined by the roof profile.
        noWalls = (z0 == z1) #building without walls, e.g. the Pyramid of Cheops
        
        #Let's create walls
        if not noWalls:
            for i in range(n): 
                verts.append(vert(verts[i][0],verts[i][1],z1)) 
            indexOffset = n   
            for i in range(n-1):
                wallFaces.append((i,i+1,i+n+1,i+n))
            wallFaces.append((n-1,0,n,2*n-1))    
        else:
            indexOffset = 0
               

        #lets create roof mesh, starting from vertexes
        for j in range(1,rows):
            for i in range(n):
                
                #координаты вершин внутренних колец (x,y)
                xi=verts[i][0] + (1-self.roof_profile[j][0])*(center[0]-verts[i][0])
                yi=verts[i][1] + (1-self.roof_profile[j][0])*(center[1]-verts[i][1])

                #Высота (z)
                zi=z1+(z2-z1)*self.roof_profile[j][1]
                verts.append(vert(xi,yi,zi))

        #let's add the top vertex
        verts.append(vert(center[0],center[1],z2))
        centreIdx=len(verts)-1

        #... and then faces
        for j in range(rows-1):
            for i in range(n-1):
                roofFaces.append((indexOffset+j*n+i,indexOffset+j*n+i+1,indexOffset+j*n+i+1+n,indexOffset+j*n+i+n))
            roofFaces.append((indexOffset+j*n+n-1,indexOffset+j*n+0,indexOffset+j*n+n,indexOffset+j*n+2*n-1))   

        #faces in the last loop are triangles.  
        for i in range(n-1):
            roofFaces.append((indexOffset+(rows-1)*n+i,indexOffset+(rows-1)*n+i+1,centreIdx))
        roofFaces.append((indexOffset+(rows-1)*n+n-1,indexOffset+(rows-1)*n+0,centreIdx))  
       

        #self.verts = verts
        self.roofIndices = roofFaces
        self.wallIndices = wallFaces

        return True

    # Centroid: we will just use center of the bounding box as a centroid
    # this alrogithm should give better results for convex n-gons than average arithmetic of all vertex coordinates 
    def getCenter(self):
        verts = self.verts
        n=len(verts)
        minX=verts[0][0]
        maxX=verts[0][0]
        minY=verts[0][1]
        maxY=verts[0][1]
        for i in range(n):
            if verts[i][0]<minX:
                minX=verts[i][0]
            if verts[i][0]>maxX:
                maxX=verts[i][0]
            if verts[i][1]<minY:
                minY=verts[i][1]
            if verts[i][1]>maxY:   
                maxY=verts[i][1]
                    
        return vert((minX+maxX)/2,(minY+maxY)/2,0 )
    
    # center for half-dome
    # for a half-dome another algorithm is needed than for dome,
    # because it center is located not in the outline center,  but somewhere _on_ the outline itself
    # we will try the following:
    # middle of the longest edge.
    def getHalfDomeCenter(self): 

        verts = self.verts
        n=len(verts)
        # let's try middle of the longest edge.
        r_max = (verts[n-1][0]-verts[0][0])**2+(verts[n-1][1]-verts[0][1])**2
        x = (verts[n-1][0]+verts[0][0])/2
        y = (verts[n-1][1]+verts[0][1])/2
        #print (r_max,x,y)  

        for i in range(n-1):
            r=(verts[i][0]-verts[i+1][0])**2+(verts[i][1]-verts[i+1][1])**2
            if r>r_max:
                r_max = r
                x = (verts[i][0]+verts[i+1][0])/2
                y = (verts[i][1]+verts[i+1][1])/2
                #print (r_max,x,y)

        return vert(x,y,0)
 
   