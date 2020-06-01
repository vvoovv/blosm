
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
from mathutils import Vector
from util.polygon import Polygon

# Utility functions
# dummy function to create vertex, wrapped into particular class required by blender-osm
def vert(x, y,z):
    """ Create a single vert """
    return Vector((x, y, z))
    #return (x, y, z)


"""
Profiles (generatrix) for zakomar roof type.
"""
zakomarRoof = (
        (0.0000, 0.000),
        (0.0208, 0.484),
        (0.0833, 0.866),
        (0.1667, 1.000),
        (0.2500, 0.866),
        (0.3125, 0.484),
        (0.3333, 0.000),
        (0.3541, 0.484),
        (0.4166, 0.866),
        (0.5000, 1.000),
        (0.5833, 0.866),
        (0.6458, 0.485),
        (0.6666, 0.000),
        (0.6874, 0.483),
        (0.7499, 0.866),
        (0.8333, 1.000),
        (0.9166, 0.866),
        (0.9791, 0.485),
        (1.0000, 0.000)
)

crossGabledRoof = (
        (0.0000, 0.000),
        (0.5000, 1.000),
        (1.0000, 0.000)
)


class RoofZakomar(Roof):
    """
    A Blender object to deal with buildings or building part with a zakomar (double direction profile) roof
    the roof mesh is created via directrix and generatrix.
    building outline is used as a directrix, and roof profiles defined above are used as a generatrix.
    practically any form can be created this way. dome, onion, pyramid
    see https://en.wikipedia.org/wiki/Generatrix 

    """
    
    defaultHeight = 4.0
    def __init__(self, strRoofType):

        super().__init__()
        self.roofType=strRoofType 
 
        if strRoofType=="zakomar":        
            self.roof_profile = zakomarRoof 

        elif strRoofType=="cross_gabled":
            self.roof_profile=crossGabledRoof

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
        # we will remove superfluous vertexes for ever. According to my tests they are not really needed anyway.
        for i in range(self.polygon.n):
          verts.append(self.verts[self.polygon.indices[i]])  
        self.verts = verts
        self.polygon = Polygon(verts)         
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 

        n=len(verts)
        if n==4:
            print("building outline will be used for zakomar roof")
        else:
            print("bbox will be used for zakomar roof")  
            print("Not supported yet") 
                    
        roof_profile=self.roof_profile #zakomarRoof
        rows=len(roof_profile)
        #z coordinates. we will use the following model 
        z0 = verts[0][2]  # bottom of the building part. Should be zero if building stands on the ground.  self.z1 ???
        z1 = self.roofVerticalPosition  # roof cornice
        z2 = self.z2 # top of the roof.

        #Walls
        noWalls = (z0 == z1) #building without walls, e.g. the Pyramid of Cheops
        if not noWalls:
            for i in range(n): 
                verts.append(vert(verts[i][0],verts[i][1],z1)) 

            for i in range(n-1):
                wallFaces.append((i,i+1,i+n+1,i+n))
            wallFaces.append((n-1,0,n,2*n-1))    
     
        #roof -- zakomar
        for i in range(2): 
            #outer cycle for directions, along and across
            
            offset=len(verts)
            #side faces of the roof
            face1=[] 
            face2=[]    
           
            #we will create some superflous vertices, otherwise algorithm becomes too complex.       
            for j in range(rows):
                
                if i==0:
                    xj1=verts[0][0] + roof_profile[j][0]*(verts[1][0]-verts[0][0])
                    yj1=verts[0][1] + roof_profile[j][0]*(verts[1][1]-verts[0][1])
                    
                    xj2=verts[3][0] + roof_profile[j][0]*(verts[2][0]-verts[3][0])
                    yj2=verts[3][1] + roof_profile[j][0]*(verts[2][1]-verts[3][1])
                else:
                    xj1=verts[1][0] + roof_profile[j][0]*(verts[2][0]-verts[1][0])
                    yj1=verts[1][1] + roof_profile[j][0]*(verts[2][1]-verts[1][1])
                    
                    xj2=verts[0][0] + roof_profile[j][0]*(verts[3][0]-verts[0][0])
                    yj2=verts[0][1] + roof_profile[j][0]*(verts[3][1]-verts[0][1])
                     
                #Высота (z)
                zj=z1+(z2-z1)*roof_profile[j][1]
                verts.append(vert(xj1,yj1,zj))
                verts.append(vert(xj2,yj2,zj))
                
            for j in range(rows):    
                face1.append(offset+(rows-1-j)*2)
                face2.append(offset+(j)*2+1)

                
            
            #let's create faces,
            # ... main faces of the roof
            for j in range(rows-1):
                roofFaces.append((offset+2*j,offset+2*j+2,offset+2*j+3,offset+2*j+1))
         
            # ... and vertical side faces   
            wallFaces.append(face1) 
            wallFaces.append(face2) 

        self.roofIndices = roofFaces
        self.wallIndices = wallFaces

        return True
