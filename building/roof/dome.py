import os
import bpy
from . import Roof
from util.blender import loadMeshFromFile


class RoofDome(Roof):
    
    mesh = "roof_dome"
    
    defaultHeight = 10.
    
    def init(self, element, minHeight, osm):
        super().init(element, minHeight, osm)
    
    def make(self, bldgMaxHeight, roofMinHeight, bldgMinHeight, osm):
        polygon = self.polygon
        
        if not bldgMinHeight is None:
            polygon.sidesPrism(roofMinHeight, self.wallIndices)
        
        c = polygon.center
        self.location = (c.x, c.y, roofMinHeight)
        
        return True
    
    def render(self, r):
        polygon = self.polygon
        op = r.op
        
        # create walls
        super().render(r)
        
        # now deal with the roof
        mesh = loadMeshFromFile(os.path.join(op.assetPath, self.assetPath), self.mesh)
        o = bpy.data.objects.new(self.mesh, mesh)
        o.location = self.location
        o.scale = (
            ( max(v.x for v in polygon.verts) - min(v.x for v in polygon.verts) )/2.,
            ( max(v.y for v in polygon.verts) - min(v.y for v in polygon.verts) )/2.,
            self.h
        )
        bpy.context.scene.objects.link(o)
        o.parent = r.obj
        o.data.materials.append(r.getMaterial(self.element))