import os
import bpy
from . import Roof
from renderer import Renderer
from util.blender import loadMeshFromFile


class RoofMesh(Roof):
    
    defaultHeight = 10.
    
    def __init__(self, mesh):
        super().__init__()
        self.mesh = mesh
    
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
        mesh = bpy.data.meshes.get(self.mesh)\
            if self.mesh in bpy.data.meshes else\
            loadMeshFromFile(os.path.join(op.assetPath, self.assetPath), self.mesh)
        if not mesh.materials:
            mesh.materials.append(None)
        o = bpy.data.objects.new(self.mesh, mesh)
        o.location = self.location
        o.scale = (
            ( max(v.x for v in polygon.verts) - min(v.x for v in polygon.verts) )/2.,
            ( max(v.y for v in polygon.verts) - min(v.y for v in polygon.verts) )/2.,
            self.h
        )
        bpy.context.scene.objects.link(o)
        o.parent = r.obj
        # link Blender material to the Blender object <o> instead of <o.data>
        slot = o.material_slots[0]
        slot.link = 'OBJECT'
        slot.material = r.getRoofMaterial(self.element)
        
        Renderer.addForJoin(o, r.obj)