import bpy, bmesh
from mathutils.bvhtree import BVHTree
from . import Renderer
from terrain import direction
from util.osm import assignTags


class CurveRenderer(Renderer):
    
    # <insetValue> the maximum way width in <assets/way_profiles.blend>
    insetValue = 10.
    
    def __init__(self, app):
        super().__init__(app)
        self.bvhTree = None
    
    def prepare(self):
        terrain = self.app.terrain
        if terrain:
            if not terrain.envelope:
                terrain.createEnvelope()
            # BMesh <bm> is used to check if a way's node is located
            # within the terrain. It's smaller than <terrain.envelope>
            # since all points of the bevel object applied to the Blender curve
            # must be located within <terrain> to avoid weird results of
            # the SHRINKWRAP modifier
            bm = bmesh.new()
            bm.from_mesh(terrain.envelope.data)
            # inset faces to avoid weird results of the BOOLEAN modifier
            insetFaces = bmesh.ops.inset_region(bm, faces=bm.faces,
                use_boundary=True, use_even_offset=True, use_interpolate=True,
                use_relative_offset=False, use_edge_rail=False, use_outset=False,
                thickness=self.insetValue, depth=0.
            )['faces']
            bmesh.ops.delete(bm, geom=insetFaces, context=5)
            self.bvhTree = BVHTree.FromBMesh(bm)
            # <bm> isn't needed anymore
            bm.free()
    
    def preRender(self, element):
        layer = element.l
        self.layer = layer
        
        if layer.singleObject:
            if not layer.obj:
                layer.obj = self.createBlenderObject(
                    layer.name,
                    layer.location,
                    self.parent
                )
            self.obj = layer.obj
        else:
            self.obj = self.createBlenderObject(
                self.getName(element),
                self.offsetZ or self.offset or layer.location,
                layer.getParent()
            )

    def renderLineString(self, element, data):
        self._renderLineString(element, element.getData(data), element.isClosed())

    def renderMultiLineString(self, element, data):
        for i,l in enumerate( element.getDataMulti(data) ):
            self._renderLineString(element, l, element.isClosed(i))
    
    def _renderLineString(self, element, coords, closed):
        spline = self.obj.data.splines.new('POLY')
        z = self.layer.meshZ
        if self.app.terrain:
            index = 0
            for coord in coords:
                # Cast a ray from the point with horizontal coords equal to <coords> and
                # z = <z> in the direction of <direction>
                if self.bvhTree.ray_cast((coord[0], coord[1], z), direction)[0]:
                    if index:
                        spline.points.add(1)
                    spline.points[index].co = (coord[0], coord[1], z, 1.)
                    index += 1
        else:
            for i, coord in enumerate(coords):
                if i:
                    spline.points.add(1)
                spline.points[i].co = (coord[0], coord[1], z, 1.)
        if len(spline.points) == 1:
            self.obj.data.splines.remove(spline)
        elif closed:
            spline.use_cyclic_u = True
    
    def postRender(self, element):
        layer = element.l
        
        if not layer.singleObject:
            obj = self.obj
            # assign OSM tags to the blender object
            assignTags(obj, element.tags)
            layer.finalizeBlenderObject(obj)
    
    def cleanup(self):
        super().cleanup()
        self.bvhTree = None

    @classmethod
    def createBlenderObject(self, name, location, parent):
        curve = bpy.data.curves.new(name, 'CURVE')
        curve.fill_mode = 'NONE'
        obj = bpy.data.objects.new(name, curve)
        if location:
            obj.location = location
        bpy.context.scene.objects.link(obj)
        if parent:
            # perform parenting
            obj.parent = parent
        return obj