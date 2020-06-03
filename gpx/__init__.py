import os, sys
import bpy, bmesh
from util.blender import loadMaterialsFromFile

_isBlender280 = bpy.app.version[1] >= 80
_curveBevelObjectName = "gpx_bevel"

_bevelCurves = {
    "line": ( ( (-1., 0., 0.), (1., 0., 0.) ), False ), # False means: the curve isn't closed
    "square": (
        ( (-1., 0., 0.), (1., 0., 0.), (1., 2., 0.), (-1., 2., 0.) ),
        True # True means: the curve is closed
    )
}


class GpxRenderer:
    
    # Default value for <offset> parameter of the SHRINKWRAP modifier;
    swOffset = 0.5
    
    # relative path to default materials
    materialPath = "assets/base.blend"
    
    # name of the default material from <Overlay.materialPath>
    defaultMaterial = "gpx"
            
    def __init__(self, app):
        self.app = app
        
        self.curve = None
        self.spline = None

    def render(self, gpx):
        app = self.app
        
        name = os.path.basename(app.gpxFilepath)
        
        if app.gpxImportType == "curve":
            obj = self.makeCurve(gpx, name)
        else:
            obj = self.makeMesh(gpx, name)
        
        context = bpy.context
        if _isBlender280:
            context.scene.collection.objects.link(obj)
            context.view_layer.objects.active = obj
            obj.select_set(True)
        else:
            context.scene.objects.link(obj)
            context.scene.objects.active = obj
            obj.select = True
            context.scene.update()
    
    def makeMesh(self, gpx, name):
        app = self.app
        terrain = app.terrain
        projection = app.projection
        gpxProjectOnTerrain = app.gpxProjectOnTerrain
        if gpxProjectOnTerrain and not terrain:
            gpxProjectOnTerrain = False
        terrainOffset = terrain.terrain.get("terrain_offset") if terrain and not gpxProjectOnTerrain else None
        
        bm = bmesh.new()
        
        # create vertices and edges for the track segments
        for segment in gpx.segments:
            prevVertex = None
            for point in segment:
                v = projection.fromGeographic(point[0], point[1])
                v = bm.verts.new((
                    v[0],
                    v[1],
                    point[2]-terrainOffset if terrainOffset else\
                    ( (terrain.maxZ + terrain.layerOffset) if gpxProjectOnTerrain else point[2] )
                ))
                if prevVertex:
                    bm.edges.new([prevVertex, v])
                prevVertex = v
        
        # finalize
        mesh = bpy.data.meshes.new(name)
        bm.to_mesh(mesh)
        # cleanup
        bm.free()
        
        return bpy.data.objects.new(name, mesh)
    
    def makeCurve(self, gpx, name):
        app = self.app
        terrain = app.terrain
        projection = app.projection
        gpxProjectOnTerrain = app.gpxProjectOnTerrain
        if gpxProjectOnTerrain and not terrain:
            gpxProjectOnTerrain = False
        terrainOffset = terrain.terrain.get("height_offset") if terrain and not gpxProjectOnTerrain else None
        
        curve = bpy.data.curves.new(name, 'CURVE')
        curve.dimensions = '3D'
        curve.twist_mode = 'Z_UP'
        self.curve = curve
        
        for segment in gpx.segments:
            self.createSpline()
            for i, point in enumerate(segment):
                if i:
                    self.spline.points.add(1)
                v = projection.fromGeographic(point[0], point[1]) 
                self.setSplinePoint((
                    v[0],
                    v[1],
                    point[2]-terrainOffset if terrainOffset else\
                    ( (terrain.maxZ + terrain.layerOffset) if gpxProjectOnTerrain else point[2] )
                ))
        
        # set bevel object
        self.setCurveBevelObject("line")
        
        obj = bpy.data.objects.new(name, curve)
        
        self.applyMaterial(obj)
        
        if terrain and gpxProjectOnTerrain:
            self.addShrinkwrapModifier(obj, terrain.terrain, GpxRenderer.swOffset)
        
        # cleanup
        self.curve = None
        self.spline = None
        
        return obj
    
    def createSpline(self, curve=None):
        if not curve:
            curve = self.curve
        self.spline = curve.splines.new('POLY')
        self.pointIndex = 0

    def setSplinePoint(self, point):
        self.spline.points[self.pointIndex].co = (point[0], point[1], point[2], 1.)
        self.pointIndex += 1
    
    def setCurveBevelObject(self, bevelCurveId):
        bevelObj = bpy.data.objects.get(_curveBevelObjectName)
        if not (bevelObj and bevelObj.type == 'CURVE'):
            # create a Blender object of the type 'CURVE' to surve as a bevel object
            bevelCurve = bpy.data.curves.new(_curveBevelObjectName, 'CURVE')
            bevelCurveData, isBevelCurveClosed = _bevelCurves[bevelCurveId]
            
            self.createSpline(bevelCurve)
            self.spline.points.add( len(bevelCurveData)-1 )
            
            for point in bevelCurveData:
                self.setSplinePoint(point)
            
            if isBevelCurveClosed:
                self.spline.use_cyclic_u = True
            
            bevelObj = bpy.data.objects.new(_curveBevelObjectName, bevelCurve)
            
            if _isBlender280:
                bevelObj.hide_viewport = True
                bevelObj.hide_select = True
                bevelObj.hide_render = True
                bpy.context.scene.collection.objects.link(bevelObj)
            else:
                bevelObj.hide = True
                bevelObj.hide_select = True
                bevelObj.hide_render = True
                bpy.context.scene.objects.link(bevelObj)
        
        self.curve.bevel_object = bevelObj
    
    def addShrinkwrapModifier(self, obj, target, offset):
        m = obj.modifiers.new(name="Shrinkwrap", type='SHRINKWRAP')
        m.wrap_method = "PROJECT"
        m.use_positive_direction = False
        m.use_negative_direction = True
        m.use_project_z = True
        m.target = target
        m.offset = offset
    
    def applyMaterial(self, obj):
        material = bpy.data.materials.get(GpxRenderer.defaultMaterial)
        if not material:
            material = loadMaterialsFromFile(
                os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    os.pardir,
                    GpxRenderer.materialPath
                ),
                False, # i.e. append rather than link
                GpxRenderer.defaultMaterial
            )[0]
        obj.data.materials.append(material)