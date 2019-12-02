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

import os, math
import bpy, bmesh
from app import app
from util.blender import makeActive, createMeshObject, getBmesh,\
    loadParticlesFromFile, loadNodeGroupsFromFile, getMaterialIndexByName, getMaterialByName, getModifier

_isBlender280 = bpy.app.version[1] >= 80
  

class TerrainRenderer:
    """
    The class assigns material for the terrain
    """
    
    materialName = "terrain"
    
    baseAssetPath = "assets/base.blend"
    
    # default width of a image texture in meters
    w = 6.
    
    # default height of a image texture in meters
    h = 6.
    
    def render(self, app):
        assetPath = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), self.baseAssetPath
        )
        terrain = app.terrain
        # the size of the terrain
        sizeX = terrain.maxX - terrain.minX
        sizeY = terrain.maxY - terrain.minY
        # Blender object for the terrain
        terrain = terrain.terrain
        
        nodeGroups = tuple(layer.id for layer in app.layers if layer.area and layer.obj)
        
        materialName = self.materialName
        materials = terrain.data.materials
        
        if materials:
            if not materials[0] or materials[0].name != materialName:
                material = getMaterialByName(terrain, materialName, assetPath)
        else:
            materials.append(None)
            material = getMaterialByName(terrain, materialName, assetPath)
        
        if not material:
            return
        
        materials[0] = material
        
        uvMap = self.getNode(material, 'UVMAP')
        if not uvMap:
            return
        inp = self.getInput(material)
        if not inp:
            return
        output = self.getOutput(material)
        if not output:
            return
        
        # set correct scale for the texture
        for node in material.node_tree.nodes:
            if node.name == materialName:
                mapping = self.getNode(node, 'MAPPING')
                if mapping:
                    mapping.scale[0] = sizeX/mapping.scale[0]
                    mapping.scale[1] = sizeY/mapping.scale[1]
        
        loadNodeGroupsFromFile(assetPath, *nodeGroups)
        
        nodes = material.node_tree.nodes
        links = material.node_tree.links
        x,y = uvMap.location
        x += 350
        y += 200
        for name in nodeGroups:
            nodeGroup = bpy.data.node_groups.get(name)
            if nodeGroup:
                node = nodes.new('ShaderNodeGroup')
                node.location = x,y
                node.node_tree = nodeGroup
                node.name = name
                node.label = name
                # set correct scale for the texture
                mapping = self.getNode(node, 'MAPPING')
                if mapping:
                    mapping.scale[0] = sizeX/mapping.scale[0]
                    mapping.scale[1] = sizeY/mapping.scale[1]
                # find vector and color inputs of <node> and create links
                for _input in node.inputs:
                    if _input.type == 'VECTOR':
                        links.new(uvMap.outputs[0], _input)
                    elif _input.type == 'RGBA':
                        links.new(output, _input)
                # set the new value for the current output
                output = node.outputs[0]
                y += 200
        links.new(output, inp)
    
    @staticmethod
    def getNode(material, nodeType):
        for node in material.node_tree.nodes:
            if node.type == nodeType:
                return node
        return None

    @staticmethod
    def getInput(material):
        for node in material.node_tree.nodes:
            for inp in node.inputs:
                if not inp.is_linked:
                    return inp
        return None
    
    @staticmethod
    def getOutput(material):
        for node in material.node_tree.nodes:
            for output in node.outputs:
                if not output.is_linked:
                    return output
        return None


class AreaRenderer:
    
    baseAssetPath = "assets/base.blend"
    
    calculateArea = False
    
    def __init__(self):
        self.assetPath = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), self.baseAssetPath
        )
    
    def finalizeBlenderObject(self, layer, app):
        terrain = app.terrain
        obj = layer.obj
        # add modifiers, slice flat mesh
        if not terrain.envelope:
            terrain.createEnvelope()
        layer.addBoolenModifier(obj, terrain.envelope)
        makeActive(obj)
        bpy.ops.object.modifier_apply(apply_as='DATA', modifier="Boolean")
        # calculate area after the BOOLEAN modifier has been applied
        if self.calculateArea:
            bm = getBmesh(obj)
            layer.surfaceArea = sum(face.calc_area() for face in bm.faces)
            bm.free()
        layer.slice(obj, terrain, app)
        layer.addShrinkwrapModifier(obj, terrain.terrain, layer.swOffset)
        bpy.ops.object.modifier_apply(apply_as='DATA', modifier="Shrinkwrap")
        if _isBlender280:
            obj.select_set(False)
        else:
            obj.select = False

    def renderTerrain(self, layer, terrain, **kwargs):
        vertexColors = kwargs.get("vertexColors", True)
        use_antialiasing = kwargs.get("use_antialiasing", True)
        
        layerId = layer.id
        obj = layer.obj
        # DYNAMIC_PAINT modifier of <terrain>
        m = getModifier(terrain, 'DYNAMIC_PAINT')
        # create a brush collection if necessary
        collectionName = "%s_brush" % layerId
        if _isBlender280:
            collection = bpy.data.collections[collectionName]\
                if collectionName in bpy.data.collections\
                else bpy.data.collections.new(collectionName)
        else:
            collection = bpy.data.groups[collectionName]\
                if collectionName in bpy.data.groups\
                else bpy.data.groups.new(collectionName)
        # add <obj> to the brush collection
        collection.objects.link(obj)
        # vertex colors
        if vertexColors:
            bpy.ops.dpaint.surface_slot_add()
            surface = m.canvas_settings.canvas_surfaces[-1]
            surface.name = "%s_colors" % layerId
            # create a target vertex colors layer for dynamically painted vertex colors
            colors = terrain.data.vertex_colors.new(name=layerId)
            AreaRenderer.prepareDynamicPaintSurface(surface, collection, colors.name, use_antialiasing)
        # vertex weights
        bpy.ops.dpaint.surface_slot_add()
        surface = m.canvas_settings.canvas_surfaces[-1]
        surface.name = "%s_weights" % layerId
        surface.surface_type = 'WEIGHT'
        # create if necessary a target vertex group for dynamically painted vertex weights
        weights = terrain.vertex_groups[layerId]\
            if layerId in terrain.vertex_groups\
            else terrain.vertex_groups.new(name=layerId)
        AreaRenderer.prepareDynamicPaintSurface(surface, collection, weights.name, use_antialiasing)
    
    def renderArea(self, layer, app):
        # setup a DYNAMIC_PAINT modifier for <layer.obj> in the brush mode
        obj = layer.obj
        makeActive(obj)
        m = obj.modifiers.new("Dynamic Paint", 'DYNAMIC_PAINT')
        m.ui_type = 'BRUSH'
        bpy.ops.dpaint.type_toggle(type='BRUSH')
        brush = m.brush_settings
        brush.paint_color = (1., 1., 1.)
        brush.paint_source = 'DISTANCE'
        brush.paint_distance = 500.
        brush.use_proximity_project = True
        brush.ray_direction = 'Z_AXIS'
        brush.proximity_falloff = 'CONSTANT'
        if _isBlender280:
            obj.hide_viewport = True
            obj.hide_render = True
            # deselect <obj> to ensure correct work of subsequent operators
            obj.select_set(False)
        else:
            obj.hide = True
            # deselect <obj> to ensure correct work of subsequent operators
            obj.select = False
            obj.hide_render = True
    
    @staticmethod
    def addSubsurfModifier(terrain):
        # check if <terrain> has a SUBSURF modifier
        for m in terrain.modifiers:
            if m.type == 'SUBSURF':
                break
        else:
            # add a SUBSURF modifier
            m = terrain.modifiers.new("Subsurf", 'SUBSURF')
            m.subdivision_type = 'SIMPLE'
            if not _isBlender280:
                m.use_subsurf_uv = False
            m.levels = 0
            m.render_levels = 2
    
    @staticmethod
    def beginDynamicPaintCanvas(terrain):
        # setup a DYNAMIC_PAINT modifier for <terrain> with canvas surfaces
        makeActive(terrain)
        # DYNAMIC_PAINT modifier of <terrain>
        m = getModifier(terrain, 'DYNAMIC_PAINT')
        if not m:
            terrain.modifiers.new("Dynamic Paint", 'DYNAMIC_PAINT')
            bpy.ops.dpaint.type_toggle(type='CANVAS')
            bpy.ops.dpaint.surface_slot_remove()

    @staticmethod
    def endDynamicPaintCanvas(terrain):
        # DYNAMIC_PAINT modifier of <terrain>
        m = getModifier(terrain, 'DYNAMIC_PAINT')
        # setup a DYNAMIC_PAINT modifier for <terrain> with canvas surfaces
        if not _isBlender280:
            m.canvas_settings.canvas_surfaces[-1].show_preview = False
        m.canvas_settings.canvas_surfaces.active_index = 0
        if _isBlender280:
            terrain.select_set(False)
        else:
            terrain.select = False
    
    @staticmethod
    def prepareDynamicPaintSurface(surface, collection, output_name_a, use_antialiasing=True):
        surface.use_antialiasing = use_antialiasing
        surface.use_drying = False
        surface.use_dry_log = False
        surface.use_dissolve_log = False
        if _isBlender280:
            surface.brush_collection = collection
        else:
            surface.brush_group = collection
        surface.output_name_a = output_name_a
        surface.output_name_b = ""


class VertexGroupBaker(AreaRenderer):
    
    def finalizeBlenderObject(self, layer, app):
        terrain = app.terrain
        terrainObj = terrain.terrain
        layerId = layer.id
        obj = layer.obj
        # add modifiers, slice flat mesh
        if not terrain.envelope:
            terrain.createEnvelope()
        layer.addBoolenModifier(obj, terrain.envelope)
        makeActive(obj)
        bpy.ops.object.modifier_apply(apply_as='DATA', modifier="Boolean")
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        # do mesh cleanup
        bpy.ops.mesh.delete_loose()
        bpy.ops.object.mode_set(mode='OBJECT')
        
        # Create a copy of <obj> to serve as an envelope;
        # it will be joined with <terrainObj>
        objEnvelope = createMeshObject("%s_envelope" % obj.name, obj.location, obj.data.copy())
        objEnvelope.parent = obj.parent
        
        layer.slice(obj, terrain, app)
        layer.addShrinkwrapModifier(obj, terrainObj, layer.swOffset)
        bpy.ops.object.modifier_apply(apply_as='DATA', modifier="Shrinkwrap")
        super().renderArea(layer, app)
        if _isBlender280:
            obj.select_set(False)
        else:
            obj.select = False
        
        AreaRenderer.beginDynamicPaintCanvas(terrainObj)
        super().renderTerrain(layer, terrainObj, vertexColors=False, use_antialiasing=False)
        # a reference to the Blender group created in <super().renderTerrain(..)>
        group = bpy.data.groups[-1]
        bpy.ops.object.modifier_apply(apply_as='DATA', modifier="Dynamic Paint")
        if _isBlender280:
            terrainObj.select_set(False)
        else:
            terrainObj.select = False
        # delete dynamic paint brush object
        bpy.data.meshes.remove(obj.data, True)
        # delete <group>
        bpy.data.groups.remove(group, True)
        
        makeActive(objEnvelope)
        # select and extrude all vertices down to create an envelope
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.delete(type='ONLY_FACE')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.extrude_edges_move(TRANSFORM_OT_translate={
            "value": (0., 0., -(obj.location[2]-terrain.minZ+terrain.layerOffset))
        })
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.object.mode_set(mode='OBJECT')
        # keep <obj> selected for the upcoming join operator
        
        # deselect vertices of the terrain Blender object
        makeActive(terrainObj)
        # force vertex select mode
        bpy.context.tool_settings.mesh_select_mode = (True, False, False)
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='DESELECT')
        bpy.ops.object.mode_set(mode='OBJECT')
        
        # join <obj> and the terrain Blender object
        bpy.ops.object.join()
        
        bpy.ops.object.mode_set(mode='EDIT')
        # the name for the vertex group for the envelope
        envelopeName = "%s_envelope" % layerId
        terrainObj.vertex_groups.new(envelopeName)
        bpy.ops.object.vertex_group_assign()
        # cut intersection of terrain mesh with the joined mesh from <obj>
        bpy.ops.mesh.intersect()
        # remove double vertices for the intersection
        bpy.ops.mesh.remove_doubles()
        terrainObj.vertex_groups.active_index = terrainObj.vertex_groups[layerId].index
        bpy.ops.object.vertex_group_assign()
        
        bpy.ops.mesh.select_all(action='DESELECT')
        # select the vertex group with the name <envelopName> and delete the vertices
        envelopeGroup = terrainObj.vertex_groups[envelopeName]
        terrainObj.vertex_groups.active_index = envelopeGroup.index
        bpy.ops.object.vertex_group_select()
        bpy.ops.mesh.delete(type='VERT')
        terrainObj.vertex_groups.remove(envelopeGroup)
        
        terrainObj.vertex_groups.active_index = terrainObj.vertex_groups[layerId].index
        bpy.ops.object.vertex_group_select()
        
        bpy.ops.object.mode_set(mode='OBJECT')
        
        if _isBlender280:
            terrain.select_set(False)
        else:
            terrain.select = False
    
    def renderTerrain(self, layer, terrain):
        pass
    
    def renderArea(self, layer, app):
        pass


class ForestRenderer(AreaRenderer):
    
    # name for the particle settings for a forest
    particles = "forest"
    
    calculateArea = True
    
    # maximum number of Blender particles object to be displayed in the Blender 3D View
    maxDisplayCount = 10000
    
    def renderArea(self, layer, app):
        super().renderArea(layer, app)
        
        # the terrain Blender object
        terrain = app.terrain.terrain
        
        layerId = layer.id
        
        # make the Blender object for the terrain the active one
        if not ((_isBlender280 and terrain.select_get()) or (not _isBlender280 and terrain.select)):
            makeActive(terrain)
        
        bpy.ops.object.particle_system_add()
        # do not show particles in the viewport for better performance
        #terrain.modifiers[-1].show_viewport = False
        # <ps> stands for particle systems
        ps = terrain.particle_systems[-1]
        ps.name = layerId
        # name for the particle settings for a forest
        name = self.particles
        particles = bpy.data.particles.get(name)\
            if name in bpy.data.particles else\
            loadParticlesFromFile(app.vegetationFilepath, name)
        # the total number of particles
        count = math.ceil(app.treeDensity/10000*layer.surfaceArea)
        if count > self.maxDisplayCount:
            if _isBlender280:
                particles.display_percentage = math.floor(self.maxDisplayCount/count*100)
            else:
                particles.draw_percentage = math.floor(self.maxDisplayCount/count*100)
        particles.count = count
        ps.vertex_group_density = layerId
        ps.settings = particles


class WaterRenderer(VertexGroupBaker):
    
    def renderArea(self, layer, app):
        # the terrain Blender object
        terrain = app.terrain.terrain
        # Vertex group with the name <layerId> has been already selected in
        # <self.finalizeBlenderObject(..)>
        makeActive(terrain)
        bpy.ops.object.mode_set(mode='EDIT')
        # if there are no materials create an empty slot
        if not terrain.data.materials:
            terrain.data.materials.append(None)
        terrain.active_material_index = getMaterialIndexByName(terrain, layer.id, self.assetPath)
        bpy.ops.object.material_slot_assign()
        bpy.ops.object.mode_set(mode='OBJECT')
        if _isBlender280:
            terrain.select_set(False)
        else:
            terrain.select = False


class BareRockRenderer(VertexGroupBaker):
    pass