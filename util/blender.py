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

import bpy, bmesh

_isBlender280 = bpy.app.version[1] >= 80


def makeActive(obj, context=None):
    if not context:
        context = bpy.context
    if _isBlender280:
        obj.select_set(True)
        context.view_layer.objects.active = obj
    else:
        obj.select = True
        context.scene.objects.active = obj


def createMeshObject(name, location=(0., 0., 0.), mesh=None, collection=None):
    if not mesh:
        mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)
    obj.location = location
    if _isBlender280:
        if not collection:
            collection = bpy.context.scene.collection
        collection.objects.link(obj)
    else:
        bpy.context.scene.objects.link(obj)
    return obj


def createEmptyObject(name, location, hide=False, collection=None, **kwargs):
    obj = bpy.data.objects.new(name, None)
    obj.location = location
    if _isBlender280:
        obj.hide_viewport = hide
    else:
        obj.hide = hide
    obj.hide_select = hide
    obj.hide_render = True
    if kwargs:
        if _isBlender280 and "empty_draw_size" in kwargs:
            kwargs["empty_display_size"] = kwargs["empty_draw_size"]
            del kwargs["empty_draw_size"]
        for key in kwargs:
            setattr(obj, key, kwargs[key])
    if _isBlender280:
        if not collection:
            collection = bpy.context.scene.collection
        collection.objects.link(obj)
    else:
        bpy.context.scene.objects.link(obj)
    return obj


def createCollection(name, parent=None, hide_viewport=False, hide_select=False, hide_render=False):
    collection = bpy.data.collections.new(name)
    if not parent:
        parent = bpy.context.scene.collection
    parent.children.link(collection)
    collection.hide_viewport = hide_viewport
    collection.hide_select = hide_select
    collection.hide_render = hide_render
    return collection


def getBmesh(obj):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    return bm


def setBmesh(obj, bm):
    bm.to_mesh(obj.data)
    bm.free()
    

def pointNormalUpward(face):
    if face.normal.z < 0.:
        face.normal_flip()


def createDiffuseMaterial(name, color):
    material = bpy.data.materials.new(name)
    material.diffuse_color = (color[0], color[1], color[2], 1.) if _isBlender280 else color
    return material


def loadMeshFromFile(filepath, name):
    """
    Loads a Blender mesh with the given <name> from the .blend file with the given <filepath>
    """
    with bpy.data.libraries.load(filepath) as (data_from, data_to):
        # a Python list (not a Python tuple!) must be set to <data_to.meshes>
        data_to.meshes = [name]
    return data_to.meshes[0]


def loadParticlesFromFile(filepath, name):
    """
    Loads Blender particles settings with the given <name> from the .blend file
    with the given <filepath>
    """
    with bpy.data.libraries.load(filepath) as (data_from, data_to):
        # a Python list (not a Python tuple!) must be set to <data_to.particles>
        data_to.particles = [name]
    return data_to.particles[0]


def loadCollectionFromFile(filepath, name):
    """
    Loads a Blender collection with the given <name> from the .blend file with the given <filepath>
    """
    with bpy.data.libraries.load(filepath) as (data_from, data_to):
        # a Python list (not a Python tuple!) must be set to <data_to.meshes>
        data_to.collections = [name]
    return data_to.collections[0]


def loadSceneFromFile(filepath, name):
    """
    Loads a Blender scene with the given <name> from the .blend file with the given <filepath>
    """
    with bpy.data.libraries.load(filepath) as (data_from, data_to):
        # a Python list (not a Python tuple!) must be set to <data_to.meshes>
        data_to.scenes = [name]
    return data_to.scenes[0]


def loadGroupFromFile(filepath, name):
    """
    Loads a Blender group with the given <name> from the .blend file with the given <filepath>
    """
    with bpy.data.libraries.load(filepath) as (data_from, data_to):
        # a Python list (not a Python tuple!) must be set to <data_to.meshes>
        data_to.groups = [name]
    return data_to.groups[0]


def loadImagesFromFile(filepath, *names):
    """
    Loads images with <names> from the .blend file with the given <filepath>.
    If an image name is available at <bpy.data.images>, the image won't be loaded
    """
    with bpy.data.libraries.load(filepath) as (data_from, data_to):
        # a Python list (not a Python tuple!) must be set to <data_to.images>
        data_to.images = [
            name for name in names if not name in bpy.data.images and name in data_from.images
        ]


def loadNodeGroupsFromFile(filepath, *names):
    """
    Loads node groups with <names> from the .blend file with the given <filepath>.
    If a node group is available at <bpy.data.node_groups>, the node groups won't be loaded
    """
    with bpy.data.libraries.load(filepath) as (data_from, data_to):
        # a Python list (not a Python tuple!) must be set to <data_to.images>
        data_to.node_groups = [
            name for name in names if not name in bpy.data.node_groups and name in data_from.node_groups
        ]


def loadTextFromFile(filepath, name):
    """
    Loads a Blender text with the given <name> from the .blend file with the given <filepath>
    """
    with bpy.data.libraries.load(filepath) as (data_from, data_to):
        # a Python list (not a Python tuple!) must be set to <data_to.meshes>
        data_to.texts = [name]
    return data_to.texts[0]


def appendObjectsFromFile(filepath, collection, *names):
    with bpy.data.libraries.load(filepath) as (data_from, data_to):
        # a Python list (not a Python tuple!) must be set to <data_to.objects>
        data_to.objects = list(names)
    # append the objects to the Blender scene
    for obj in data_to.objects:
        if obj:
            if _isBlender280:
                if collection:
                    collection.objects.link(obj)
                    obj.select_set(False)
            else:
                obj.select = False
                bpy.context.scene.objects.link(obj)
    # return the appended Blender objects
    return data_to.objects


def getMaterialIndexByName(obj, name, filepath):
    """
    Check if Blender material with the <name> is already set for <obj>,
    if not, check if the material is available in bpy.data.material
    (if yes, append it to <obj>),
    if not, load the material with the <name> from the .blend with the given <filepath>
    and append it to <obj>.
    """
    if name in obj.data.materials:
        material = obj.data.materials[name]
        # find index of the material
        for materialIndex,m in enumerate(obj.data.materials):
            if material == m:
                break
    elif name in bpy.data.materials:
        materialIndex = len(obj.data.materials)
        obj.data.materials.append( bpy.data.materials[name] )
    else:
        with bpy.data.libraries.load(filepath) as (data_from, data_to):
            data_to.materials = [name]
        material = data_to.materials[0]
        materialIndex = len(obj.data.materials)
        obj.data.materials.append(material)
    return materialIndex


def getMaterialByName(obj, name, filepath=None):
    """
    Check if Blender material with the <name> is already set for <obj>,
    if not, check if the material is available in bpy.data.material,
    if not and if <filepath> is provided, load the material with the <name>
    from the .blend with the given <filepath>.
    The material is NOT appended to <obj>.
    """
    material = None
    if name in obj.data.materials:
        material = obj.data.materials[name]
    elif name in bpy.data.materials:
        material = bpy.data.materials[name]
    elif filepath:
        with bpy.data.libraries.load(filepath) as (data_from, data_to):
            data_to.materials = [name]
        material = data_to.materials[0]
    return material


def loadMaterialsFromFile(filepath, link, *names):
    with bpy.data.libraries.load(filepath, link=link) as (data_from, data_to):
        # a Python list (not a Python tuple!) must be set to <data_to.objects>
        data_to.materials = list(names)
    return data_to.materials


def getModifier(obj, modifierType):
    for m in obj.modifiers:
        if m.type == modifierType:
            break
    else:
        m = None
    return m