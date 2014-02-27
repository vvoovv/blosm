bl_info = {
	"name": "Build Neighborhood",
	"author": "Vladimir Elistratov",
	"description": "Builds Neighborhood",
	"version": (0, 1),
	"blender": (2, 6, 8),
	"category": "Object",
	"location": "Object > Build Neighborhood",
	"warning": "",
	"wiki_url": "",
	"tracker_url": ""
}

import sys, imp
import bpy

sys.path.append("D:\\projects\\blender\\blender-geo")

import main, handlers, srtm

def doit(**kwargs):
	imp.reload(main)
	imp.reload(handlers)
	imp.reload(srtm)
	if "clear" in kwargs and kwargs["clear"]: delete_objects()
	main.build_neighborhood(kwargs)

def delete_objects():
	for item in bpy.data.objects:
		if item.type == "MESH" or item.type == "EMPTY":
			item.select = True
		else:
			item.select = False
	bpy.ops.object.delete()
	# delete all meshes
	for item in bpy.data.meshes:
		# skipping linked meshes
		if item.library is None:
			bpy.data.meshes.remove(item)


def register():
	bpy.doit = doit