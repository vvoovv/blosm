import bpy

def findEmptyGeoObject(context):
	"""
	Find an empty Blender object with "latitude" and "longitude" as custom properties
	"""
	for o in context.scene.objects:
		if o.type == "EMPTY" and "latitude" in o and "longitude":
			return o
	return None