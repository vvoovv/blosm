import bpy

def findGeoObject(context):
	"""
	Find a Blender object with "latitude" and "longitude" as custom properties
	"""
	for o in context.scene.objects:
		if "latitude" in o and "longitude" in o:
			return o
	return None