#!/usr/bin/env python3

from multiple_images import MultipleImages

files = ["untitled5.blend", "church.blend"]
manager = MultipleImages(
	files=files,
	blenderFilesDir="../models",
	outputImagesDir="C:/Users/vvoovv/Documents/MapBox/project/tula/models",
	angleX = -45,
	angleY = -45
)
manager.render()
