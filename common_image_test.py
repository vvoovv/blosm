#!/usr/bin/env python3

from common_image import CommonImage

files = ["untitled5.blend", "church.blend"]
manager = CommonImage(
	files=files,
	blenderFilesDir="../models",
	gdalDir="C:/Program Files/GDAL",
	angleX = -45,
	angleY = -45
)
manager.render()
