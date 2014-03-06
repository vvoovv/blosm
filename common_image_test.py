#!/usr/bin/env python3

from common_image import CommonImage

files = ["untitled4.blend", "church.blend"]
manager = CommonImage(files=files, blenderFilesDir="../models", gdalDir="C:/Program Files/GDAL")
manager.render()
