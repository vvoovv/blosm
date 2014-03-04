#!/usr/bin/env python3

from separate_objects import SeparateObjects

files = ["untitled4.blend", "church.blend"]
manager = SeparateObjects(files=files, blenderFilesDir="../models", outputImagesDir="C:/Users/vvoovv/Documents/MapBox/project/tula/models")
manager.render()
