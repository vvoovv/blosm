#!/usr/bin/env python3

from separate_objects import SeparateObjects

files = ["untitled4.blend", "untitled4.blend"]
manager = SeparateObjects(files=files, blenderFilesDir="../")
manager.render()
