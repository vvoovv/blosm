import os, json
import bpy
from app import app


def loadTextFromFile(filepath, name):
    """
    Loads a Blender collection with the given <name> from the .blend file with the given <filepath>
    """
    with bpy.data.libraries.load(filepath) as (data_from, data_to):
        # a Python list (not a Python tuple!) must be set to <data_to.meshes>
        data_to.texts = [name]
    return data_to.texts[0]


class TextureStore:
    
    textureInfoFilename = "texture_info.blend"
    
    def __init__(self):
        self.byBuildingType = {}
        
        textures = json.loads(
            loadTextFromFile(
                os.path.join(
                    os.path.dirname(os.path.abspath(app.bldgMaterialsFilepath)),
                    self.textureInfoFilename
                ),
                "textures_facade"
            ).as_string()
        )["textures"]
        
        byBuildingType = self.byBuildingType
        for texture in textures:
            buildingType = texture["type"]
            if not buildingType in byBuildingType:
                byBuildingType[buildingType] = []
            byBuildingType[buildingType].append(texture)