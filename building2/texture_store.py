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
    
    def getTextureInfo(self, building, facadePatternInfo):
        buildingAttrs = building.metaStyleBlock.attrs
        textures = self.byBuildingType[ buildingAttrs.get("buildingType") ]
        bestScore = 0
        bestFit = None
        for texture in textures:
            # calculate the score
            score = 0
            for _item in facadePatternInfo:
                if _item in texture["content"]:
                    # difference in the number of items in the texture and in the style block
                    dif = abs(texture["content"][_item] - facadePatternInfo[_item])
                    if not dif:
                        score += 10
                    elif dif == 1:
                        score += 5
                    elif dif == 2:
                        score += 3
                    else:
                        score +=1
            if score > bestScore:
                bestScore = score
                bestFit = texture
        return bestFit