from .container import Container
from ..level import CurtainWall as CurtainWallBase


class CurtainWall(CurtainWallBase, Container):
    
    def __init__(self):
        # a reference to the Container class used in the parent classes
        self.Container = Container
        Container.__init__(self, exportMaterials=True)
        CurtainWallBase.__init__(self)
    
    def makeTexture(self, textureFilename, textureDir, textureFilepath, color, facadeTextureInfo, claddingTextureInfo, uvs):
        textureExporter = self.r.textureExporter
        scene = textureExporter.getTemplateScene("compositing_facade_specular_color")
        nodes = textureExporter.makeCommonPreparations(
            scene,
            textureFilename,
            textureDir
        )
        # facade texture
        textureExporter.setImage(
            facadeTextureInfo["name"],
            facadeTextureInfo["path"],
            nodes,
            "facade_texture"
        )
        specularMapName = facadeTextureInfo.get("specularMapName")
        if specularMapName:
            textureExporter.setImage(
                    specularMapName,
                    facadeTextureInfo["path"],
                    nodes,
                    "specular_map"
                )
        # cladding color
        textureExporter.setColor(color, nodes, "cladding_color")
        # render the resulting texture
        textureExporter.renderTexture(scene, textureFilepath)