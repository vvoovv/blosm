import os
import bpy
from manager import Manager
from . import ItemRenderer
from grammar.arrangement import Horizontal, Vertical
from grammar.symmetry import MiddleOfLast, RightmostOfLast

from util.blender_extra.material import createMaterialFromTemplate, setImage, setCustomNodeValue

from util import zAxis


_customNode = "FacadeOverlay"

_claddingMaterialTemplateFilename = "building_material_templates.blend"
_claddingMaterialTemplateName = "tiles_color_template"


class Container(ItemRenderer):
    """
    The base class for the item renderers Facade, Div, Layer, Basement
    """
        
    def getItemRenderer(self, item):
        return self.itemRenderers[item.__class__.__name__]
    
    def renderMarkup(self, item):
        item.prepareMarkupItems()
        
        if item.styleBlock.markup[0].isLevel:
            self.renderLevels(item)
        else:
            self.renderDivs(item)
        if not item.valid:
            return
    
    def renderLevels(self, item):
        parentIndices = item.indices
        levelGroups = item.levelGroups
        levelGroups.init()
        # sanity check
        width = item.getWidthForVerticalArrangement()
        if width > item.width:
            item.valid = False
            return
        
        footprint = item.footprint
        building = item.building
        levelHeights = footprint.levelHeights
        
        prevIndex1 = parentIndices[0]
        prevIndex2 = parentIndices[1]
        index1 = len(building.verts)
        index2 = index1 + 1
        texU1, texV = item.uvs[0]
        texU2 = item.uvs[1][0]
        
        # treat the basement
        if not footprint.minHeight:
            basementHeight = item.getStyleBlockAttr("basementHeight")
            if basementHeight is None:
                basementHeight = levelHeights.basementHeight
            if basementHeight:
                prevIndex1, prevIndex2, index1, index2, texV = self.generateLevelDiv(
                    building, levelGroups.basement, item, self.basementRenderer, basementHeight,
                    prevIndex1, prevIndex2, index1, index2,
                    texU1, texU2, texV
                )
        
        # treat the level groups
        groups = levelGroups.groups
        numGroups = levelGroups.numActiveGroups
        minLevel = footprint.minLevel
        groupFound = not minLevel
        if numGroups > 1:
            for i in range(numGroups-1):
                group = groups[i]
                if not groupFound and group.index1 <= minLevel <= group.index2:
                    groupFound = True
                if groupFound:
                    height = levelHeights.getLevelHeight(group.index1)\
                        if group.singleLevel else\
                        levelHeights.getHeight(group.index1, group.index2)
                    prevIndex1, prevIndex2, index1, index2, texV = self.generateLevelDiv(
                        building, group, item, self.levelRenderer.getRenderer(group), height,
                        prevIndex1, prevIndex2, index1, index2,
                        texU1, texU2, texV
                    )
        
        # the last level group
        group = groups[numGroups-1]
        indices = (prevIndex1, prevIndex2, parentIndices[2], parentIndices[3])
        texV2 = item.uvs[2][1]
        self.levelRenderer.getRenderer(group).render(
            building, group, item,
            indices,
            ( (texU1, texV), (texU2, texV), (texU2, texV2), (texU1, texV2) ),
            texU1,
            texV
        )
    
    def renderDivs(self, item):
        # <r> is the global building renderer
        r = self.r
        building = item.building
        parentIndices = item.indices
        
        if item.arrangement is Horizontal:
            # get markup width and number of repeats
            item.calculateMarkupDivision()
            if not item.valid:
                return
            # create vertices for the markup items
            numItems = len(item.markup)
            if numItems == 1:
                # the special case
                _item = item.markup[0]
                _item.indices = parentIndices
                _item.uvs = item.uvs
                r.createFace(building, _item.indices, _item.uvs)
            else:
                numRepeats = item.numRepeats
                symmetry = item.symmetry
                verts = building.verts
                prevIndex1 = parentIndices[0]
                prevIndex2 = parentIndices[3]
                #self.v1 = verts[self.prevIndex1]
                #self.v2 = verts[self.prevIndex2]
                unitVector = (verts[parentIndices[1]] - verts[prevIndex1]) / item.width
                index1 = len(building.verts)
                index2 = index1 + 1
                # <texU> is the current U-coordinate for texturing
                # <texV1> and <texV2> are the lower and upper V-coordinates for texturing
                texU, texV1 = item.uvs[0]
                texV2 = item.uvs[3][1]
                
                # Generate Div items but the last one;
                # the special case is when a symmetry is available
                if numRepeats>1:
                    for _ in range(numRepeats-1):
                        prevIndex1, prevIndex2, index1, index2, texU = self.generateDivs(
                            building, item, unitVector,
                            0, numItems, 1,
                            prevIndex1, prevIndex2, index1, index2,
                            texU, texV1, texV2
                        )
                        if symmetry:
                            prevIndex1, prevIndex2, index1, index2, texU = self.generateDivs(
                                building, item, unitVector,
                                numItems-2 if symmetry is MiddleOfLast else numItems-1, -1, -1,
                                prevIndex1, prevIndex2, index1, index2,
                                texU, texV1, texV2
                            )
                prevIndex1, prevIndex2, index1, index2, texU = self.generateDivs(
                    building, item, unitVector,
                    0, numItems if symmetry else numItems-1, 1,
                    prevIndex1, prevIndex2, index1, index2,
                    texU, texV1, texV2
                )
                if symmetry:
                    prevIndex1, prevIndex2, index1, index2, texU = self.generateDivs(
                        building, item, unitVector,
                        numItems-2 if symmetry is MiddleOfLast else numItems-1, 0, -1,
                        prevIndex1, prevIndex2, index1, index2,
                        texU, texV1, texV2
                    )
                # process the last item
                lastItem = item.markup[0] if symmetry else item.markup[-1]
                texU2 = item.uvs[1][0]
                self.getItemRenderer(lastItem).render(
                    lastItem,
                    (prevIndex1, parentIndices[1], parentIndices[2], prevIndex2),
                    ( (texU, texV1), (texU2, texV1), (texU2, texV2), (texU, texV2) )
                )
        else:
            pass
        
    def generateDivs(self,
            building, item, unitVector, markupItemIndex1, markupItemIndex2, step,
            prevIndex1, prevIndex2, index1, index2, texU, texV1, texV2
        ):
        verts = building.verts
        v1 = verts[prevIndex1]
        v2 = verts[prevIndex2]
        for _i in range(markupItemIndex1, markupItemIndex2, step):
            _item = item.markup[_i]
            incrementVector = _item.width * unitVector
            v1 = v1 + incrementVector
            verts.append(v1)
            v2 = v2 + incrementVector
            verts.append(v2)
            texU2 = texU + _item.width
            self.getItemRenderer(_item).render(
                _item,
                (prevIndex1, index1, index2, prevIndex2),
                ( (texU, texV1), (texU2, texV1), (texU2, texV2), (texU, texV2) )
            )
            prevIndex1 = index1
            prevIndex2 = index2
            index1 = len(building.verts)
            index2 = index1 + 1
            texU = texU2
        return prevIndex1, prevIndex2, index1, index2, texU
    
    def generateLevelDiv(self,
            building, levelGroup, parentItem, renderer, height,
            prevIndex1, prevIndex2, index1, index2, texU1, texU2, texV
        ):
            verts = building.verts
            verts.append(verts[prevIndex1] + height*zAxis)
            verts.append(verts[prevIndex2] + height*zAxis)
            texV2 = texV + height
            renderer.render(
                building, levelGroup, parentItem,
                (prevIndex1, prevIndex2, index2, index1),
                ( (texU1, texV), (texU2, texV), (texU2, texV2), (texU1, texV2) ),
                texU1,
                texV
            )
            prevIndex1 = index1
            prevIndex2 = index2
            index1 += 2
            index2 = index1 + 1
            return prevIndex1, prevIndex2, index1, index2, texV2

    def setData(self, face, layerName, uv):
        if not isinstance(uv, tuple):
            uv = (uv, uv)
        uvLayer = self.r.bm.loops.layers.uv[layerName]
        for loop in face.loops:
            loop[uvLayer].uv = uv
            
    def setVertexColor(self, item, face):
        color = Manager.normalizeColor(item.getStyleBlockAttrDeep("claddingColor"))
        if color:
            color = Manager.getColor(color)
            vertexColorLayer = self.r.bm.loops.layers.color[self.vertexColorLayer]
            for loop in face.loops:
                loop[vertexColorLayer] = (color[0], color[1], color[2], 1.)
    
    def setMaterialId(self, item, building, buildingPart, itemRenderer):
        facadePatternInfo = self.facadePatternInfo
        if self.initFacadePatternInfo:
            # reset <facadePatternInfo>
            for key in facadePatternInfo:
                facadePatternInfo[key] = 0
            # initalize <facadePatternInfo>
            for _item in item.markup:
                className = _item.__class__.__name__
                if className in facadePatternInfo:
                    facadePatternInfo[className] += 1
        # get a texture that fits to the Level markup pattern
        facadeTextureInfo = self.r.facadeTextureStore.getTextureInfo(
            building,
            buildingPart,
            facadePatternInfo
        )
        
        claddingMaterial = item.getStyleBlockAttrDeep("claddingMaterial")
        claddingTextureInfo = self.getCladdingTextureInfo(claddingMaterial, building)
        
        if facadeTextureInfo:
            materialId = self.getFacadeMaterialId(facadeTextureInfo, claddingTextureInfo)
            if itemRenderer.createFacadeMaterial(materialId, facadeTextureInfo, claddingTextureInfo):
                item.materialId = materialId
            else:
                item.materialId = ""
        else:
            item.materialId = ""
    
    def render(self, building, levelGroup, parentItem, indices, uvs, texOffsetU, texOffsetV):
        face = self.r.createFace(building, indices, uvs)
        if levelGroup:
            item = levelGroup.item
            if item.materialId is None:
                if item.markup:
                    self.setMaterialId(
                        item,
                        building,
                        # getting building part
                        item.buildingPart if item.buildingPart else (
                            "groundlevel" if levelGroup.singleLevel and not levelGroup.index1 else "level"
                        ),
                        self
                    )
                else:
                    self.renderCladding(building, item, face)
            if item.materialId:
                self.setData(
                    face,
                    self.r.layer.uvNameSize,
                    (
                        # face width
                        parentItem.width,
                        self.getHeightForMaterial(levelGroup)
                    )
                )
                self.setData(
                    face,
                    self.uvLayer,
                    (
                        # offset for the texture U-coordinate
                        texOffsetU,
                        # offset for the texture V-coordinate
                        texOffsetV
                    )
                )
                self.setVertexColor(item, face)
            self.r.setMaterial(face, item.materialId)
        else:
            self.renderCladding(building, parentItem, face)
    
    def renderCladding(self, building, parentItem, face):
        materialId = ''
        claddingMaterial = parentItem.getStyleBlockAttrDeep("claddingMaterial")
        if claddingMaterial:
            claddingTextureInfo = self.getCladdingTextureInfo(claddingMaterial, building)
            if claddingTextureInfo:
                materialId = self.getCladdingMaterialId(claddingTextureInfo)
                self.createCladdingMaterial(materialId, claddingTextureInfo)
                self.setVertexColor(parentItem, face)
        self.r.setMaterial(face, materialId)
    
    def getFacadeMaterialId(self, facadeTextureInfo, claddingTextureInfo):
        return "%s_%s" % (facadeTextureInfo["name"], claddingTextureInfo["material"])\
            if claddingTextureInfo\
            else facadeTextureInfo["name"]
    
    def getCladdingMaterialId(self, claddingTextureInfo):
        return claddingTextureInfo["name"]
    
    def createFacadeMaterial(self, materialName, facadeTextureInfo, claddingTextureInfo):
        textureWidthPx = facadeTextureInfo["textureWidthPx"]
        textureHeightPx = facadeTextureInfo["textureHeightPx"]
        numberOfTilesU = facadeTextureInfo["numTilesU"]
        numberOfTilesV = facadeTextureInfo["numTilesV"]
        tileWidthPx = textureWidthPx/numberOfTilesU
        # factor = windowWidthM/windowWidthPx
        factor = facadeTextureInfo["windowWidthM"]/(facadeTextureInfo["windowRpx"]-facadeTextureInfo["windowLpx"])

        textureWidthM = factor*textureWidthPx
        tileSizeUdefaultM = factor*tileWidthPx
        textureUoffsetM = 0.
        
        textureLevelHeightM = factor*textureHeightPx/numberOfTilesV
        textureHeightM = factor*textureHeightPx
        textureVoffsetM = 0.
        
        materialTemplate = self.getMaterialTemplate(
            self.facadeMaterialTemplateFilename,
            self.facadeMaterialTemplateName
        )
        if not materialName in bpy.data.materials:
            bldgMaterialsDirectory = os.path.dirname(self.r.app.bldgMaterialsFilepath)
            nodes = createMaterialFromTemplate(materialTemplate, materialName)
            # the overlay texture
            setImage(
                facadeTextureInfo["name"],
                os.path.join(bldgMaterialsDirectory, facadeTextureInfo["path"]),
                nodes,
                "Overlay"
            )
            if claddingTextureInfo:
                # The wall material (i.e. background) texture,
                # set it just in case
                setImage(
                    claddingTextureInfo["name"],
                    os.path.join(bldgMaterialsDirectory, claddingTextureInfo["path"]),
                    nodes,
                    "Wall Material"
                )
                nodes["Mapping"].scale[0] = 1./claddingTextureInfo["textureWidthM"]
                nodes["Mapping"].scale[1] = 1./claddingTextureInfo["textureHeightM"]
            # the mask for the emission
            #setImage(fileName, directory, nodes, "Emission Mask", "emissive")
            # setting nodes
            n = nodes[_customNode]
            setCustomNodeValue(n, "Texture Width", textureWidthM)
            setCustomNodeValue(n, "Number of Tiles U", numberOfTilesU)
            setCustomNodeValue(n, "Tile Size U Default", tileSizeUdefaultM)
            setCustomNodeValue(n, "Texture U-Offset", textureUoffsetM)
            setCustomNodeValue(n, "Number of Tiles V", numberOfTilesV)
            setCustomNodeValue(n, "Texture Level Height", textureLevelHeightM)
            setCustomNodeValue(n, "Texture Height", textureHeightM)
            setCustomNodeValue(n, "Texture V-Offset", textureVoffsetM)
        return True
    
    def createCladdingMaterial(self, materialName, claddingTextureInfo):
        materialTemplate = self.getMaterialTemplate(
            _claddingMaterialTemplateFilename,
            _claddingMaterialTemplateName
        )
        if not materialName in bpy.data.materials:
            bldgMaterialsDirectory = os.path.dirname(self.r.app.bldgMaterialsFilepath)
            nodes = createMaterialFromTemplate(materialTemplate, materialName)
            # The wall material (i.e. background) texture,
            # set it just in case
            setImage(
                claddingTextureInfo["name"],
                os.path.join(bldgMaterialsDirectory, claddingTextureInfo["path"]),
                nodes,
                "Cladding Texture"
            )
            nodes["Mapping"].scale[0] = 1./claddingTextureInfo["textureWidthM"]
            nodes["Mapping"].scale[1] = 1./claddingTextureInfo["textureHeightM"]
        # return True for consistency with <self.getFacadeMaterialId(..)>
        return True
    
    def getCladdingTextureInfo(self, claddingMaterial, building):
        if claddingMaterial:
            if claddingMaterial in building._cache:
                claddingTextureInfo = building._cache[claddingMaterial]
            else:
                claddingTextureInfo = self.r.claddingTextureStore.getTextureInfo(claddingMaterial)
                building._cache[claddingMaterial] = claddingTextureInfo
        else:
            claddingTextureInfo = None
        return claddingTextureInfo