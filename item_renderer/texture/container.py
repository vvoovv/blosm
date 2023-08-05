from math import floor
from . import ItemRendererTexture
from .corner import Corner
from grammar.arrangement import Horizontal, Vertical
from grammar.symmetry import MiddleOfLast, RightmostOfLast
from util import rgbToHex
from util.blender import createMeshObject, addGeometryNodesModifier, useAttributeForGnInput


def _getTileWidthM(facadeTextureInfo):
    if not "tileWidthM" in facadeTextureInfo:
        # cache <tileWidthM>
        facadeTextureInfo["tileWidthM"] = facadeTextureInfo["textureSize"][0]/\
            (facadeTextureInfo["featureRpx"] - facadeTextureInfo["featureLpx"])*\
            facadeTextureInfo["featureWidthM"] / facadeTextureInfo["numTilesU"]
    return facadeTextureInfo["tileWidthM"]


class RenderState:
    """
    The class instance hold
    """
    
    def __init__(self):
        #
        # Variables for <Container.renderLevels(..)>
        #
        # <indexBL> and <indexBR> are indices of the bottom vertices of an level item to be created
        # in <Container.renderLevels(..)> out of the parent item
        # The prefix <BL> means "bottom left"
        self.indexBL = 0
        # The prefix <BR> means "bottom rights"
        self.indexBR = 0
        # <texVb> is the current V-coordinate for texturing the bottom vertices of
        # level items to be created in <Container.renderLevels(..)> out of the parent item
        self.texVb = 0
        
        #
        # Variables for <Container.renderDivs(..)>
        #
        # <indexLB> and <indexLT> are indices of the leftmost vertices of an item to be created
        # The prefix <LB> means "left bottom"
        self.indexLB = 0
        # The prefix <LT> means "left top"
        self.indexLT = 0
        # <texUl> is the current U-coordinate for texturing the leftmost vertices of
        # items to be created out of <item>
        self.texUl = 0.
        # <texVlt> is the current V-coordinate for texturing the top leftmost vertex of
        # items to be created out of <item>
        self.texVlt = 0.
        # <startIndex> is only used by geometry <TrapezoidChainedRv>.
        # It's auxiliary variable used for optimization purposes. The task is to find intersection of
        # the uppper part of the chained trapezoid with the vertical line with the given X-coordinate.
        # <startIndex> helps to limit the search.
        # One starts with <startIndex> = <n> - 1, where <n> is the number of vertices in
        # the chained trapezoid, i.e. it points to the top leftmost vertex of the chained trapezoid.
        # Then <startIndex> is gradually decreased, i.e. <startIndex> points to a vertex to the left
        # from the previous vertex.
        self.startIndex = 0
        
        self.tmpTriangle = True


def getParamValue(param, item, obj):
    return item.getStyleBlockAttrDeep(param) or obj[param]


def getParamValueGn(gnInput, item, obj):
    # isPmlParamater, mAttr, (pmlParameter, inputType) = gnInput
    return item.getStyleBlockAttrDeep(gnInput[2][0]) or obj.modifiers[0][gnInput[1]]


def getStringForGnKey(gnInput, item, obj):
    gnInputType = gnInput[2][1]
    value = getParamValueGn(gnInput, item, obj)
    
    if gnInputType == 'VALUE':
        return str(round( value, 3 ))
    elif gnInputType == 'STRING':
        return value
    elif gnInputType == 'RGBA':
        return rgbToHex(value)
    elif gnInputType == 'BOOLEAN':
        return "1" if value else '0'
    elif gnInputType == 'INT':
        return str(value)


class Container(ItemRendererTexture):
    """
    The base class for the item renderers Facade, Div, Layer, Bottom
    """
    
    def __init__(self, exportMaterials):
        super().__init__(exportMaterials)
        self.claddingTexture = True
        self.renderState = RenderState()
    
    def renderMarkup(self, item):
        item.prepareMarkupItems()
        
        if item.styleBlock.markup[0].isLevel:
            #face = self.r.createFace(item.footprint, item.indices)
            #self.renderCladding(item, face, item.uvs)
            #return
            if item.highEnoughForLevel:
                self.renderLevels(item)
            else:
                # No space for levels, so we render cladding only.
                self.renderCladding(
                    item,
                    self.r.createFace(item.footprint, item.indices),
                    item.uvs
                )
        else:
            self.renderDivs(item)
        if not item.valid:
            return
    
    def renderLevels(self, item):
        item.levelGroups.init()
        
        for _item in item.markup:
            _item.prepareMarkupItems()
            # inherit the width from <item> to the markup items
            _item.width = item.width
            if _item.markup:
                _item.calculateMarkupDivision(self.r)
        
        item.geometry.renderLevelGroups(item, self)
    
    def renderDivs(self, item, levelGroup):
        # If <levelGroup> is given, that actually means that <item> is a level or contained
        # inside another level item. In this case the call to <self.renderLevelGroup(..)>
        # will be made later in the code
        
        # <r> is the global building renderer
        r = self.r
        parentIndices = item.indices
        geometry = item.geometry
        
        if item.arrangement is Horizontal:
            # get markup width and number of repeats
            #item.calculateMarkupDivision(r.assetStore)
            if not item.valid:
                return
            # create vertices for the markup items
            numItems = len(item.markup)
            if numItems == 1:
                # the special case
                _item = item.markup[0]
                _item.indices = parentIndices
                _item.uvs = item.uvs
                r.createFace(item.footprint, _item.indices)
            else:
                numRepeats = item.numRepeats
                symmetry = item.symmetry
                rs = self.renderState
                geometry.initRenderStateForDivs(rs, item)
                
                # Generate Div items but the last one;
                # the special case is when a symmetry is available
                if numRepeats>1:
                    for _ in range(numRepeats-1):
                        geometry.renderDivs(
                            self, item, levelGroup,
                            0, numItems, 1,
                            rs
                        )
                        if symmetry:
                            geometry.renderDivs(
                                self, item, levelGroup,
                                numItems-2 if symmetry is MiddleOfLast else numItems-1, -1, -1,
                                rs
                            )
                geometry.renderDivs(
                    self, item, levelGroup,
                    0, numItems if symmetry else numItems-1, 1,
                    rs
                )
                if symmetry:
                    geometry.renderDivs(
                        self, item, levelGroup,
                        numItems-2 if symmetry is MiddleOfLast else numItems-1, 0, -1,
                        rs
                    )
                # process the last item
                lastItem = item.markup[0] if symmetry else item.markup[-1]
                if lastItem.width:
                    geometry.renderLastDiv(
                        self, item, levelGroup, lastItem,
                        rs
                    )
        else:
            pass

    def setData(self, face, layerName, uv):
        if not isinstance(uv, tuple):
            uv = (uv, uv)
        uvLayer = self.r.bm.loops.layers.uv[layerName]
        for loop in face.loops:
            loop[uvLayer].uv = uv
    
    def setMaterialId(self, item, facadeTextureInfo, uvs):
        claddingTextureInfo = self.getCladdingTextureInfo(item)\
            if facadeTextureInfo.get("cladding") and self.claddingTexture else\
            None
        
        materialId = self.getFacadeMaterialId(item, facadeTextureInfo, claddingTextureInfo)
        if self.createFacadeMaterial(item, materialId, facadeTextureInfo, claddingTextureInfo, uvs):
            item.materialId = materialId
            item.materialData = facadeTextureInfo, claddingTextureInfo
        else:
            item.materialId = ""
    
    def renderLevelGroup(self, item, levelGroup, indices, uvs):
        """
        Args:
            item (item.container.Container): It's needed to get <building> and
                width for the placeholder of <levelGroup>. Example configurations:
                facade.level (<referenceItem> is facade)
                facade.div.level (<referenceItem> is div)
                facade.level.div (<referenceItem> is div)
                facade.div.level.div (<referenceItem> is the second div)
            levelGroup: level group
            indices (list or tuple): indices of vertices
            uvs (list or tuple): UV-coordinates of the vertices
        """
        assetInfo = None
        
        if levelGroup.item:
            # asset info could have been set in the call to item.getWidth(..)
            assetInfo = item.assetInfo
            # if <assetInfo=0>, then it was already queried in the asset store and nothing was found
            if assetInfo is None:
                assetInfo = self.getAssetInfo(item)
            
            if assetInfo:
                if assetInfo["type"] == "mesh":
                    #
                    # mesh
                    #
                    
                    if "object" in assetInfo:
                        #
                        # a separate Blender object
                        #
                        
                        # first we check for corner modules (mesh only) defined implicitly
                        if floor(item.width/assetInfo["tileWidthM"]) > 1:
                            if item.cornerL or item.cornerR:
                                assetInfoCorner = self.getAssetInfoCorner(item, assetInfo["class"])
                                if assetInfoCorner:
                                    self.processImplicitCornerItems(item, levelGroup, indices, assetInfo, assetInfoCorner)
                                    return
                        
                        objName = assetInfo["object"]
                        
                        if not self.processModuleObject(objName, assetInfo):
                            return
                    else:
                        #
                        # a Blender collection (e.g. of corner modules)
                        #
                        obj = self.processCollection(item, assetInfo, indices)
                        if not obj:
                            # something is wrong or no actions are needed
                            return
                        objName = obj.name
                        if not objName in self.r.meshAssets:
                            self.processAssetMeshObject(obj, objName)
                    
                    self.prepareGnVerts(
                        item, levelGroup, indices, assetInfo,
                        self.getGnInstanceObject(item, objName)
                    )
                else:
                    #
                    # texture
                    #
                    face = self.r.createFace(item.footprint, indices)
                    
                    if item.materialId is None:
                        self.setMaterialId(
                            item,
                            assetInfo,
                            uvs
                        )
                    if item.materialId:
                        facadeTextureInfo, claddingTextureInfo = item.materialData
                        layer = item.building.element.l
                        self.r.setUvs(
                            face,
                            self.getUvs(item, levelGroup, facadeTextureInfo, uvs),
                            layer,
                            layer.uvLayerNameFacade
                        )
                        self.renderExtra(item, face, facadeTextureInfo, claddingTextureInfo, uvs)
                        self.r.setMaterial(layer, face, item.materialId)
                    else:
                        self.renderCladding(item, face, uvs)
        
        if not assetInfo:
            self.renderCladding(
                item,
                self.r.createFace(item.footprint, indices),
                uvs
            )
            item.materialId = ""
    
    def getGnInstanceObject(self, item, objName):
        obj, objParams, gnInputs, instances = self.r.meshAssets[objName]
        
        if objParams or gnInputs:
            # create a key out of <objParams> and <gnInputs>
            objKey = gnKey = ''
            if objParams:
                objKey = '_'.join(
                    str(paramIndex) + '_' + rgbToHex( getParamValue(param, item, obj) )\
                        for paramIndex, param in enumerate(objParams)
                )
            numObjParams = len(objParams)
            if gnInputs:
                gnKey = '_'.join(
                    str(numObjParams+paramIndex) + '_' + getStringForGnKey(gnInput, item, obj)\
                        for paramIndex, gnInput in enumerate(gnInputs) if gnInput[0]
                )
            key = objKey + "_" +gnKey if objKey and gnKey else objKey or gnKey
            if not key in instances:
                # the created Blender object <_obj> that shares the mesh data with <obj>
                _obj = instances[key] = createMeshObject(objName, mesh = obj.data, collection = self.r.buildingAssetsCollection)
                #
                # set object properties for <_obj>
                #
                if objParams:
                    for param in objParams:
                        _obj[param] = getParamValue(param, item, obj)
                #
                # now deal with Geometry Nodes
                #
                if gnInputs:
                    m = obj.modifiers[0]
                    # create a Geometry Nodes modifier
                    _m = addGeometryNodesModifier(_obj, m.node_group, '')
                    for isPmlParamater, mAttr, extraData in gnInputs:
                        if isPmlParamater:
                            #
                            # pmlParameter = extraData[0]
                            #
                            itemAttrValue = item.getStyleBlockAttrDeep(extraData[0])
                            if itemAttrValue:
                                if extraData[1] == 'RGBA':
                                    # At the moment  it isn't possible to set a color for
                                    # a Geometry Nodes attribute through
                                    # _m[mAttr] = itemAttrValue
                                    # We have to set the color like it's done below:
                                    _m[mAttr][0], _m[mAttr][1], _m[mAttr][2], _m[mAttr][3] =\
                                    itemAttrValue[0], itemAttrValue[1], itemAttrValue[2], itemAttrValue[3]
                                else:
                                    _m[mAttr] = itemAttrValue
                            else:
                                _m[mAttr] = m[mAttr]
                        else:
                            #
                            # useDataAttribute = dataAttribute = extraData
                            #
                            if extraData:
                                # If <mAttr> in <obj> uses an <obj>'s attribute defined in <obj>'s data, than
                                # <mAttr> in <_obj> also uses the <_obj>'s attribute defined in <_obj>'s data.
                                # Note, that <obj> and <_obj> use the same data.
                                useAttributeForGnInput(_m, mAttr, extraData)
                            else:
                                _m[mAttr] = m[mAttr]
                         
            obj = instances[key]
        
        return obj
    
    def getUvs(self, item, levelGroup, facadeTextureInfo, uvs):
        return item.geometry.getFinalUvs(
            max( round(item.width/_getTileWidthM(facadeTextureInfo)), 1 ),
            self.getNumLevelsInFace(levelGroup),
            facadeTextureInfo["numTilesU"],
            facadeTextureInfo["numTilesV"],
            uvs
        )
    
    def getNumLevelsInFace(self, levelGroup):
        return 1 if levelGroup.singleLevel else (levelGroup.index2-levelGroup.index1+1)
    
    def processAssetMeshObject(self, obj, objName):
        # We need only the properties of a Blender object created by a designer.
        # <rna_properties> is used to filter our the other properties
        rna_properties = {
            prop.identifier for prop in obj.bl_rna.properties if prop.is_runtime
        }
        objParams = [_property[2:] for _property in obj.keys() if not _property in rna_properties and _property.startswith("p_")]
        
        # now get the properties from the Geometry Nodes modifier
        gnInputs = []
        if obj.modifiers:
            # only a single Geometry Nodes modifier is allowed!
            m = obj.modifiers[0]
            inputs = m.node_group.inputs
            
            mAttrs = list(obj.modifiers[0].keys())
            attrIndex = 0
            # the number of inputs with the name starting with <_p>
            numGnParams = sum(inputs[inputIndex].name.startswith("p_") for inputIndex in range(1, len(inputs)))
            if numGnParams:
                for inputIndex in range(1, len(inputs)):
                    inp = inputs[inputIndex]
                    mAttr = mAttrs[attrIndex]
                    if inp.name.startswith("p_"):
                        gnInputs.append((
                            True, # <True> since it represents PML parameter, because it starts with <p_>
                            mAttr, # the related modifier's attribute
                            (inp.name[2:], inp.type) # PML parameter, input type
                        ))
                        attrIndex += 3 if mAttr + "_use_attribute" in m else 1
                    else:
                        useDataAttribute = m.get(mAttr + "_use_attribute")
                        gnInputs.append((
                            False, # <False> since it doesn't represent a PML parameter, i.e. it doesn't start with <p_>
                            mAttr, # the related modifier's attribute
                            m[mAttr + "_attribute_name"] if useDataAttribute else None # the name of the data attribute or None
                        ))
                        attrIndex += 1 if useDataAttribute is None else 3
        
        self.r.meshAssets[objName] = (obj, objParams, gnInputs, {} if objParams or gnInputs else None)
        
        if not objParams and not gnInputs:
            # use <obj> as is (i.e. linked from another Blender file)
            self.r.buildingAssetsCollection.objects.link(obj)
    
    def prepareGnVerts(self, item, levelGroup, indices, assetInfo, obj):
        layer = item.building.element.l
        
        tileWidth = assetInfo["tileWidthM"]
        
        numTilesX = max( floor(item.width/tileWidth), 1 )
        numTilesY = 1 if levelGroup.singleLevel else levelGroup.index2 - levelGroup.index1 + 1
        scaleX = item.width/(numTilesX*tileWidth)
        scaleY = levelGroup.levelHeight/assetInfo["tileHeightM"]
        
        tileWidth *= scaleX
        
        # increment along X-axis of <item>
        incrementVector = tileWidth * item.facade.vector.unitVector3d
        
        bmVerts = layer.bmGn.verts
        attributeValuesGn = layer.attributeValuesGn
        
        _vertLocation = item.building.renderInfo.verts[indices[0]] + 0.5*incrementVector
        if numTilesY == 1:
            for _ in range(numTilesX):
                bmVerts.new(_vertLocation)
                _vertLocation += incrementVector
                attributeValuesGn.append((
                    obj.name,
                    item.facade.vector.vector3d,
                    scaleX,
                    scaleY
                ))
        else:
            for _ in range(numTilesY):
                vertLocation = _vertLocation.copy()
                for _ in range(numTilesX):
                    bmVerts.new(vertLocation)
                    attributeValuesGn.append((
                        obj.name,
                        item.facade.vector.vector3d,
                        scaleX,
                        scaleY
                    ))
                    vertLocation += incrementVector
                _vertLocation[2] += levelGroup.levelHeight
    
    def processImplicitCornerItems(self, item, levelGroup, indices, assetInfo, assetInfoCorner):
        # the same code as in self.prepareGnVerts(..)
        # the beginning of the code >
        layer = item.building.element.l
        
        tileWidth = assetInfo["tileWidthM"]
        
        numTilesX = max( floor(item.width/tileWidth), 1 )
        numTilesY = 1 if levelGroup.singleLevel else levelGroup.index2 - levelGroup.index1 + 1
        scaleX = item.width/(numTilesX*tileWidth)
        scaleY = levelGroup.levelHeight/assetInfo["tileHeightM"]
        
        tileWidth *= scaleX
        
        # increment along X-axis of <item>
        incrementVector = tileWidth * item.facade.vector.unitVector3d
        
        bmVerts = layer.bmGn.verts
        attributeValuesGn = layer.attributeValuesGn
        # the end of the code <
        
        if item.cornerL:
            self._processImplicitCornerItem(
                item, levelGroup, assetInfoCorner, True,
                item.building.renderInfo.verts[indices[0]] + incrementVector,
                numTilesY
            )
        if item.cornerR:
            self._processImplicitCornerItem(
                item, levelGroup, assetInfoCorner, False,
                item.building.renderInfo.verts[indices[1]] - incrementVector,
                numTilesY
            )
        
        objName = assetInfo["object"]
        
        if not self.processModuleObject(objName, assetInfo):
            return
        
        obj = self.getGnInstanceObject(item, objName)
        
        _vertLocation = item.building.renderInfo.verts[indices[0]] + (1.5 if item.cornerL else 0.5) *incrementVector
        if numTilesY == 1:
            for _ in range(1 if item.cornerL else 0, numTilesX-1 if item.cornerR else numTilesX):
                # the same code as in self.prepareGnVerts(..)
                bmVerts.new(_vertLocation)
                _vertLocation += incrementVector
                attributeValuesGn.append((
                    obj.name,
                    item.facade.vector.vector3d,
                    scaleX,
                    scaleY
                ))
        else:
            for _ in range(numTilesY):
                vertLocation = _vertLocation.copy()
                for _ in range(1 if item.cornerL else 0, numTilesX-1 if item.cornerR else numTilesX):
                    # the same code as in self.prepareGnVerts(..)
                    bmVerts.new(vertLocation)
                    attributeValuesGn.append((
                        obj.name,
                        item.facade.vector.vector3d,
                        scaleX,
                        scaleY
                    ))
                    vertLocation += incrementVector
                _vertLocation[2] += levelGroup.levelHeight
    
    def getAssetInfoCorner(self, item, baseClass):
        building, group =\
            item.building, item.getStyleBlockAttrDeep("group")
        
        return self.app.assetStore.getAssetInfo(
            True,
            building,
            group,
            "corner",
            baseClass + "_corner"
        )
    
    def _processImplicitCornerItem(self, item, levelGroup, assetInfoCorner, cornerL, cornerVert, numTilesY):
        obj = Corner.processCorner(self, item, assetInfoCorner, cornerL, cornerVert)
        if obj:
            objName = obj.name
            if not objName in self.r.meshAssets:
                self.processAssetMeshObject(obj, objName)
            
            obj = self.getGnInstanceObject(item, objName)
            Corner.prepareGnVerts(self, item, levelGroup, None, assetInfoCorner, obj, numTilesY)
            
            return obj