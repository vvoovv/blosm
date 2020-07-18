import math
from grammar import *
from grammar.scope import PerBuilding, PerFootprint
from grammar import units, symmetry, smoothness
from grammar.value import Value, FromAttr, Alternatives, Conditional, FromStyleBlockAttr, Constant
from grammar.value import RandomWeighted, RandomNormal
from action.volume.roof import Roof as RoofDefs
from item.defs import RoofShapes, CladdingMaterials


minHeightForLevels = 1.5
minWidthForOpenings = 1.


styles = {
"mid rise apartments zaandam": [
    Meta(
        buildingUse = "apartments",
        buildingLaf = "modern",
        height = "mid rise"
    ),
    Footprint(
        height = Value(FromAttr("height", FromAttr.Float, FromAttr.Positive)),
        minHeight = Value(FromAttr("min_height", FromAttr.Float, FromAttr.Positive)),
        numLevels = Value(Alternatives(
            FromAttr("building:levels", FromAttr.Integer, FromAttr.Positive),
            RandomWeighted(( (4, 10), (5, 40), (6, 10) ))
        )),
        hasNumLevelsAttr = Value(FromAttr("building:levels", FromAttr.Integer, FromAttr.Positive)),
        #numRoofLevels = 1,
        minLevel = Value(Alternatives(
            FromAttr("building:min_level", FromAttr.Integer, FromAttr.NonNegative),
            Constant(0)
        )),
        #topHeight = Value( RandomNormal(1.) ),
        topHeight = 0.,
        #lastLevelHeight = PerBuilding( Value( RandomNormal(0.7*3.) ) ),
        levelHeight = Value( RandomNormal(3.) ),
        groundLevelHeight = Value( RandomNormal(1.4*3) ),
        bottomHeight = Value( RandomNormal(1.) ),
        roofShape = Value(Alternatives(
            #FromAttr("roof:shape", FromAttr.String, RoofShapes),
            #Constant("dome"),
            Constant("flat"),
            Constant("saltbox")
            #RandomWeighted(( ("gabled", 10), ("flat", 40) ))
        )),
        roofHeight = Value(Alternatives(
            FromAttr("roof:height", FromAttr.Float, FromAttr.NonNegative),
            Constant(5)#RandomNormal(3.7)
        )),
        roofAngle = Value(FromAttr("roof:angle", FromAttr.Float)),
        roofDirection = Value(Alternatives(
            FromAttr("roof:direction", FromAttr.String, RoofDefs.directions),
            FromAttr("roof:direction", FromAttr.Float),
            FromAttr("roof:slope:direction", FromAttr.String, RoofDefs.directions),
            FromAttr("roof:slope:direction", FromAttr.Float)
        )),
        roofOrientation = "across",#Value( FromAttr("roof:orientation", FromAttr.String) ),
        lastLevelOffsetFactor = Value(RandomWeighted((
            (0., 50), (0.05, 3), (0.1, 5), (0.15, 5), (0.2, 5), (0.25, 5), (0.3, 5),
            (0.35, 5), (0.4, 5), (0.45, 5), (0.5, 3), (0.55, 2), (0.6, 2)
        ))),
        claddingColor = PerBuilding(Value(RandomWeighted((
            ((0.647, 0.165, 0.165, 1.), 1), # brown
            ((0.565, 0.933, 0.565, 1.), 1), # lightgreen
            ((1., 0.855, 0.725, 1.), 1) # peachpuff
        )))),
        claddingMaterial = "brick"
        #claddingMaterial = Value(RandomWeighted((
        #    ("brick", 1), ("plaster", 1), ("gravel", 1)
        #)))
    ),
    Facade(
        defName = "brown_brick"
    ),
    Level(
        defName = "level_window_balcony",
        markup = [
            Window(
                width = 1.8,
                height = 2.1,
                rows = 1,
                panels = 2
            ),
            Balcony()
        ]
    ),
    Level(
        defName = "staircase",
        offset = (0.5, units.Level)
    ),
    Window(
        defName = "back_facade_window",
        width = 1.2,
        height = 1.8,
        panels = 1
    ),
    Window(
        defName = "roof_window",
        width = 0.8,
        height = 0.8,
        rows = 1,
        panels = 1
    ),
    Div(
        defName = "window_and_balcony",
        label = "Window and Balcony",
        markup = [
            Level(
                use = ("level_window_balcony",),
                indices = (4, -1),
                claddingMaterial = "plaster",
                claddingColor = (0., 0., 1., 1.) # blue
            ),
            Level(
                use = ("level_window_balcony",),
                indices = (3, 3),
                claddingColor = (0., 0.502, 0., 1.) # green
            ),
            Level(
                use = ("level_window_balcony",),
                indices = (0, 2)
            ),
            Bottom(
                markup = [
                    Window(
                        width = 1.,
                        height = 1.,
                        rows = 1,
                        panels = 1
                    )
                ]
            )
        ]
    ),
    Div(
        defName = "staircase",
        label = "Staircase",
        bottomHeight = 0,
        markup = [
            Level(
                repeat = False,
                indices = (1, -1),
                markup = [
                    Window(
                        width = 0.8,
                        height = 0.8,
                        rows = 1,
                        panels = 1
                    )
                ]
            ),
            Level(
                indices = (0, 0),
                markup = [
                    Door(label = "entrance door")
                ]
            )
        ]
    ),
    Div(
        defName = "roof_side",
        width = useFrom("main_section"),
        symmetry = symmetry.RightmostOfLast
    ),
    Facade(
        label = "cladding only for too low structures",
        condition = lambda facade: facade.footprint.height - facade.footprint.minHeight < minHeightForLevels
    ),
    Facade(
        use = ("brown_brick",),
        label = "front facade",
        condition = lambda facade: facade.front,
        # None or symmetry.MiddleOfLast or symmetry.RightmostOfLast
        symmetry = symmetry.MiddleOfLast,
        # flip items for the total symmetry or leave them intact
        symmetryFlip = True,
        markup = [
            Div(
                use = ("window_and_balcony",),
                id = "main_section",
                label = "Window and Balcony"
            ),
            Div(
                use = ("staircase",),
                label = "Staircase"
            )
        ] 
    ),
    Facade(
        use = ("brown_brick",),
        label = "back facade",
        condition = lambda facade: facade.back,
        markup = [
            Level(
                indices = (0, -1),
                markup = [
                    Balcony(),
                    Window(use = ("back_facade_window",)),
                    Window(use = ("back_facade_window",))
                ]
            )
        ]
    ),
    Roof(
        roofCladdingMaterial = "brick",
        roofCladdingColor = (0.98, 0.502, 0.447, 1.), # salmon
        faces = smoothness.Smooth,
        #sharpEdges = smoothness.Side
    ),
    RoofSide(
        condition = lambda side: side.front,
        markup = [
            Div(
                use = ("roof_side",),
                markup = [
                    # openable skylight or roof window
                    Window(use = ("roof_window",)),
                    Window(use = ("roof_window",))
                ]
            )
        ]
    ),
    RoofSide(
        condition = lambda side: side.back,
        markup = [
            Div(
                use = ("roof_side",),
                markup = [
                    Dormer(), Dormer()
                ]
            )
        ]
    ),
    Ridge(
        markup = [
            Div(
                width = useFrom("main_section"),
                repeat = False,
                markup = [
                    Chimney()
                ]
            )
        ]    
    )
],
"high rise mirrored glass": [
    Meta(
        buildingUse = "office",
        buildingLaf = "curtain_wall",
        height = "high rise"
    ),
    Footprint(
        height = Value(FromAttr("height", FromAttr.Float, FromAttr.Positive)),
        minHeight = Value(FromAttr("min_height", FromAttr.Float, FromAttr.Positive)),
        hasNumLevelsAttr = Value(FromAttr("building:levels", FromAttr.Integer, FromAttr.Positive)),
        numLevels = Value(Alternatives(
            FromAttr("building:levels", FromAttr.Integer, FromAttr.Positive),
            RandomWeighted(( (4, 10), (5, 40), (6, 10) ))
        )),
        minLevel = Value(Alternatives(
            FromAttr("building:min_level", FromAttr.Integer, FromAttr.NonNegative),
            Constant(0)
        )),
        #topHeight = Value( RandomNormal(1.) ),
        topHeight = 0.,
        #lastLevelHeight = PerBuilding( Value( RandomNormal(0.7*3.) ) ),
        levelHeight = Value( RandomNormal(3.) ),
        #groundLevelHeight = Value( RandomNormal(1.4*3) ),
        #bottomHeight = Value( RandomNormal(1.) ),
        roofShape = Value(Alternatives(
            FromAttr("roof:shape", FromAttr.String, RoofShapes),
        )),
        claddingColor = PerBuilding(Value(RandomWeighted((
            ((0.647, 0.165, 0.165, 1.), 1), # brown
            ((0.565, 0.933, 0.565, 1.), 1), # lightgreen
            ((1., 0.855, 0.725, 1.), 1) # peachpuff
        )))),
        claddingMaterial = "glass"
    ),
    Facade(
        label = "cladding only for too low structures",
        condition = lambda facade: facade.footprint.height - facade.footprint.minHeight < minHeightForLevels
    ),
    Facade(
        label = "cladding only for structures without levels",
        condition = lambda facade: not facade.footprint.numLevels
    ),
    Facade(
        markup = [
            CurtainWall(
                indices = (0, -1)#,
                #width = 1.
            )
        ]
    ),
    Roof(
        roofCladdingMaterial = "concrete",
        roofCladdingColor = (0.98, 0.502, 0.447, 1.), # salmon
        faces = smoothness.Smooth,
        #sharpEdges = smoothness.Side
    )
],
"high rise": [
    Meta(
        buildingUse = "office",
        buildingLaf = "modern",
        height = "high rise"
    ),
    Footprint(
        height = Value(FromAttr("height", FromAttr.Float, FromAttr.Positive)),
        minHeight = Value(FromAttr("min_height", FromAttr.Float, FromAttr.Positive)),
        hasNumLevelsAttr = Value(FromAttr("building:levels", FromAttr.Integer, FromAttr.Positive)),
        numLevels = Value(Alternatives(
            FromAttr("building:levels", FromAttr.Integer, FromAttr.Positive),
            RandomWeighted(( (4, 10), (5, 40), (6, 10) ))
        )),
        minLevel = Value(Alternatives(
            FromAttr("building:min_level", FromAttr.Integer, FromAttr.NonNegative),
            Constant(0)
        )),
        #topHeight = Value( RandomNormal(1.) ),
        topHeight = 0.,
        #lastLevelHeight = PerBuilding( Value( RandomNormal(0.7*3.) ) ),
        levelHeight = Value( RandomNormal(3.) ),
        #groundLevelHeight = Value( RandomNormal(1.4*3) ),
        #bottomHeight = Value( RandomNormal(1.) ),
        roofShape = Value(Alternatives(
            FromAttr("roof:shape", FromAttr.String, RoofShapes),
            Constant("flat")
        )),
        roofHeight = Value(FromAttr("roof:height", FromAttr.Float, FromAttr.NonNegative)),
        claddingMaterial = PerBuilding(Value(Alternatives(
            FromAttr("building:material", FromAttr.String, CladdingMaterials),
            RandomWeighted(( ("brick", 1), ("plaster", 1) ))
        ))),
        claddingColor = PerBuilding(Value(Alternatives(
            FromAttr("building:colour", FromAttr.Color),
            Conditional(
                lambda footprint: footprint.getStyleBlockAttr("claddingMaterial") == "brick",
                RandomWeighted((
                    ((0.647, 0.165, 0.165, 1.), 1), # brown
                    ((0.98, 0.502, 0.447, 1.), 1), # salmon
                    ((0.502, 0., 0., 1.), 1) # maroon
                ))
            ),
            Conditional(
                lambda footprint: footprint.getStyleBlockAttr("claddingMaterial") == "plaster",
                RandomWeighted((
                    ((1., 0.627, 0.478, 1.), 1), # lightsalmon
                    ((0.565, 0.933, 0.565, 1.), 1), # lightgreen
                    ((1., 0.855, 0.725, 1.), 1) # peachpuff
                ))
            ),
            Conditional(
                lambda footprint: footprint.getStyleBlockAttr("claddingMaterial") == "glass",
                RandomWeighted((
                    ((0.306, 0.447, 0.573, 1.), 1),
                    ((0.169, 0.318, 0.361, 1.), 1),
                    ((0.094, 0.18, 0.271, 1.), 1)
                ))
            )
        )))
    ),
    Facade(
        label = "cladding only for structures without levels or too low structures or too narrow facades",
        condition = lambda facade: not facade.footprint.numLevels or\
            facade.footprint.height - facade.footprint.minHeight < minHeightForLevels or\
            facade.width < minWidthForOpenings
    ),
    Facade(
        condition = lambda item: item.footprint.getStyleBlockAttr("claddingMaterial") == "glass",
        markup = [
            CurtainWall(
                indices = (0, -1)
            )
        ]
    ),
    Facade(
        markup = [
            Level(
                indices = (0, -1)#,
                #width = 1.
            )
        ]
    ),
    Roof(
        roofCladdingMaterial = Value(Alternatives(
            FromAttr("roof:material", FromAttr.String, CladdingMaterials),
            Conditional(
                lambda roof: roof.footprint.getStyleBlockAttr("roofShape") == "flat",
                Constant("concrete")
            ),
            # roofShape in ("pyramidal", "dome", "half-dome", "onion")
            Constant("metal")
        )),
        roofCladdingColor = Value(Alternatives(
            FromAttr("roof:colour", FromAttr.Color),
            Conditional(
                lambda roof: roof.getStyleBlockAttr("roofCladdingMaterial") == "concrete",
                RandomWeighted((
                    ((0.686, 0.686, 0.686, 1.), 1),
                    ((0.698, 0.698, 0.651, 1.), 1),
                    ((0.784, 0.761, 0.714, 1.), 1)
                ))
            ),
            # roofCladdingMaterial == "metal"
            RandomWeighted((
                ((0.686, 0.686, 0.686, 1.), 1),
                ((0.698, 0.698, 0.651, 1.), 1),
                ((0.784, 0.761, 0.714, 1.), 1)
            ))
        )),
        faces = Value(Conditional(
            lambda item: item.footprint.getStyleBlockAttr("roofShape") in ("dome", "half-dome", "onion"),
            Constant(smoothness.Smooth)
        ))
        #sharpEdges = smoothness.Side
    )
],
"place of worship": [
    Footprint(
        height = Value(FromAttr("height", FromAttr.Float, FromAttr.Positive)),
        minHeight = Value(FromAttr("min_height", FromAttr.Float, FromAttr.Positive)),
        numLevels = 0,
        roofShape = Value(Alternatives(
            FromAttr("roof:shape", FromAttr.String, RoofShapes),
            Constant("flat")
        )),
        roofHeight = Value(FromAttr("roof:height", FromAttr.Float, FromAttr.NonNegative)),
        claddingMaterial = PerBuilding(Value(Alternatives(
            FromAttr("building:material", FromAttr.String, CladdingMaterials),
            Constant("plaster")
        ))),
        claddingColor = PerBuilding(Value(Alternatives(
            FromAttr("building:colour", FromAttr.Color),
            RandomWeighted((
                ((1., 0.627, 0.478, 1.), 1), # lightsalmon
                ((0.565, 0.933, 0.565, 1.), 1), # lightgreen
                ((1., 0.855, 0.725, 1.), 1) # peachpuff
            ))
        )))
    ),
    Facade(),
    Roof(
        roofCladdingMaterial = Value(Alternatives(
            FromAttr("roof:material", FromAttr.String, CladdingMaterials),
            Constant("metal")
        )),
        roofCladdingColor = Value(Alternatives(
            FromAttr("roof:colour", FromAttr.Color),
            RandomWeighted((
                ((0.686, 0.686, 0.686, 1.), 1),
                ((0.698, 0.698, 0.651, 1.), 1),
                ((0.784, 0.761, 0.714, 1.), 1)
            ))
        )),
        faces = Value(Conditional(
            lambda item: item.footprint.getStyleBlockAttr("roofShape") in ("dome","half-dome", "onion"),
            Constant(smoothness.Smooth)
        ))
        #sharpEdges = smoothness.Side
    )
],
"man made": [
    Footprint(
        height = Value(FromAttr("height", FromAttr.Float, FromAttr.Positive)),
        minHeight = Value(FromAttr("min_height", FromAttr.Float, FromAttr.Positive)),
        numLevels = 0,
        roofShape = Value(Alternatives(
            FromAttr("roof:shape", FromAttr.String, RoofShapes),
            Constant("flat")
        )),
        roofHeight = Value(FromAttr("roof:height", FromAttr.Float, FromAttr.NonNegative)),
        claddingMaterial = PerBuilding(Value(Alternatives(
            FromAttr("building:material", FromAttr.String, CladdingMaterials),
            Constant("brick")
        ))),
        claddingColor = PerBuilding(Value(Alternatives(
            FromAttr("building:colour", FromAttr.Color),
            RandomWeighted((
                ((0.647, 0.165, 0.165, 1.), 1), # brown
                ((0.98, 0.502, 0.447, 1.), 1), # salmon
                ((0.502, 0., 0., 1.), 1) # maroon
            ))
        )))
    ),
    Facade(),
    Roof(
        roofCladdingMaterial = Value(Alternatives(
            FromAttr("roof:material", FromAttr.String, CladdingMaterials),
            Constant("metal")
        )),
        roofCladdingColor = Value(Alternatives(
            FromAttr("roof:colour", FromAttr.Color),
            RandomWeighted((
                ((0.686, 0.686, 0.686, 1.), 1),
                ((0.698, 0.698, 0.651, 1.), 1),
                ((0.784, 0.761, 0.714, 1.), 1)
            ))
        )),
        faces = Value(Conditional(
            lambda item: item.footprint.getStyleBlockAttr("roofShape") in ("dome", "half-dome", "onion"),
            Constant(smoothness.Smooth)
        ))
        #sharpEdges = smoothness.Side
    )
],
"residential": [
    Meta(
        buildingUse = "apartments",
        buildingLaf = "modern",
        height = "high rise"
    ),
    Footprint(
        height = Value(FromAttr("height", FromAttr.Float, FromAttr.Positive)),
        minHeight = Value(FromAttr("min_height", FromAttr.Float, FromAttr.Positive)),
        hasNumLevelsAttr = Value(FromAttr("building:levels", FromAttr.Integer, FromAttr.Positive)),
        numLevels = Value(Alternatives(
            FromAttr("building:levels", FromAttr.Integer, FromAttr.Positive),
            RandomWeighted(( (4, 10), (5, 40), (6, 10) ))
        )),
        minLevel = Value(Alternatives(
            FromAttr("building:min_level", FromAttr.Integer, FromAttr.NonNegative),
            Constant(0)
        )),
        #topHeight = Value( RandomNormal(1.) ),
        topHeight = 0.,
        #lastLevelHeight = PerBuilding( Value( RandomNormal(0.7*3.) ) ),
        levelHeight = Value( RandomNormal(3.) ),
        #groundLevelHeight = Value( RandomNormal(1.4*3) ),
        #bottomHeight = Value( RandomNormal(1.) ),
        roofShape = Value(Alternatives(
            FromAttr("roof:shape", FromAttr.String, RoofShapes),
            Constant("flat")
        )),
        roofHeight = Value(FromAttr("roof:height", FromAttr.Float, FromAttr.NonNegative)),
        claddingMaterial = PerBuilding(Value(Alternatives(
            FromAttr("building:material", FromAttr.String, CladdingMaterials),
            RandomWeighted(( ("brick", 1), ("plaster", 1) ))
        ))),
        claddingColor = PerBuilding(Value(Alternatives(
            FromAttr("building:colour", FromAttr.Color),
            Conditional(
                lambda footprint: footprint.getStyleBlockAttr("claddingMaterial") == "brick",
                RandomWeighted((
                    ((0.647, 0.165, 0.165, 1.), 1), # brown
                    ((0.98, 0.502, 0.447, 1.), 1), # salmon
                    ((0.502, 0., 0., 1.), 1) # maroon
                ))
            ),
            Conditional(
                lambda footprint: footprint.getStyleBlockAttr("claddingMaterial") == "plaster",
                RandomWeighted((
                    ((1., 0.627, 0.478, 1.), 1), # lightsalmon
                    ((0.565, 0.933, 0.565, 1.), 1), # lightgreen
                    ((1., 0.855, 0.725, 1.), 1) # peachpuff
                ))
            ),
            Conditional(
                lambda footprint: footprint.getStyleBlockAttr("claddingMaterial") == "glass",
                RandomWeighted((
                    ((0.306, 0.447, 0.573, 1.), 1),
                    ((0.169, 0.318, 0.361, 1.), 1),
                    ((0.094, 0.18, 0.271, 1.), 1)
                ))
            )
        )))
    ),
    Facade(
        label = "cladding only for structures without levels or too low structures or too narrow facades",
        condition = lambda facade: not facade.footprint.numLevels or\
            facade.footprint.height - facade.footprint.minHeight < minHeightForLevels or\
            facade.width < minWidthForOpenings
    ),
    Facade(
        markup = [
            Level(
                indices = (0, -1)#,
                #width = 1.
            )
        ]
    ),
    Roof(
        roofCladdingMaterial = Value(Alternatives(
            FromAttr("roof:material", FromAttr.String, CladdingMaterials),
            Conditional(
                lambda roof: roof.footprint.getStyleBlockAttr("roofShape") == "flat",
                Constant("concrete")
            ),
            # roofShape in ("pyramidal", "dome", "half-dome", "onion")
            Constant("metal")
        )),
        roofCladdingColor = Value(Alternatives(
            FromAttr("roof:colour", FromAttr.Color),
            Conditional(
                lambda roof: roof.getStyleBlockAttr("roofCladdingMaterial") == "concrete",
                RandomWeighted((
                    ((0.686, 0.686, 0.686, 1.), 1),
                    ((0.698, 0.698, 0.651, 1.), 1),
                    ((0.784, 0.761, 0.714, 1.), 1)
                ))
            ),
            # roofCladdingMaterial == "metal"
            RandomWeighted((
                ((0.686, 0.686, 0.686, 1.), 1),
                ((0.698, 0.698, 0.651, 1.), 1),
                ((0.784, 0.761, 0.714, 1.), 1)
            ))
        )),
        faces = Value(Conditional(
            lambda item: item.footprint.getStyleBlockAttr("roofShape") in ("dome", "half-dome", "onion"),
            Constant(smoothness.Smooth)
        ))
        #sharpEdges = smoothness.Side
    )
]
}