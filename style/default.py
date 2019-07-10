from grammar import *
from grammar.util import perBuilding, perFootprint
from grammar import units, symmetry
from grammar.value import Value, FromAttr, Alternatives, Constant
from util.random import RandomWeighted, RandomNormal
from action.volume.roof import Roof


styles = {
"mid rise residential zaandam": [
    Footprint(
        levels = Value(Alternatives(
            FromAttr("building:levels", FromAttr.Integer, FromAttr.Positive),
            RandomWeighted(( (4, 10), (5, 40), (6, 10) ))
        )),
        roofLevels = 1,
        minLevel = Value(Alternatives(
            FromAttr("building:min_level", FromAttr.Integer, FromAttr.NonNegative),
            Constant(0)
        )),
        lastLevelHeight = perBuilding( Value( RandomNormal(0.7*3.) ) ),
        levelHeight = Value( RandomNormal(3.) ),
        groundLevelHeight = Value( RandomNormal(1.4*3) ),
        basementHeight = Value( RandomNormal(1.) ),
        roofShape = Value(Alternatives(
            FromAttr("roof:shape", FromAttr.String, Roof.shapes),
            Constant("gabled")
            #RandomWeighted(( ("gabled", 10), ("flat", 40) ))
        )),
        roofHeight = Value(Alternatives(
            FromAttr("roof:height", FromAttr.Float, FromAttr.NonNegative),
            RandomNormal(3.7)
        )),
        roofAngle = Value(FromAttr("roof:angle", FromAttr.Float)),
        roofDirection = Value(Alternatives(
            FromAttr("roof:direction", FromAttr.String, Roof.directions),
            FromAttr("roof:direction", FromAttr.Float),
            FromAttr("roof:slope:direction", FromAttr.String, Roof.directions),
            FromAttr("roof:slope:direction", FromAttr.Float)
        )),
        roofOrientation = Value( FromAttr("roof:orientation", FromAttr.String) ),
        lastLevelOffsetFactor = Value(RandomWeighted((
            (0., 50), (0.05, 3), (0.1, 5), (0.15, 5), (0.2, 5), (0.25, 5), (0.3, 5),
            (0.35, 5), (0.4, 5), (0.45, 5), (0.5, 3), (0.55, 2), (0.6, 2)
        )))
    ),
    Facade(
        defName = "brown_brick",
        claddingMaterial = "brick",
        claddingColor = "brown",
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
            Basement(
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
        markup = [
            Level(
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
                index = 0,
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
        use = ["brown_brick"],
        label = "front facade",
        condition = lambda facade: facade.front,
        # None or symmetry.MiddleOfLast or symmetry.RightmostOfLast
        symmetry = symmetry.MiddleOfLast,
        # flip items for the total symmetry or leave them intact
        symmetryFlip = True,
        markup = [
            Div(
                use = ["window_and_balcony"],
                id = "main_section",
                label = "Window and Balcony"
            ),
            Div(
                use = ["staircase"],
                label = "Staircase"
            )
        ] 
    ),
    Facade(
        use = ["brown_brick"],
        label = "back facade",
        condition = lambda facade: facade.back,
        markup = [
            Level(
                markup = [
                    Balcony(),
                    Window(use = "back_facade_window"),
                    Window(use = "back_facade_window")
                ]
            )
        ]
    ),
    RoofSide(
        condition = lambda side: side.front,
        markup = [
            Div(
                use = ["roof_side"],
                markup = [
                    # openable skylight or roof window
                    Window(use = "roof_window"),
                    Window(use = "roof_window")
                ]
            )
        ]
    ),
    RoofSide(
        condition = lambda side: side.back,
        markup = [
            Div(
                use = ["roof_side"],
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
]
}