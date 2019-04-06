from grammar import *
from grammar.value import Value, FromAttr
from util.random import RandomWeighted, RandomNormal

levelHeight = 3.

styles = {
"mid rise residential zaandam": [
    Footprint(
        levels = Value( RandomWeighted(( (4, 10), (5, 40), (6, 10) )) )
    ),
    Footprint(
        levels = Value( FromAttr("building:levels", FromAttr.Integer, FromAttr.Positive) ),
        minLevel = Value( FromAttr("building:min_level", FromAttr.Integer, FromAttr.NonNegative) ),
        levelHeight = Value( RandomNormal(levelHeight) ),
        roofShape = Roof.gabled
    ),
    Facade(
        name = "front facade",
        condition = lambda facade: facade.front,
        # None or Grammar.MiddleOfLast or Grammar.RightmostOfLast
        symmetry = Grammar.MiddleOfLast,
        # flip items for the total symmetry or leave them intact
        symmetryFlip = True,
        id = "main_section",
        markup = [
            Div(
                name = "Window and Balcony",
                markup = [
                    Level(
                        markup = [
                            Window(rows=1, colums=2),
                            Balcony()
                        ]
                    ),
                    Basement(
                        markup = [
                            Window(rows=1, colums=1)
                        ]
                    )
                ]
            ),
            Div(
                name = "Staircase",
                markup = [
                    Level(
                        markup = [
                            Window(panels=(60, WindowPanel(40, True)), width=3., height=1.6)
                        ]
                    ),
                    Level(
                        index = 0,
                        condition = lambda level: level.index == 0,
                        markup = [
                            Door()
                        ]
                    )
                ]
            )
        ] 
    ),
    Facade(
        name = "back facade",
        cl = "back",
        condition = lambda facade: facade.back,
        markup = [
            Level(
                markup = [
                    Window(), Balcony(), Balcony(), Window()
                ]
            )
        ],
        defs = [
            Window(
                panels=(50, WindowPanel(50, True)), width=1.4, height=1.6
            ),
            Balcony(
                fencing = "bars"
            )
        ]
    ),
    RoofSide(
        condition = lambda side: side.back,
        markup = [
            Div(
                width = useFrom("main_section"),
                markup = [
                    Dormer(), Dormer()
                ]
            )
        ],
        defs = [
            Dormer(
                shape = "rectangular",
                markup = [
                    Window(), Window()
                ],
                defs = [
                    Window(
                        panels=(50, WindowPanel(50, True)),
                        width=1.,
                        height=1.6
                    )
                ]
            )
        ]
    ),
    RoofSide(
        condition = lambda side: side.front,
        markup = [
            Div(
                width = useFrom("main_section"),
                markup = [
                    # openable skylight or roof window
                    Window(), Window()
                ]
            )
        ]
    ),
    Ridge(
        markup = [
            Div(
                width = useFrom("main_section"),
                markup = [
                    Chimney()
                ]
            )  
        ]    
    )
]
}