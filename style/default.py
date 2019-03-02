from grammar import *
from util.random import RandomWeighted, RandomNormal

levelHeight = 3.

styles = {
"mid rise residential zaandam": [
    Footprint(
        levels = RandomWeighted(( (4, 10), (5, 40), (6, 10) ))
    ),
    Footprint(
        levels = lambda i: i.data["building:levels"],
        minLevel = lambda i: i.data["building:min_level"],
        levelHeight = RandomNormal(levelHeight),
        shape = Roof.gabled,
        height = 5.
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
                    ),
                    Basement(
                        markup = [
                            Window(rows=1, colums=1)
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
        condition = lambda side: side.index == 0,
        # width of the section
        width = useFrom("main_section"),
        markup = [
            Dormer(), Dormer()
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
        condition = lambda side: side.index == 1,
        # width of the section
        width = useFrom("main_section"),
        markup = [
            # operable skylight or roof window
            Window(), Window()
        ]
    ),
    Ridge(
        width = useFrom("main_section"),
        markup = [
            Chimney()    
        ]    
    )
]
}