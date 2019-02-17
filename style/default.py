from grammar import *


style = []


style.add("mid rise residential zaandam", [
    Footprint(
        levels = RandomWeighted(( (4, 10), (5, 40), (6, 10) ))
    ),
    Footprint(
        levels = lambda i: i.data["building:levels"],
        minLevel = lambda i: i.data["building:min_level"],
        levelHeight = RandomNormal(levelHeight)
    ),
    Facade(
        name = "front facade",
        condition = lambda facade: facade.front,
        symmetry = None or last_middle or last_right,
        id = "main_section",
        [
            Div(
                name = "Window and Balcony",
                [
                    Level(
                        [
                            Window(rows=1, colums=2),
                            Balcony()
                        ]
                    )
                ]
            ),
            Div(
                name = "Staircase",
                [
                    Level(
                        [
                            Window(panels=(60, Panel(40, true)), width=3., height=1.6)
                        ]
                    ),
                    Level(
                        index = 0,
                        condition = lambda level: level.index == 0,
                        [
                            Door()
                        ]
                    ),
                    Basement(
                        [
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
        [
            Level(
                [
                    Window(), Balcony(), Balcony(), Window()
                ]
            )
        ],
        defs = [
            Window(
                panels=(50, Panel(50, true)), width=1.4, height=1.6
            ),
            Balcony(
                fencing = "bars"
            )
        ]
    ),
    Roof(
        shape = Roof.gabled,
        height = 5.,
        # pitch = 30,
        # orientation = "across" or "along"
        [
            RoofSide(
                condition = lambda side: side.index == 0,
                # width of the section
                width = _from("main_section"),
                [
                    Dormer(), Dormer()
                ]
            ),
            RoofSide(
                condition = lambda side: side.index == 1,
                # width of the section
                width = _from("main_section"),
                [
                    # operable skylight or roof window
                    Window(), Window()
                ]
            ),
            Ridge(
                width = _from("main_section"),
                [
                    Chimney()    
                ]    
            )
        ],
        defs = [
            Dormer(
                shape = "rectangular"
                [
                    Window(), Window()
                ],
                defs = [
                    Window(
                        panels=(50, Panel(50, true)), width=1., height=1.6
                    )
                ]
            )
        ]
    )
])