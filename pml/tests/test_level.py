from . import makeTest


def test_nothing():
    makeTest(
"""
level {
    class: myclass;
}
""",
"""
Level(
    roofLevels = False,
    allLevels = False,
    cl = "myclass"
)
"""
    )


def test_indices_positive():
    makeTest(
"""
level[0:3] {
    class: myclass;
}
""",
"""
Level(
    indices = (0,3),
    roofLevels = False,
    allLevels = False,
    cl = "myclass"
)
"""
    )


def test_indices_negative():
    makeTest(
"""
level[-2:-1] {
    class: myclass;
}
""",
"""
Level(
    indices = (-2,-1),
    roofLevels = False,
    allLevels = False,
    cl = "myclass"
)
"""
    )


def test_indices_positive_negative():
    makeTest(
"""
level[2:-1] {
    class: myclass;
}
""",
"""
Level(
    indices = (2,-1),
    roofLevels = False,
    allLevels = False,
    cl = "myclass"
)
"""
    )


def test_single_index_positive():
    makeTest(
"""
level[2] {
    class: myclass;
}
""",
"""
Level(
    indices = (2,2),
    roofLevels = False,
    allLevels = False,
    cl = "myclass"
)
"""
    )


def test_single_index_negative():
    makeTest(
"""
level[-2] {
    class: myclass;
}
""",
"""
Level(
    indices = (-2,-2),
    roofLevels = False,
    allLevels = False,
    cl = "myclass"
)
"""
    )


def test_condition():
    makeTest(
"""
level(item.front) {
    class: myclass;
}
""",
"""
Level(
    roofLevels = False,
    allLevels = False,
    condition = lambda item: item.front,
    cl = "myclass"
)
"""
    )


def test_condition():
    makeTest(
"""
level(item.front) {
    class: myclass;
}
""",
"""
Level(
    roofLevels = False,
    allLevels = False,
    condition = lambda item: item.front,
    cl = "myclass"
)
"""
    )


def test_indices_condition():
    makeTest(
"""
level[0:3](item.front) {
    class: myclass;
}
""",
"""
Level(
    indices = (0,3),
    roofLevels = False,
    allLevels = False,
    condition = lambda item: item.front,
    cl = "myclass"
)
"""
    )


def test_single_index_condition():
    makeTest(
"""
level[2](item.front) {
    class: myclass;
}
""",
"""
Level(
    indices = (2,2),
    roofLevels = False,
    allLevels = False,
    condition = lambda item: item.front,
    cl = "myclass"
)
"""
    )


def test_roof():
    makeTest(
"""
level[@roof] {
    class: myclass;
}
""",
"""
Level(
    roofLevels = True,
    allLevels = False,
    cl = "myclass"
)
"""
    )


def test_all():
    makeTest(
"""
level[@all] {
    class: myclass;
}
""",
"""
Level(
    roofLevels = False,
    allLevels = True,
    cl = "myclass"
)
"""
    )


def test_roof_condition():
    makeTest(
"""
level[@roof](item.front) {
    class: myclass;
}
""",
"""
Level(
    roofLevels = True,
    allLevels = False,
    condition = lambda item: item.front,
    cl = "myclass"
)
"""
    )


def test_all_condition():
    makeTest(
"""
level[@all](item.front) {
    class: myclass;
}
""",
"""
Level(
    roofLevels = False,
    allLevels = True,
    condition = lambda item: item.front,
    cl = "myclass"
)
"""
    )


def test_roof_indices():
    makeTest(
"""
level[@roof][0:1] {
    class: myclass;
}
""",
"""
Level(
    indices = (0,1),
    roofLevels = True,
    allLevels = False,
    cl = "myclass"
)
"""
    )


def test_all_indices():
    makeTest(
"""
level[@all][0:1] {
    class: myclass;
}
""",
"""
Level(
    indices = (0,1),
    roofLevels = False,
    allLevels = True,
    cl = "myclass"
)
"""
    )


def test_roof_single_index():
    makeTest(
"""
level[@roof][0] {
    class: myclass;
}
""",
"""
Level(
    indices = (0,0),
    roofLevels = True,
    allLevels = False,
    cl = "myclass"
)
"""
    )


def test_all_single_index():
    makeTest(
"""
level[@all][0] {
    class: myclass;
}
""",
"""
Level(
    indices = (0,0),
    roofLevels = False,
    allLevels = True,
    cl = "myclass"
)
"""
    )


def test_roof_indices_condition():
    makeTest(
"""
level[@roof][0:1](item.front) {
    class: myclass;
}
""",
"""
Level(
    indices = (0,1),
    roofLevels = True,
    allLevels = False,
    condition = lambda item: item.front,
    cl = "myclass"
)
"""
    )


def test_all_indices_condition():
    makeTest(
"""
level[@all][0:1](item.front) {
    class: myclass;
}
""",
"""
Level(
    indices = (0,1),
    roofLevels = False,
    allLevels = True,
    condition = lambda item: item.front,
    cl = "myclass"
)
"""
    )


def test_roof_single_index_condition():
    makeTest(
"""
level[@roof][1](item.front) {
    class: myclass;
}
""",
"""
Level(
    indices = (1,1),
    roofLevels = True,
    allLevels = False,
    condition = lambda item: item.front,
    cl = "myclass"
)
"""
    )


def test_all_single_index_condition():
    makeTest(
"""
level[@all][1](item.front) {
    class: myclass;
}
""",
"""
Level(
    indices = (1,1),
    roofLevels = False,
    allLevels = True,
    condition = lambda item: item.front,
    cl = "myclass"
)
"""
    )