from . import makeTest


def test_one():
    makeTest(
"""
facade(item.gable and 2.5 < item.width < 12.) {
    claddingColor: if (item.front) green;
}
""",
"""
Facade(
    condition = lambda item : item.gable and 2.5 < item.width < 12.,
    claddingColor = Value(Conditional(
        lambda item: item.front,
        Constant((0.0, 0.502, 0.0, 1.0))
    ))
)
"""
    )


def test_two():
    makeTest(
"""
facade {
    claddingColor: if (item.front) green | if (item.back) red;
}
""",
"""
Facade(
    claddingColor =  Value(Alternatives(
        Conditional(
            lambda item: item.front,
            Constant((0.0, 0.502, 0.0, 1.0))
        ),
        Conditional(
            lambda item: item.back,
            Constant((1.0, 0.0, 0.0, 1.0))
        )
    ))
)
"""
    )


def test_one_and_fixed():
    makeTest(
"""
facade (item.gable and 2.5 < item.width < 12.){
    claddingColor: if (item.front) green | red;
}
""",
"""
Facade(
    condition = lambda item : item.gable and 2.5 < item.width < 12.,
    claddingColor =  Value(Alternatives(
        Conditional(
            lambda item: item.front,
            Constant((0.0, 0.502, 0.0, 1.0))
        ),
        Constant((1.0, 0.0, 0.0, 1.0))
    ))
)
"""
    )


def test_fixed_and_one():
    makeTest(
"""
facade (item.gable and 2.5 < item.width < 12.){
    claddingColor: red | if (item.front) green;
}
""",
"""
Facade(
    condition = lambda item : item.gable and 2.5 < item.width < 12.,
    claddingColor =  Value(Alternatives(
        Constant((1.0, 0.0, 0.0, 1.0)),
        Conditional(
            lambda item: item.front,
            Constant((0.0, 0.502, 0.0, 1.0))
        )
    ))
)
"""
    )