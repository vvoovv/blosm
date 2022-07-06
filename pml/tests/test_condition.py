from . import makeTest


def test_style_variable():
    makeTest(
"""
facade(item.front and style.count<=1) {
    class: facade_with_door;
}
""",
"""
Facade(
    condition = lambda item : item.front and self.count <= 1,
    cl = "facade_with_door"
)
"""
    )


def test_nested_attributes():
    makeTest(
"""
facade(item.aa or item.aa.bb or item.aa.bb.cc or item.aa.bb.cc.dd or item.aa.bb.cc.dd.ee) {
    class: my_class;
}
""",
"""
Facade(
    condition = lambda item : item.aa or item.aa.bb or item.aa.bb.cc or item.aa.bb.cc.dd or item.aa.bb.cc.dd.ee,
    cl = "my_class"
)
"""
    )