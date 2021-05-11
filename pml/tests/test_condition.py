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