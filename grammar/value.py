import math
from util.osm import parseNumber


_values = ( (1,), (1,), (1,) )


class Value:
    
    def __init__(self, value):
        self.value = value
        self.fromItem = getattr(value, "fromItem", False)
        self.prev = None


class Alternatives:
    """
    A value out of two or more alternatives
    """
    def __init__(self, *values):
        self.values = values
    
    @property
    def value(self):
        for value in self.values:
            if hasattr(value, "value"):
                value = value.value
            if not value is None:
                break
        return value


class FromAttr:
    
    Integer = _values[0]
    Float = _values[1]
    String = _values[2]
    
    Positive = _values[0]
    NonNegative = _values[1]
    
    def __init__(self, attr, valueType, valueCondition):
        self.attr = attr
        self.valueType = valueType
        self.valueCondition = valueCondition
        self.item = None
        self.fromItem = True
        self.prev = None
    
    @property
    def value(self):
        value = self.item.attr(self.attr)
        if self.valueType is FromAttr.Integer:
            if not value is None:
                value = parseNumber(value)
                if not value is None:
                    value = math.ceil(value)
                    if self.valueCondition is FromAttr.Positive and value < 1.:
                        value = None
                    elif self.valueCondition is FromAttr.NonNegative and value < 0.:
                        value = None
        return value