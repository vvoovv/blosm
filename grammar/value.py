import math
from util.osm import parseNumber


_values = ( (1,), (1,), (1,) )


class Value:
    
    def __init__(self, value):
        self.value = value


class Alternatives:
    """
    A value out of two or more alternatives
    """
    def __init__(self, *values):
        self.values = values
        self.item = None
    
    def setData(self, item):
        self.item = item
    
    @property
    def value(self):
        for value in self.values:
            value.setData(self.item)
            value = value.value
            if not value is None:
                break
        return value


class Constant:
    
    def __init__(self, value):
        self._value = value
    
    @property
    def value(self):
        return self._value

    def setData(self, data):
        # the method is needed for consistency with the other classes providing functionality of "value"
        return


class FromAttr:
    
    Integer = _values[0]
    Float = _values[1]
    String = _values[2]
    
    Positive = _values[0]
    NonNegative = _values[1]
    
    def __init__(self, attr, valueType, valueCondition=None):
        self.attr = attr
        self.valueType = valueType
        self.valueCondition = valueCondition
        # if the attribute <self.item> is set, it means that the value is data driven
        self.item = None
    
    @property
    def value(self):
        value = self.item.attr(self.attr)
        valueCondition = self.valueCondition
        if self.valueType is FromAttr.Integer:
            if not value is None:
                value = parseNumber(value)
                if not value is None:
                    value = math.ceil(value)
                    if valueCondition is FromAttr.Positive and value < 1.:
                        value = None
                    elif valueCondition is FromAttr.NonNegative and value < 0.:
                        value = None
        elif self.valueType is FromAttr.Float:
            if not value is None:
                value = parseNumber(value)
                if not value is None:
                    if valueCondition is FromAttr.Positive and value <= 0.:
                        value = None
                    elif valueCondition is FromAttr.NonNegative and value < 0.:
                        value = None
        elif valueCondition:
            # We have a string and <valueCondition> is dictionary
            # where the keys are the possible values for the string
            if not value in valueCondition:
                value = None
        return value
    
    def setData(self, item):
        self.item = item