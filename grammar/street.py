from .value import Value

from . import Item


class StreetItem(Item):
    
    def __init__(self, defName, use, markup, condition, attrs):
        super().__init__(defName, use, markup, condition, attrs)
        self.init()
    
    def initAttrs(self):
        attrs = self.attrs
        # restructure <self.attrs>
        for attr in attrs:
            value = attrs[attr]
            isComplexValue = isinstance(value, Value)
            if isinstance(value, str):
                if value == "yes":
                    value = True
                elif value == "no":
                    value = False
                elif value == "none":
                    value = None
            
            attrs[attr] = (value.value if isComplexValue else value, isComplexValue)


class Street(StreetItem):
    
    def __init__(self, defName=None, use=None, markup=None, condition=None, **attrs):
        super().__init__(defName, use, markup, condition, attrs)


class Section(StreetItem):
    
    def __init__(self, defName=None, use=None, markup=None, condition=None, **attrs):
        super().__init__(defName, use, markup, condition, attrs)


class SideLane(StreetItem):
    
    def __init__(self, defName=None, use=None, markup=None, condition=None, **attrs):
        super().__init__(defName, use, markup, condition, attrs)


class Sidewalk(StreetItem):
    
    def __init__(self, defName=None, use=None, markup=None, condition=None, **attrs):
        super().__init__(defName, use, markup, condition, attrs)


class Vegetation(StreetItem):
    
    def __init__(self, defName=None, use=None, markup=None, condition=None, **attrs):
        super().__init__(defName, use, markup, condition, attrs)


class PtStop(StreetItem):
    
    def __init__(self, defName=None, use=None, markup=None, condition=None, **attrs):
        super().__init__(defName, use, markup, condition, attrs)


class Crosswalk(StreetItem):
    
    def __init__(self, defName=None, use=None, markup=None, condition=None, **attrs):
        super().__init__(defName, use, markup, condition, attrs)