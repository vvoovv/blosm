

class Item:
    
    def __init__(self, markup, condition, defs, attrs):
        self.markup = markup
        self.condition = condition
        self.defs = defs
        self.attrs = attrs


class Footprint(Item):
    
    def __init__(self, markup=None, condition=None, defs=None, **attrs):
        super().__init__(markup, condition, defs, attrs)


class Facade(Item):
    
    def __init__(self, markup=None, condition=None, defs=None, **attrs):
        super().__init__(markup, condition, defs, attrs)


class Roof(Item):
    
    def __init__(self, markup=None, condition=None, defs=None, **attrs):
        super().__init__(markup, condition, defs, attrs)


class Section(Item):
    
    def __init__(self, markup=None, condition=None, defs=None, **attrs):
        super().__init__(markup, condition, defs, attrs)


class Level(Item):
    
    def __init__(self, markup=None, condition=None, defs=None, **attrs):
        super().__init__(markup, condition, defs, attrs)


class Window(Item):
    
    def __init__(self, markup=None, condition=None, defs=None, **attrs):
        super().__init__(markup, condition, defs, attrs)


class Balcony(Item):
    
    def __init__(self, markup=None, condition=None, defs=None, **attrs):
        super().__init__(markup, condition, defs, attrs)


class Door(Item):
    
    def __init__(self, markup=None, condition=None, defs=None, **attrs):
        super().__init__(markup, condition, defs, attrs)
        

class Chimney(Item):
    
    def __init__(self, markup=None, condition=None, defs=None, **attrs):
        super().__init__(markup, condition, defs, attrs)