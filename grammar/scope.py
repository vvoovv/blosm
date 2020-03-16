

# Constants for the scope.
perBuilding = 1
perFootprint = 2


class Scope:
    """
    The base class for the wrapper classes below.
    
    An attribute in the style block can be defined in the form <ScopeClass(value)>,
    where <ScopeClass> is one of the classes defined below and,
    <value> is an instance of <grammar.value.Value>
    """
    
    def __init__(self, value):
        self.value = value
    

class PerBuilding(Scope):
    """
    A wrapper class for the scope.
    <value> is evaluated once per building
    """
    scope = perBuilding


class PerFootprint(Scope):
    """
    A wrapper class for the scope.
    <value> is evaluated for every footprint in the building
    """
    scope = perFootprint