
_values = (
    (1,),
    (1,),
    (1,)
)

# Constants for the scope.
perBuilding = _values[0]
perFootprint = _values[1]


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