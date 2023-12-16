"""Approximations of graph properties and Heuristic methods for optimization.

The functions in this class are not imported into the top-level ``blosm_networkx``
namespace so the easiest way to use them is with::

    >>> from lib.blosm_networkx.algorithms import approximation

Another option is to import the specific function with
``from lib.blosm_networkx.algorithms.approximation import function_name``.

"""
from lib.blosm_networkx.algorithms.approximation.clustering_coefficient import *
from lib.blosm_networkx.algorithms.approximation.clique import *
from lib.blosm_networkx.algorithms.approximation.connectivity import *
from lib.blosm_networkx.algorithms.approximation.distance_measures import *
from lib.blosm_networkx.algorithms.approximation.kcomponents import *
from lib.blosm_networkx.algorithms.approximation.matching import *
from lib.blosm_networkx.algorithms.approximation.ramsey import *
from lib.blosm_networkx.algorithms.approximation.steinertree import *
from lib.blosm_networkx.algorithms.approximation.treewidth import *
from lib.blosm_networkx.algorithms.approximation.vertex_cover import *
from lib.blosm_networkx.algorithms.approximation.maxcut import *
