"""
blosm_networkx
========

blosm_networkx is a Python package for the creation, manipulation, and study of the
structure, dynamics, and functions of complex networks.

See https://blosm_networkx.org for complete documentation.
"""

__version__ = "3.3rc0.dev0"


# These are imported in order as listed
from lib.blosm_networkx.exception import *

from lib.blosm_networkx import utils
from lib.blosm_networkx.utils.backends import _dispatch

from lib.blosm_networkx import classes
from lib.blosm_networkx.classes import filters
from lib.blosm_networkx.classes import *

from lib.blosm_networkx import convert
from lib.blosm_networkx.convert import *

from lib.blosm_networkx import relabel
from lib.blosm_networkx.relabel import *

from lib.blosm_networkx import generators
from lib.blosm_networkx.generators import *

from lib.blosm_networkx import readwrite
from lib.blosm_networkx.readwrite import *

# Need to test with SciPy, when available
from lib.blosm_networkx import algorithms
from lib.blosm_networkx.algorithms import *
