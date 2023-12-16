"""Functions for computing and measuring community structure.

The ``community`` subpackage can be accessed by using :mod:`blosm_networkx.community`, then accessing the
functions as attributes of ``community``. For example::

    >>> import lib.blosm_networkx as nx
    >>> G = nx.barbell_graph(5, 1)
    >>> communities_generator = nx.community.girvan_newman(G)
    >>> top_level_communities = next(communities_generator)
    >>> next_level_communities = next(communities_generator)
    >>> sorted(map(sorted, next_level_communities))
    [[0, 1, 2, 3, 4], [5], [6, 7, 8, 9, 10]]

"""
from lib.blosm_networkx.algorithms.community.asyn_fluid import *
from lib.blosm_networkx.algorithms.community.centrality import *
from lib.blosm_networkx.algorithms.community.kclique import *
from lib.blosm_networkx.algorithms.community.kernighan_lin import *
from lib.blosm_networkx.algorithms.community.label_propagation import *
from lib.blosm_networkx.algorithms.community.lukes import *
from lib.blosm_networkx.algorithms.community.modularity_max import *
from lib.blosm_networkx.algorithms.community.quality import *
from lib.blosm_networkx.algorithms.community.community_utils import *
from lib.blosm_networkx.algorithms.community.louvain import *
