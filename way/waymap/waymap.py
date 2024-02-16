from mathutils import Vector
from itertools import tee

import lib.blosm_networkx as nx
from way.item.dummy_node import DummyNode

class WayMap(nx.MultiDiGraph):
    ID = 0
    def __init__(self):
        super(WayMap,self).__init__()
        self.id = WayMap.ID
        WayMap.ID += 1

    # Store a street node object (like Intersection, SymLane, SideLane,
    # Crosswalk, PtStop, ...) into map.
    def addStreetNode(self, streetNode):
        location = streetNode.location.freeze()
        self.add_node(location, object = streetNode)

    # Get a street node at a given location
    def getStreeNode(self,location):
        location.freeze()
        return self.nodes[location]

    # Replace an existing street node by a new one at the same location,
    # keeping all edges from and to this object.
    # If the location does not yet exist in the map, a new node is created.
    def replaceStreetNodeBy(self, newStreetNode):
        location = newStreetNode.location.freeze()
        self.add_node(location, object = newStreetNode)

    # Remove a street node and all adjacent edges. Attempting to 
    # remove a nonexistent node will raise an exception.
    def removeStreetNode(self, location):
        location.freeze()
        self.remove_node(location)

    # Iterate over street nodes, optionaly of a given type
    def iterNodes(self,nodeType=None):
        if nodeType:
            for location, data in self.nodes(data=True):
                if data:
                    streetNode = data['object']
                    if isinstance(streetNode,nodeType):
                        yield location, streetNode
        else:
            for location, data in self.nodes(data=True):
                streetNode = data['object'] if data else DummyNode()
                yield location, streetNode

    # Add a line section object (like Section, Street, Bundle) into map
    def addSection(self, section):
        src = section.src.freeze()
        dst = section.dst.freeze()
        self.add_edge(src,dst,None,object=section)

    # Get the line section objects between two given nodes
    def getSections(self,src, dst):
        src.freeze()
        dst.freeze()
        for sec in self.get_edge_data(src,dst).values():
            yield sec['object']

    def getSectionObject(self,src,dst,key):
        src.freeze()
        dst.freeze()
        return self[src][dst][key]['object']
    
    # Iterate over line section object, optionally of a given type
    def iterSections(self,sectionType=None):
        if sectionType:
            for src, dst, key, section in self.edges(data='object',keys=True):
                if isinstance(section,sectionType):
                    yield src, dst, key, section
        else:
            for src, dst, key, section in self.edges(data='object',keys=True):
                yield src, dst, key, section

    # Remove an existing section object.
    def removeSection(self, section):
        src = section.src.freeze()
        dst = section.dst.freeze()
        self.remove_edge(src,dst)

    # Replace an existing section object by a new one at the same positions.
    # If the positions do not yet exist in the map, a new section is created.
    def replaceSectionBy(self, newSec):
        src = newSec.src.freeze()
        dst = newSec.dst.freeze()
        if self.has_edge(src,dst):
            self.remove_edge(src,dst)
        self.addSection(newSec)

    def getInOutSections(self, node):
        location = node if isinstance(node,Vector) else node.location
        inSections = [section for _, _, _, section in self.in_edges(location,data='object',keys=True)]
        outSections = [section for _, _, _, section in self.out_edges(location,data='object',keys=True)]
        return inSections, outSections


    # def splitSectionBy(self, src, dst, splitPos, node):
    #     if self.has_edge(src,dst):
    #         section = [sec for sec in self.getSections(src,dst)][0] # multi-section?
    #         # replace by two section parts
    #         part1 = 
    #   self.add_path([src,spliPos,dst]) ???


            
    # Create an iterator for all paths between nodes of a given class (e.g. Intersection).
    # The first node (of class nodeClass) is not included.
    # Parameters: 
    #   nodeClasses: class or tuple of classes of nodes at start and end of  a path.
    # Returns:
    #   src: Location of the first node.
    #   sections: List of instances of the line section classes along path (path to last nodeClass included).
    #   nodes: List of instances of the lstreet node classes along path (last nodeClass included).
    def pathBetweenNodesOfType(self,nodeClasses):
        srcIter, dstIter = tee(node for node, data in self.nodes(data=True) if isinstance(data.get('object',None),nodeClasses) )
        for src in srcIter:
            for dst in dstIter:
                if src != dst:
                    pathsBetween = list(nx.all_simple_edge_paths(self, source=src, target=dst))
                    for path in pathsBetween:
                        sections = [ self.edges[edge]['object'] for edge in path ]
                        nodes = [self.nodes[dst]['object'] for _,dst,_ in path]
                        yield src, sections, nodes


 