# -*- coding:utf-8 -*-

# ##### BEGIN LGPL LICENSE BLOCK #####
# GEOS - Geometry Engine Open Source
# http://geos.osgeo.org
#
# Copyright (C) 2011 Sandro Santilli <strk@kbt.io>
# Copyright (C) 2005 2006 Refractions Research Inc.
# Copyright (C) 2001-2002 Vivid Solutions Inc.
# Copyright (C) 1995 Olivier Devillers <Olivier.Devillers@sophia.inria.fr>
#
# This is free software you can redistribute and/or modify it under
# the terms of the GNU Lesser General Public Licence as published
# by the Free Software Foundation.
# See the COPYING file for more information.
#
# ##### END LGPL LICENSE BLOCK #####

# <pep8 compliant>

# ----------------------------------------------------------
# Partial port (version 3.7.0) by: Stephen Leger (s-leger)
#
# ----------------------------------------------------------


"""
@TODO:
implement nearestneighboor and BoundablePair
"""


from .shared import Envelope
from math import ceil, sqrt


class Interval():
    """
     * Represents an (1-dimensional) closed interval on the Real number line.
    """
    def __init__(self, nmin, nmax):
        self._mini = nmin
        self._maxi = nmax

    def getCentre(self):
        return (self._maxi + self._mini) / 2.0

    def expandToInclude(self, other):
        if other._maxi > self._maxi:
            self._maxi = other._maxi
        if other._mini < self._mini:
            self._mini = other._mini
        return self

    def intersects(self, other):
        return not (self._mini > other._maxi or self._maxi < other._mini)

    def equals(self, other):
        return (self._mini == other._mini and self._maxi == other._maxi)


class SpatialIndex():
    """
     * Abstract class defines basic insertion and query operations supported by
     * classes implementing spatial index algorithms.
     *
     * A spatial index typically provides a primary filter for range rectangle queries. A
     * secondary filter is required to test for exact intersection. Of course, this
     * secondary filter may consist of other tests besides intersection, such as
     * testing other kinds of spatial relationships.
     *
     * Last port: index/SpatialIndex.java rev. 1.11 (JTS-1.7)
     *
    """
    def insert(self, itemEnv, item):
        """
         * Adds a spatial item with an extent specified by the given Envelope
         * to the index
         *
         * @param itemEnv
         *    Envelope of the item, ownership left to caller.
         *    TODO: Reference hold by this class ?
         *
         * @param item
         *    Opaque item, ownership left to caller.
         *    Reference hold by this class.
        """
        raise NotImplementedError()

    def query(self, searchEnv, visitor=None):
        """
         * Queries the index for all items whose extents intersect the given search Envelope
         *
         * Note that some kinds of indexes may also return objects which do not in fact
         * intersect the query envelope.
         *
         * @param searchEnv the envelope to query for
         * @param visitor a visitor object to apply to the items found
         * @return a list of the items found by the query in a newly allocated vector
        """
        raise NotImplementedError()

    def remove(self, itemEnv, item):
        """
         * Removes a single item from the tree.
         *
         * @param itemEnv the Envelope of the item to remove
         * @param item the item to remove
         * @return true if the item was found
        """
        raise NotImplementedError()


class ItemsListItem():

    item_is_geometry = 0
    item_is_list = 1

    def __init__(self, item):
        self.g = None
        self.l = None
        if type(item).__name__ == 'ItemsList':
            self.t = ItemsListItem.item_is_list
            self.l = item
        else:
            self.t = ItemsListItem.item_is_geometry
            self.g = item

    def get_itemlist(self):
        assert(self.t == ItemsListItem.item_is_list), "wrong type"
        return self.l

    def get_geometry(self):
        assert(self.t == ItemsListItem.item_is_geometry), "wrong type"
        return self.g

    def __str__(self):
        if self.t == ItemsListItem.item_is_list:
            return "ItemsListItem t:{}\n{}".format(self.t, str(self.t))
        else:
            return "ItemsListItem t:{} {}".format(self.t, type(self.g).__name__)


class ItemsList(list):
    # list contains boundables or lists of boundables. The lists are owned by
    # this class, the plain boundables are held by reference only.
    def __init__(self):
        list.__init__(self)

    def delete_item(self, item):
        if ItemsListItem.item_is_list == item.t:
            del item.l

    def add(self, item):
        self.append(ItemsListItem(item))

    def __str__(self):
        return "ItemsList {}".format("\n".join([str(item) for item in self]))


class Boundable():
    """
     * Returns a representation of space that encloses this Boundable,
     * preferably not much bigger than this Boundable's boundary yet
     * fast to test for intersection with the bounds of other Boundables.
     *
     * The class of object returned depends
     * on the subclass of AbstractSTRtree.
     *
     * @return an Envelope (for STRtrees), an Interval (for SIRtrees),
     * or other object (for other subclasses of AbstractSTRtree)
     *
     * @see AbstractSTRtree.IntersectsOp
    """
    @property
    def bounds(self):
        return self._bounds


class ItemBoundable(Boundable):
    """
     * Boundable wrapper for a non-Boundable spatial object.
     * Used internally by AbstractSTRtree.
    """
    def __init__(self, newBounds, newItem):
        Boundable.__init__(self)
        self.item = newItem
        self._bounds = newBounds


class BoundablePair():
    """
      * A pair of {Boundable}s, whose leaf items
     * support a distance metric between them.
     * Used to compute the distance between the members,
     * and to expand a member relative to the other
     * in order to produce new branches of the
     * Branch-and-Bound evaluation tree.
     * Provides an ordering based on the distance between the members,
     * which allows building a priority queue by minimum distance.
     *
     * @author Martin Davis
    """


class AbstractNode(Boundable):
    """
     * A node of the STR tree.
     *
     * The children of this node are either more nodes
     * (AbstractNodes) or real data (ItemBoundables).
     *
     * If this node contains real data (rather than nodes),
     * then we say that this node is a "leaf node".
    """
    def __init__(self, newLevel, capacity=10):
        # Boundable
        self.childs = []
        self.level = newLevel
        self._bounds = None

    def _computeBounds(self):
        raise NotImplementedError()

    @property
    def bounds(self):
        """
         * Returns a representation of space that encloses this Boundable,
         * preferably not much bigger than this Boundable's boundary yet fast to
         * test for intersection with the bounds of other Boundables.
         * The class of object returned depends on the subclass of
         * AbstractSTRtree.
         *
         * @return an Envelope (for STRtrees), an Interval (for SIRtrees),
         *  or other object (for other subclasses of AbstractSTRtree)
         *
         * @see AbstractSTRtree.IntersectsOp
        """
        if self._bounds is None:
            self._bounds = self._computeBounds()

        return self._bounds

    def addChild(self, child):
        """
         * Adds either an AbstractNode, or if this is a leaf node, a data object
         * (wrapped in an ItemBoundable)
        """
        self.childs.append(child)


class AbstractSTRtree():
    """
     * Base class for STRtree and SIRtree.
     *
     * STR-packed R-trees are described in:
     * P. Rigaux, Michel Scholl and Agnes Voisard. Spatial Databases With
     * Application To GIS. Morgan Kaufmann, San Francisco, 2002.
     *
     * This implementation is based on Boundables rather than just AbstractNodes,
     * because the STR algorithm operates on both nodes and
     * data, both of which are treated here as Boundables.
    """
    def __init__(self, nodeCapacity: int):

        self._built = False

        # BoundableList
        self.itemBoundables = []

        self.nodeCapacity = nodeCapacity

        # AbstractNode
        self.nodes = []
        self.root = None
        """
         * A test for intersection between two bounds, necessary because
         * subclasses of AbstractSTRtree have different implementations of
         * bounds.
        """
        self._intersectOp = None

    def _createHigherLevels(self, childsOfALevel, level: int):
        """
         * Creates the levels higher than the given level
         *
         * @param childsOfALevel
         *            the level to build on
         * @param level
         *            the level of the Boundables, or -1 if the childs are item
         *            childs (that is, below level 0)
         * @return the root, which may be a ParentNode or a LeafNode
        """
        parentBoundables = self._createParentBoundables(childsOfALevel, level + 1)
        if len(parentBoundables) == 1:
            # AbstractNode
            return parentBoundables[0]
        return self._createHigherLevels(parentBoundables, level + 1)

    def _sortBoundables(self, input):
        raise NotImplementedError()

    def remove(self, searchBounds, node, item=None):

        if item is None:

            if not self._built:
                self.build()

            if self._intersectOp.intersects(self.root.bounds, searchBounds):
                return self.remove(searchBounds, self.root, node)

        else:
            if self._removeItem(node, item):
                return True
            # BoundableList
            childs = node.childs
            for i, child in enumerate(childs):

                if not self._intersectOp.intersects(child.bounds, searchBounds):
                    continue

                if type(child).__name__ == 'AbstractNode':
                    if self.remove(searchBounds, child, item):
                        if len(child.childs) == 0:
                            childs.pop(i)
                        return True
        return False

    def _removeItem(self, node, item):

        # BoundableList
        childs = node.childs
        childToRemove = None

        for i, child in enumerate(childs):
            if type(child).__name__ == 'ItemBoundable':
                if child.item == item:
                    childToRemove = i

        if childToRemove is not None:
            node.childs.pop(i)
            return True

    def _createNode(self, level: int):
        raise NotImplementedError()

    def _createParentBoundables(self, childs: list, newLevel: int) -> list:
        """
         * Sorts the childs then divides them into groups of size M, where
         * M is the node capacity.
        """
        # BoundableList
        parentBoundables = []
        parentBoundables.append(self._createNode(newLevel))
        sortedChilds = self._sortBoundables(childs)
        for child in sortedChilds:
            # AbstractNode
            last = self._lastNode(parentBoundables)
            if len(last.childs) == self.nodeCapacity:
                last = self._createNode(newLevel)
                parentBoundables.append(last)
            last.addChild(child)
            
        return parentBoundables

    def _lastNode(self, nodeList: list):
        return nodeList[-1]

    def insert(self, bounds, item):
        self.itemBoundables.append(ItemBoundable(bounds, item))

    def query(self, searchBounds, foundItems):
        # Also builds the tree, if necessary.
        if not self._built:
            self.build()
        if self._intersectOp.intersects(self.root.bounds, searchBounds):
            self._query(searchBounds, self.root, foundItems)
            
    def visit(self, searchBounds, visitor):
        # Also builds the tree, if necessary.
        if not self._built:
            self.build()
        if self._intersectOp.intersects(self.root.bounds, searchBounds):
            self._visit(searchBounds, self.root, visitor)
    
    def build(self):
        """
         * Creates parent nodes, grandparent nodes, and so forth up to the root
         * node, for the data that has been inserted into the tree. Can only be
         * called once, and thus can be called only after all of the data has been
         * inserted into the tree.
        """
        if self._built:
            return

        if len(self.itemBoundables) == 0:
            self.root = self._createNode(0)
        else:
            self.root = self._createHigherLevels(self.itemBoundables, -1)
        self._built = True

    def _visit(self, searchBounds, node, visitor):
        childs = node.childs
        for child in childs:
            if not self._intersectOp.intersects(child.bounds, searchBounds):
                continue

            cls = type(child)
            if issubclass(cls, AbstractNode):
                self._visit(searchBounds, child, visitor)
            elif issubclass(cls, ItemBoundable):
                visitor.visiteItem(child.item)

    def _query(self, searchBounds, node, matches):

        childs = node.childs
        for child in childs:
            if not self._intersectOp.intersects(child.bounds, searchBounds):
                continue

            cls = type(child)
            if issubclass(cls, AbstractNode):
                self._query(searchBounds, child, matches)
            elif issubclass(cls, ItemBoundable):
                matches.append(child.item)

    def iterate(self, visitor):
        """
         * Iterate over all items added thus far.  Explicitly does not build
         * the tree.
        """
        boundables = self.itemBoundables
        for boundable in boundables:
            cls = type(boundable)
            if issubclass(cls, ItemBoundable):
                visitor.visitItem(boundable.item)

    def boundablesAtLevel(self, level):
        boundables = []
        self._boundablesAtLevel(level, self._root, boundables)
        return boundables

    def _boundablesAtLevel(self, level, top, boundables):

        if top.level == level:
            boundables.append(top)
            return

        childs = top.childs
        for child in childs:

            cls = type(child)

            if issubclass(cls, AbstractNode):
                self.boundablesAtLevel(level, child, boundables)

            elif issubclass(cls, ItemBoundable):
                if level == -1:
                    boundables.append(child)

        return

    def childsAtLevel(self, level, top, childs):
        """
         * @param level -1 to get items
        """
        raise NotImplementedError()

    def itemsTree(self):
        """
         * Gets a tree structure (as a nested list)
         * corresponding to the structure of the items and nodes in this tree.
         * <p>
         * The returned {List}s contain either {Object} items,
         * or Lists which correspond to subtrees of the tree
         * Subtrees which do not contain any items are not included.
         * <p>
         * Builds the tree if necessary.
         *
         * @note The caller is responsible for releasing the list
         *
         * @return a List of items and/or Lists
        """
        if not self._built:
            self.build()
        # ItemsList
        valuesTree = self._itemsTree(self.root)

        if valuesTree is None:
            return ItemsList()

        return valuesTree

    def _itemsTree(self, node):

        valuesTreeForNode = ItemsList()

        for child in node.childs:
            # Boundable
            cls = type(child)

            if issubclass(cls, AbstractNode):
                # ItemsList
                valuesTreeForChild = self._itemsTree(child)
                # only add if not null (which indicates an item somewhere in this tree
                if valuesTreeForChild is not None:
                    valuesTreeForNode.add(valuesTreeForChild)

            elif issubclass(cls, ItemBoundable):
                # ItemBoundable
                valuesTreeForNode.add(child.item)

        if len(valuesTreeForNode) == 0:
            return None

        return valuesTreeForNode


class STRIntersectsOp():

    def intersects(self, aBounds, bBounds):
        """
        """
        return aBounds.intersects(bBounds)


class STRAbstractNode(AbstractNode):
    def __init__(self, level, capacity):
        AbstractNode.__init__(self, level, capacity)

    def _computeBounds(self):
        self._bounds = None
        childs = self.childs
        if len(childs) == 0:
            return None
        for i, child in enumerate(childs):
            if i == 0:
                self._bounds = Envelope(child.bounds)
            else:
                self._bounds.expandToInclude(child.bounds)

        return self._bounds


class STRtree(AbstractSTRtree, SpatialIndex):
    """
     * A query-only R-tree created using the Sort-Tile-Recursive (STR) algorithm.
     * For two-dimensional spatial data.
     *
     * The STR packed R-tree is simple to implement and maximizes space
     * utilization; that is, as many leaves as possible are filled to capacity.
     * Overlap between nodes is far less than in a basic R-tree. However, once the
     * tree has been built (explicitly or on the first call to #query), items may
     * not be added or removed.
     *
     * Described in: P. Rigaux, Michel Scholl and Agnes Voisard. Spatial
     * Databases With Application To GIS. Morgan Kaufmann, San Francisco, 2002.
     *
    """
    def __init__(self, nodeCapacity: int=10):
        SpatialIndex.__init__(self)
        AbstractSTRtree.__init__(self, nodeCapacity)
        self._intersectOp = STRIntersectsOp()

    def _createParentBoundables(self, childs: list, newLevel: int) -> list:
        """
         * Creates the parent level for the given child level. First, orders the items
         * by the x-values of the midpoints, and groups them into vertical slices.
         * For each slice, orders the items by the y-values of the midpoints, and
         * group them into runs of size M (the node capacity). For each run, creates
         * @param childs BoundableList
         * @param newLevel int
        """
        minLeafCount = int(ceil(len(childs) / self.nodeCapacity))
        # BoundableList
        sortedChildBoundables = self._sortBoundables(childs)
        # BoundableList
        verticalSlices = self._verticalSlices(sortedChildBoundables, int(ceil(sqrt(minLeafCount))))
        return self._createParentBoundablesFromVerticalSlices(verticalSlices, newLevel)

    def _createParentBoundablesFromVerticalSlices(self, verticalSlices: list, newLevel: int) -> list:
        """
        """
        parentBoundables = []

        for slice in verticalSlices:
            toAdd = self._createParentBoundablesFromVerticalSlice(slice, newLevel)
            parentBoundables.extend(toAdd)

        return parentBoundables

    def _sortBoundables(self, input: list) -> list:
        return list(sorted(input, key=lambda n: (n.bounds.miny + n.bounds.maxy) / 2.0))

    def _createParentBoundablesFromVerticalSlice(self, childs: list, newLevel: int) -> list:
        return AbstractSTRtree._createParentBoundables(self, childs, newLevel)

    def _verticalSlices(self, childs: list, sliceCount: int) -> list:
        """
         * @param childs Must be sorted by the x-value of
         *        the envelope midpoints
         * @return
        """
        nchilds = len(childs)
        sliceCapacity = int(ceil(nchilds / sliceCount))
        slices = []
        
        start = 0
        end = sliceCapacity
        
        for j in range(sliceCount):
            
            if end > nchilds:
                end = nchilds
            
            slices.append(childs[start:end])
            start += sliceCapacity
            end += sliceCapacity
            
        return slices

    def _createNode(self, level: int):
        an = STRAbstractNode(level, self.nodeCapacity)
        self.nodes.append(an)
        return an

    @staticmethod    
    def avg(a: float, b: float) -> float:
        return (a + b) / 2.0

    def centreY(self, env) -> float:
        return STRtree.avg(env.miny, env.maxy)

    def nearestNeighbour(self, env, item, itemDist):
        raise NotImplementedError()

    def insert(self, env, item):
        if env.isNull:
            return
        AbstractSTRtree.insert(self, env, item)
