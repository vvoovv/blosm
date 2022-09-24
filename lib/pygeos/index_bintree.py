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


from math import log, floor


class Interval():
    """
     * Represents an (1-dimensional) closed interval on the Real number line.
    """
    def __init__(self, nmin=None, nmax=None):

        if nmin is None:
            mi, ma = 0.0, 0.0
        elif nmax is None:
            mi, ma = nmin.mini, nmin.maxi
        else:
            mi, ma = nmin, nmax

        if mi > ma:
            mi, ma = ma, mi

        self.mini = mi
        self.maxi = ma

    def getWidth(self):
        return self.maxi - self.mini

    def expandToInclude(self, interval):
        if interval.maxi > self.maxi:
            self.maxi = interval.maxi
        if interval.mini < self.mini:
            self.mini = interval.mini

    def overlaps(self, nmin, nmax=None):
        if nmax is None:
            nmin, nmax = nmin.mini, nmin.maxi

        if self.mini > nmax or self.maxi < nmin:
            return False
        return True

    def contains(self, nmin, nmax=None):
        if nmax is None:
            cls = type(nmin).__name__
            if cls == 'Interval':
                nmin, nmax = nmin.mini, nmin.maxi
            else:
                return self.mini <= nmin <= self.maxi
        return self.mini <= nmin and nmax <= self.maxi


class Key():

    def __init__(self, newInterval):
        self._interval = None
        self._pt = 0.0
        self._level = 0
        self.computeKey(newInterval)

    def computeLevel(self, newInterval):
        dx = newInterval.getWidth()
        if dx <= 0:
            level = 1
        elif dx < 1:
            level = 1 + int((log(dx) / log(2.0)) - 0.9)
        else:
            level = 1 + int((log(dx) / log(2.0)) + 0.00000000001)
        return level

    def computeKey(self, itemInterval):
        """
         * return a square envelope containing the argument envelope,
         * whose extent is a power of two and which is based at a power of 2
        """
        self._level = self.computeLevel(itemInterval)
        self._interval = Interval()
        self.computeInterval(self._level, itemInterval)
        while (not self._interval.contains(itemInterval)):
            self._level += 1
            self.computeInterval(self._level, itemInterval)

    def computeInterval(self, level, itemInterval):
        size = pow(2.0, level)
        self._pt = floor(itemInterval.mini / size) * size
        self._interval.__init__(self._pt, self._pt + size)


class NodeBase():
    """
     * The base class for nodes in a Bintree.
    """
    def __init__(self):

        self._items = []

        # subnodes are numbered ast follows:
        # 0 | 1
        # Node
        self._subnode = [None] * 2

    def _isSearchMatch(self, interval):
        return True

    @staticmethod
    def getSubnodeIndex(interval, centre):
        """
         * Returns the index of the subnode that wholely contains the given interval.
         * If none does, returns -1.
        """
        subnodeIndex = -1
        if interval.mini >= centre:
            subnodeIndex = 1
        if interval.maxi <= centre:
            subnodeIndex = 0
        return subnodeIndex

    def getItems(self):
        return self._items

    def add(self, item):
        self._items.append(item)

    def addAllItems(self, newItems):
        self._items.extend(newItems)
        for i in range(2):
            if self._subnode[i] is not None:
                self._subnode[i].addAllItems(newItems)
        return self._items

    def addAllItemsFromOverlapping(self, interval, resultItems):
        if not self._isSearchMatch(interval):
            return self._items
        resultItems.extend(self._items)
        for i in range(2):
            if self._subnode[i] is not None:
                self._subnode[i].addAllItemsFromOverlapping(interval, resultItems)
        return self._items

    def depth(self):
        maxSubDepth = 0
        for i in range(2):
            if self._subnode[i] is not None:
                sqd = self._subnode[i].depth()
                if sqd > maxSubDepth:
                    maxSubDepth = sqd
        return maxSubDepth

    def size(self):
        subSize = 0
        for i in range(2):
            if self._subnode[i] is not None:
                subSize += self._subnode[i].size()
        return subSize + len(self._items)

    def nodeSize(self):
        subSize = 0
        for i in range(2):
            if self._subnode[i] is not None:
                subSize += self._subnode[i].nodeSize()
        return subSize + 1


class Node(NodeBase):
    """
     * A node of a Bintree.
    """
    def __init__(self, newInterval, newLevel):
        NodeBase.__init__(self)
        self._interval = newInterval
        self._centre = (newInterval.mini + newInterval.maxi) / 2.0
        self._level = newLevel

    def _getSubNode(self, index):
        if self._subnode[index] is None:
            self._subnode[index] = self._createSubNode(index)
        return self._subnode[index]

    def _createSubNode(self, index):
        mi, ma = 0.0, 0.0
        if index == 0:
            mi = self._interval.mini
            ma = self._centre
        else:
            mi = self._centre
            ma = self._interval.maxi
        subInt = Interval(mi, ma)
        return Node(subInt, self._level - 1)

    def _isSearchMatch(self, itemInterval):
        return itemInterval.overlaps(self._interval)

    def creatNode(self, itemInterval):
        key = Key(itemInterval)
        return Node(Interval(key._interval), key._level)

    def createExpanded(self, node, addInterval):
        expandInt = Interval(addInterval)

        if node is not None:
            expandInt.eypandToInclude(node._interval)

        largerNode = self.createNode(expandInt)

        if node is not None:
            largerNode.insert(node)

        return largerNode

    def getNode(self, searchInterval):

        subnodeIndex = NodeBase.getSubnodeIndex(searchInterval, self._centre)

        if subnodeIndex > -1:
            node = self._getSubNode(subnodeIndex)
            return node.getNode(searchInterval)
        else:
            return self

    def find(self, searchInterval):

        subnodeIndex = NodeBase.getSubnodeIndex(searchInterval, self._centre)

        if subnodeIndex == -1:
            return self

        if self._subnode[subnodeIndex] is not None:
            node = self._subnode[subnodeIndex]
            return node.find(searchInterval)

        return self

    def insert(self, node):

        index = NodeBase.getSubnodeIndex(node._interval, self._centre)

        if node._level == self._level - 1:
            self._subnode[index] = node
        else:
            childNode = self._createSubNode(index)
            childNode.intert(node)
            self._subnode[index] = childNode


class Root(NodeBase):
    """
     * The root node of a single Bintree.
     *
     * It is centred at the origin,
     * and does not have a defined extent.
    """
    def __init__(self):
        NodeBase.__init__(self)
        self._origin = 0

    def _insertContained(self, tree, itemInterval, item):
        """
         * Do NOT create a new node for zero-area intervals - this would lead
         * to infinite recursion. Instead, use a heuristic of simply returning
         * the smallest existing node containing the query
        """
        isZeroArea = itemInterval.mini - itemInterval.maxi == 0
        if isZeroArea:
            node = tree.find(itemInterval)
        else:
            node = tree.getNode(itemInterval)
        node.add(item)

    def insert(self, itemInterval, item):
        """
         * @param itemInterval
         * @param item
        """
        index = NodeBase.getSubnodeIndex(itemInterval, self._origin)
        # if index is -1, itemEnv must contain the origin.
        if index == -1:
            self.add(item)
            return
        # the item must be contained in one interval, so insert it into the
        # tree for that interval (which may not yet exist)
        node = self._subnode[index]
        if node is None or not node._interval.contains(itemInterval):
            largerNode = Node.createExpanded(self, node, itemInterval)
            self._subnode[index] = largerNode
        self._insertContained(self._subnode[index], itemInterval, item)


class Bintree():
    """
     * An BinTree (or "Binary Interval Tree")
     * is a 1-dimensional version of a quadtree.
     *
     * It indexes 1-dimensional intervals (which of course may
     * be the projection of 2-D objects on an axis).
     * It supports range searching
     * (where the range may be a single point).
     *
     * This implementation does not require specifying the extent of the inserted
     * items beforehand.  It will automatically expand to accomodate any extent
     * of dataset.
     *
     * This index is different to the Interval Tree of Edelsbrunner
     * or the Segment Tree of Bentley.
    """
    def __init__(self):
        # Interval
        self._newIntervals = []
        self._root = Root()
        """
         *  Statistics
         *
         * minExtent is the minimum extent of all items
         * inserted into the tree so far. It is used as a heuristic value
         * to construct non-zero extents for features with zero extent.
         * Start with a non-zero extent, in case the first feature inserted has
         * a zero extent in both directions.  This value may be non-optimal, but
         * only one feature will be inserted with this value.
        """
        self._minExtent = 1.0

    def depth(self):
        if self._root is not None:
            return self._root.depth()
        return 0

    def size(self):
        if self._root is not None:
            return self._root.size()
        return 0

    def nodeSize(self):
        if self._root is not None:
            return self._root.nodeSize()
        return 0

    def ensureExtent(self, itemInterval, minExtent):
        """
         * Ensure that the Interval for the inserted item has non-zero extents.
         * Use the current minExtent to pad it, if necessary
         *
         * NOTE: in GEOS this function always return a newly allocated object
         *       with ownership transferred to caller. TODO: change this ?
         *
         * @param itemInterval
         *      Source interval, ownership left to caller, no references hold.
        """
        min = itemInterval.mini
        max = itemInterval.maxi
        # has a non-zero extent
        if min != max:
            return itemInterval

        # pad extend
        if min == max:
            min = min - minExtent / 2.0
            max = min + minExtent / 2.0
        return Interval(min, max)

    def insert(self, itemInterval, item):
        self.collectStats(itemInterval)
        insertInterval = self.ensureExtent(itemInterval, self._minExtent)

        if insertInterval != itemInterval:
            self.newIntervals.append(insertInterval)

        self._root.insert(insertInterval, item)

    def query(self, x, foundItems=None):
        if foundItems is None:
            do_return = True
            foundItems = []

        if type(x).__name__ == 'Interval':
            self._root.addAllItemsFromOverlapping(x, foundItems)
        else:
            return self.query(Interval(x, x))

        if do_return:
            return foundItems

    def collectStats(self, interval):
        d = interval.getWidth()
        if d < self._minExtent and d > 0.0:
            self._minExtent = d
