# -*- coding:utf-8 -*-

# ##### BEGIN LGPL LICENSE BLOCK #####
# GEOS - Geometry Engine Open Source
# http:#geos.osgeo.org
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


from .shared import quicksort


class SortedPackedIntervalRTree():
    """
     * A static index on a set of 1-dimensional intervals,
     * using an R-Tree packed based on the order of the interval midpoints.
     *
     * It supports range searching,
     * where the range is an interval of the real line (which may be a single point).
     * A common use is to index 1-dimensional intervals which
     * are the projection of 2-D objects onto an axis of the coordinate system.
     *
     * This index structure is <i>static</i>
     * - items cannot be added or removed once the first query has been made.
     * The advantage of this characteristic is that the index performance
     * can be optimized based on a fixed set of items.
     *
     * @author Martin Davis
     *
    """
    def __init__(self):

        # IntervalRTreeNode.ConstVect
        self.leaves = []
        # IntervalRTreeNode
        self.root = None
        # int
        self.level = 0

    def init(self) -> None:
        if self.root is not None:
            return
        self.root = self.buildTree()

    def buildLevel(self, src, dest) -> None:
        self.level += 1
        dest.clear()
        ni = len(src)
        for i in range(0, ni, 2):
            n1 = src[i]
            if i + 1 < ni:
                n2 = src[i + 1]
                node = IntervalRTreeBranchNode(n1, n2)
                dest.append(node)
            else:
                dest.append(n1)

    # IntervalRTreeNode
    def buildTree(self):
        quicksort(self.leaves, IntervalRTreeNode.compare)
        src = self.leaves
        dest = []
        while(True):
            self.buildLevel(src, dest)

            if len(dest) == 1:
                return dest[0]

            src, dest = dest, src

    def insert(self, mini: float, maxi: float, item=None)-> None:
        """
         * Adds an item to the index which is associated with the given interval
         *
         * @param mini the lower bound of the item interval
         * @param maxi the upper bound of the item interval
         * @param item the item to insert, ownership left to caller
         *
         * @throw IllegalStateException if the index has already been queried
        """
        if self.root is not None:
            raise ValueError("Index cannot be added to once it has been queried")
        node = IntervalRTreeLeafNode(mini, maxi, item)
        self.leaves.append(node)

    def query(self, mini: float, maxi: float, visitor)-> None:
        """
         * Search for intervals in the index which intersect the given closed interval
         * and apply the visitor to them.
         *
         * @param mini the lower bound of the query interval
         * @param maxi the upper bound of the query interval
         * @param visitor the visitor to pass any matched items to
        """
        self.init()
        self.root.query(mini, maxi, visitor)


class IntervalRTreeNode():

    def __init__(self, mini: float=0, maxi: float=0):
        self.mini = mini
        self.maxi = maxi

    def intersects(self, queryMin: float, queryMax: float) -> bool:
        if self.mini > queryMax or self.maxi < queryMin:
            return False
        return True

    @staticmethod
    def compare(n1, n2) -> bool:
        mid1 = (n1.mini + n1.maxi) / 2
        mid2 = (n2.mini + n2.maxi) / 2
        return mid1 > mid2


class IntervalRTreeLeafNode(IntervalRTreeNode):
    def __init__(self, mini: float=0, maxi: float=0, item=None):
        IntervalRTreeNode.__init__(self, mini, maxi)
        self.item = item

    def query(self, mini: float, maxi: float, visitor)-> None:
        if not self.intersects(mini, maxi):
            return
        visitor.visitItem(self.item)


class IntervalRTreeBranchNode(IntervalRTreeNode):

    def __init__(self, n1=None, n2=None):
        IntervalRTreeNode.__init__(self, min(n1.mini, n2.mini), max(n1.maxi, n2.maxi))
        self.node1 = n1
        self.node2 = n2

    def query(self, mini: float, maxi: float, visitor)-> None:
        if not self.intersects(mini, maxi):
            return

        if self.node1 is not None:
            self.node1.query(mini, maxi, visitor)

        if self.node2 is not None:
            self.node2.query(mini, maxi, visitor)
