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


import struct
from math import log, pow, floor
from .shared import (
    logger,
    Envelope,
    Coordinate
    )
ASSUME_IEEE_DOUBLE = True


class Quadtree():
    """
     * A Quadtree is a spatial index structure for efficient querying
     * of 2D rectangles.  If other kinds of spatial objects
     * need to be indexed they can be represented by their
     * envelopes
     *
     * The quadtree structure is used to provide a primary filter
     * for range rectangle queries.  The query() method returns a list of
     * all objects which <i>may</i> intersect the query rectangle.  Note that
     * it may return objects which do not in fact intersect.
     * A secondary filter is required to test for exact intersection.
     * Of course, this secondary filter may consist of other tests besides
     * intersection, such as testing other kinds of spatial relationships.
     *
     * This implementation does not require specifying the extent of the inserted
     * items beforehand.  It will automatically expand to accomodate any extent
     * of dataset.
     *
     * This data structure is also known as an <i>MX-CIF quadtree</i>
     * following the usage of Samet and others.
    """
    def __init__(self):
        """
         * Constructs a Quadtree with zero items.
        """
        # Root
        self.root = Root()
        """
         *  Statistics
         *
         * minExtent is the minimum envelope extent of all items
         * inserted into the tree so far. It is used as a heuristic value
         * to construct non-zero envelopes for features with zero X and/or
         * Y extent.
         * Start with a non-zero extent, in case the first feature inserted has
         * a zero extent in both directions.  This value may be non-optimal, but
         * only one feature will be inserted with this value.
        """
        self.minExtent = 1.0

    def collectStats(self, env) -> None:
        delx = env.width
        if delx < self.minExtent and delx > 0.0:
            self.minExtent = delx
        dely = env.height
        if dely < self.minExtent and dely > 0.0:
            self.minExtent = dely

    def ensureExtent(self, env, minExtent) -> None:
        """
         * Ensure that the envelope for the inserted item has non-zero extents.
         *
         * Use the current minExtent to pad the envelope, if necessary.
         * Can return a new Envelope or the given one (casted to non-const).
        """
        minx = env.minx
        maxx = env.maxx
        miny = env.miny
        maxy = env.maxy

        if minx != maxx and miny != maxy:
            return env

        if minx == maxx:
            minx = minx - minExtent / 2.0
            maxx = maxx + minExtent / 2.0

        if miny == maxy:
            miny = miny - minExtent / 2.0
            maxy = maxy + minExtent / 2.0

        return Envelope(minx, maxx, miny, maxy)

    @property
    def depth(self) -> int:
        # Returns the number of levels in the tree.
        return self.root.depth

    @property
    def size(self) -> int:
        # Returns the number of items in the tree.
        return self.root.size

    def insert(self, env, item) -> None:
        self.collectStats(env)
        insertEnv = self.ensureExtent(env, self.minExtent)
        self.root.insert(insertEnv, item)

    def query(self, env, ret: list) -> None:
        """
         * Queries the tree and returns items which may lie
         * in the given search envelope.
         *
         * Precisely, the items that are returned are all items in the tree
         * whose envelope <b>may</b> intersect the search Envelope.
         * Note that some items with non-intersecting envelopes may be
         * returned as well;
         * the client is responsible for filtering these out.
         * In most situations there will be many items in the tree which do not
         * intersect the search envelope and which are not returned - thus
         * providing improved performance over a simple linear scan.
         *
         * @param searchEnv the envelope of the desired query area.
         * @param ret a vector where items which may intersect the
         *        search envelope are pushed
        """
        self.root.addAllItemsFromOverlapping(env, ret)

    def visit(self, env, visitor) -> None:
        """
         * Queries the tree and visits items which may lie in
         * the given search envelope.
         *
         * Precisely, the items that are visited are all items in the tree
         * whose envelope <b>may</b> intersect the search Envelope.
         * Note that some items with non-intersecting envelopes may be
         * visited as well;
         * the client is responsible for filtering these out.
         * In most situations there will be many items in the tree which do not
         * intersect the search envelope and which are not visited - thus
         * providing improved performance over a simple linear scan.
         *
         * @param searchEnv the envelope of the desired query area.
         * @param visitor a visitor object which is passed the visited items
        """
        """
         * the items that are matched are the items in quads which
         * overlap the search envelope
        """
        self.root.visit(env, visitor)

    def remove(self, env, item) -> bool:
        """
         * Removes a single item from the tree.
         *
         * @param itemEnv the Envelope of the item to be removed
         * @param item the item to remove
         * @return <code>true</code> if the item was found (and thus removed)
        """
        posEnv = self.ensureExtent(env, self.minExtent)
        return self.root.remove(posEnv, item)

    def queryAll(self) -> list:
        # Return a list of all items in the Quadtree
        foundItems = []
        self.root.addAllItems(foundItems)
        return foundItems

    def __str__(self):
        return ""


class DoubleBits():
    """
     * DoubleBits manipulates Double numbers
     * by using bit manipulation and bit-field extraction.
     *
     * For some operations (such as determining the exponent)
     * this is more accurate than using mathematical operations
     * (which suffer from round-off error).
     *
     * The algorithms and constants in this class
     * apply only to IEEE-754 double-precision floating point format.
     *
    """

    EXPONENT_BIAS = 1023
    
    def __init__(self, x):
        if ASSUME_IEEE_DOUBLE:
            self.x = DoubleBits.float2bits(x)
        else:
            self.x = x
    
    @staticmethod
    def float2bits(f: float)-> int:
        s = struct.pack('>d', f)
        return struct.unpack('>q', s)[0]
    
    @staticmethod
    def bitsToFloat(b: int) -> float:
        s = struct.pack('>q', b)
        return struct.unpack('>d', s)[0]
    
    def getDouble(self) -> float:
        if ASSUME_IEEE_DOUBLE:
            return DoubleBits.bitsToFloat(self.x)
        return self.x
        
    @staticmethod
    def powerOf2(exp: int) -> float:
        
        if exp > 1023 or exp < -1022:
            raise ValueError("Exponent out of bounds valid range:[-1022 - 1023] got:{}".format(exp))
            
        if ASSUME_IEEE_DOUBLE:
            expBias = exp + DoubleBits.EXPONENT_BIAS
            res = expBias << 52
            return DoubleBits.bitsToFloat(res)
        
        return pow(2.0, exp)

    @staticmethod
    def truncateToPowerOfTwo(d: float) -> float:
        db = DoubleBits(d)
        db.zeroLowerBits(52)
        return db.getDouble()

    @staticmethod
    def maximumCommonMantissa(d1: float, d2: float) -> float:
        if d1 == 0 or d2 == 0:
            return 0.0
        db1 = DoubleBits(d1)
        db2 = DoubleBits(d2)
        if db1.getExponent() != db2.getExponent():
            return 0.0
        maxi = db1.numCommonMantissaBits(db2)
        db1.zeroLowerBits(64 - (12 + maxi))
        return db1.getDouble()

    @property
    def biasedExponent(self) -> int:
        # Determines the exponent for the number
        signExp = self.x >> 52
        exp = signExp & 0x07ff
        return exp

    def getExponent(self) -> int:
        # Determines the exponent for the number
        if ASSUME_IEEE_DOUBLE:
            return self.biasedExponent - DoubleBits.EXPONENT_BIAS

        if self.x <= 0:
            return 0
        if self.x < 1:
            return int(log(self.x) / log(2)) - 0.9
        else:
            return int(log(self.x) / log(2)) + 0.00000000001

    @staticmethod
    def exponent(x) -> int:
        db = DoubleBits(x)
        return db.getExponent()

    def zeroLowerBits(self, nBits: int) -> None:
        invMask = (1 << nBits) - 1
        mask = ~invMask
        self.x &= mask

    def getBit(self, i: int) -> int:
        mask = 1 << i
        if (self.x & mask) != 0:
            return 1
        return 0

    def numCommonMantissaBits(self, db) -> int:
        """
         * This computes the number of common most-significant bits in
         * the mantissa.
         *
         * It does not count the hidden bit, which is always 1.
         * It does not determine whether the numbers have the same exponent;
         * if they do not, the value computed by this function is meaningless.
         *
         * @param db
         *
         * @return the number of common most-significant mantissa bits
        """
        for i in range(52):
            if self.getBit(i) != db.getBit(i):
                return i
        return 52

    def __str__(self):
        # A representation of the Double bits formatted for easy readability
        return ""


class IntervalSize():
    """
     * Provides a test for whether an interval is
     * so small it should be considered as zero for the purposes of
     * inserting it into a binary tree.
     *
     * The reason this check is necessary is that round-off error can
     * cause the algorithm used to subdivide an interval to fail, by
     * computing a midpoint value which does not lie strictly between the
     * endpoints.
    """

    """
     * This value is chosen to be a few powers of 2 less than the
     * number of bits available in the double representation (i.e. 53).
     * This should allow enough extra precision for simple computations
     * to be correct, at least for comparison purposes.
    """
    MIN_BINARY_EXPONENT = -50

    @staticmethod
    def isZeroWidth(mini: float, maxi: float) -> bool:
        """
         * Computes whether the interval [min, max] is effectively zero width.
         * I.e. the width of the interval is so much less than the
         * location of the interval that the midpoint of the interval
         * cannot be represented precisely.
        """
        width = maxi - mini
        if width == 0.0:
            return True

        maxAbs = max(abs(mini), abs(maxi))
        scaledInterval = width / maxAbs
        level = DoubleBits.exponent(scaledInterval)
        return level <= IntervalSize.MIN_BINARY_EXPONENT


class Key():
    """
     * A Key is a unique identifier for a node in a quadtree.
     *
     * It contains a lower-left point and a level number. The level number
     * is the power of two for the size of the node envelope
    """
    def __init__(self, env):
        self.env = Envelope()
        self.coord = Coordinate()
        self.level = 0
        self.computeKey(env)

    def computeQuadLevel(self, env) -> int:
        dx = env.width
        dy = env.height
        if dx > dy:
            dmax = dx
        else:
            dmax = dy
        level = DoubleBits.exponent(dmax) + 1
        logger.debug("Maxdelta : %s  exponent: %s", dmax, level - 1)
        return level

    @property
    def centre(self):
        return self.env.getCentre()

    def computeKey(self, env) -> None:
        """
         * return a square envelope containing the argument envelope,
         * whose extent is a power of two and which is based at a power of 2
        """
        self.level = self.computeQuadLevel(env)
        logger.debug("computeKey env:%s", env)
        self.env.__init__()
        self._computeKey(self.level, env)
        while not self.env.contains(env):
            self.level += 1
            self._computeKey(self.level, env)

    def _computeKey(self, level, env) -> None:
        quadSize = DoubleBits.powerOf2(level)
        
        self.coord.x = floor(env.minx / quadSize) * quadSize
        self.coord.y = floor(env.miny / quadSize) * quadSize
        
        self.env.init(
            self.coord.x,
            self.coord.x + quadSize,
            self.coord.y,
            self.coord.y + quadSize)
        logger.debug("_computeKey level:%s quadSize:%s env:%s", level, quadSize, self.env)
        

class NodeBase():
    """
     * The base class for nodes in a Quadtree.
     *
    """
    def __init__(self):
        self.items = []
        """
         * subquads are numbered as follows:
         * <pre>
         *  2 | 3
         *  --+--
         *  0 | 1
         * </pre>
         *
         * Nodes are owned by this class
        """
        # Node
        self.subnode = [None, None, None, None]

    def visitItems(self, env, visitor) -> None:
        for item in self.items:
            visitor.visitItem(item)

    def visit(self, env, visitor):
        if not self.isSearchMatch(env):
            return
            
        self.visitItems(env, visitor)

        for i in range(4):
            if self.subnode[i] is not None:
                self.subnode[i].visit(env, visitor)

    @staticmethod
    def getSubnodeIndex(env, centre) -> int:
        subnodeIndex = -1
        if env.minx >= centre.x:
            if env.miny >= centre.y:
                subnodeIndex = 3
            if env.maxy <= centre.y:
                subnodeIndex = 1

        if env.maxx <= centre.x:
            if env.miny >= centre.y:
                subnodeIndex = 2
            if env.maxy <= centre.y:
                subnodeIndex = 0

        return subnodeIndex

    def add(self, item) -> None:
        self.items.append(item)

    def addAllItems(self, resultItems) -> None:
        resultItems.extend(self.items)
        for i in range(4):
            if self.subnode[i] is not None:
                self.subnode[i].addAllItems(resultItems)
        return resultItems

    def addAllItemsFromOverlapping(self, env, resultItems):
        if not self.isSearchMatch(env):
            return

        resultItems.extend(self.items)
        for i in range(4):
            if self.subnode[i] is not None:
                self.subnode[i].addAllItemsFromOverlapping(env, resultItems)
        return resultItems

    def remove(self, env, item):
        """
         * Removes a single item from this subtree.
         *
         * @param searchEnv the envelope containing the item
         * @param item the item to remove
         * @return <code>true</code> if the item was found and removed
        """
        if not self.isSearchMatch(env):
            return False

        found = False
        for i in range(4):
            if self.subnode[i] is not None:
                found = self.subnode[i].remove(env, item)
                if found:
                    if self.subnode[i].isPrunable:
                        self.subnode[i] = None
                    break
        if found:
            return True

        for i, el in enumerate(self.items):
            if el is item:
                self.items.pop(i)
                return True

        return False

    @property
    def depth(self):
        depth = 0
        for i in range(4):
            if self.subnode[i] is not None:
                sqd = self.subnode[i].depth()
                if sqd > depth:
                    depth = sqd
        return depth

    @property
    def size(self):
        size = 0
        for i in range(4):
            if self.subnode[i] is not None:
                size += self.subnode[i].size()
        return size + len(self.items)

    @property
    def numnodes(self):
        size = 0
        for i in range(4):
            if self.subnode[i] is not None:
                size += self.subnode[i].size()
        return size + 1

    @property
    def hasItems(self) -> bool:
        return len(self.items) > 0

    @property
    def hasChildren(self) -> bool:
        for i in range(4):
            if self.subnode[i] is not None:
                return True
        return False

    @property
    def isPrunable(self) -> bool:
        return not (self.hasChildren or self.hasItems)

    def isSearchMatch(self, env) -> bool:
        raise NotImplementedError()


class Node(NodeBase):
    """
     * Represents a node of a Quadtree.
     *
     * Nodes contain items which have a spatial extent corresponding to
     * the node's position in the quadtree.
     *
    """
    def __init__(self, env, level):
        NodeBase.__init__(self)
        # Envelope
        self.env = env
        # Coordinate
        self.centre = env.getCentre()
        self.level = level

    def getSubNode(self, index: int):
        """
         * Get the subquad for the index.
         * If it doesn't exist, create it.
         *
        """
        if self.subnode[index] is None:
            self.subnode[index] = self.createSubNode(index)
        return self.subnode[index]

    def createSubNode(self, index: int):
        minx, maxx, miny, maxy = 0, 0, 0, 0
        if index == 0:
            minx = self.env.minx
            maxx = self.centre.x
            miny = self.env.miny
            maxy = self.centre.y
        elif index == 1:
            minx = self.centre.x
            maxx = self.env.maxx
            miny = self.env.miny
            maxy = self.centre.y
        elif index == 2:
            minx = self.env.minx
            maxx = self.centre.x
            miny = self.centre.y
            maxy = self.env.maxy
        elif index == 3:
            minx = self.centre.x
            maxx = self.env.maxx
            miny = self.centre.y
            maxy = self.env.maxy

        env = Envelope(minx, maxx, miny, maxy)
        return Node(env, self.level - 1)

    def isSearchMatch(self, env) -> bool:
        return self.env.intersects(env)

    @staticmethod
    def createNode(env):
        # Create a node computing level from given envelope
        key = Key(env)
        envelope = Envelope(key.env)
        node = Node(envelope, key.level)
        return node

    @staticmethod
    def createExpanded(node, env):
        """
         * Create a node containing the given node and envelope
         *
         *  @param node if not null, will be inserted to the returned node
         *  @param addEnv minimum envelope to use for the node
         *
        """
        expandEnv = Envelope(env)

        if node is not None:
            expandEnv.expandToInclude(node.env)

        largerNode = Node.createNode(expandEnv)

        if node is not None:
            largerNode.insertNode(node)

        return largerNode

    def getNode(self, env):
        """
         * Returns the subquad containing the envelope.
         * Creates the subquad if
         * it does not already exist.
        """
        subnodeIndex = NodeBase.getSubnodeIndex(env, self.centre)
        if subnodeIndex != -1:
            node = self.getSubNode(subnodeIndex)
            return node.getNode(env)

        return self

    def find(self, env):
        """
         * Returns the smallest <i>existing</i>
         * node containing the envelope.
        """
        subnodeIndex = NodeBase.getSubnodeIndex(env, self.centre)
        if subnodeIndex == -1:
            return self

        if self.subnode[subnodeIndex] is not None:
            node = self.subnode[subnodeIndex]
            return node.find(env)

        return self

    def insertNode(self, node) -> None:
        index = NodeBase.getSubnodeIndex(node.env, self.centre)
        if node.level == self.level - 1:
            self.subnode[index] = node
        else:
            childNode = self.createSubNode(index)
            childNode.insertNode(node)
            self.subnode[index] = childNode

    def __str__(self) -> str:
        return ""


class Root(NodeBase):
    """
     * QuadRoot is the root of a single Quadtree.  It is centred at the origin,
     * and does not have a defined extent.
    """
    def __init__(self):
        NodeBase.__init__(self)
        self.origin = Coordinate(0.0, 0.0)

    def insertContained(self, tree, env, item) -> None:
        """
         * insert an item which is known to be contained in the tree rooted at
         * the given QuadNode root.  Lower levels of the tree will be created
         * if necessary to hold the item.
        """
        isZerox = IntervalSize.isZeroWidth(env.minx, env.maxx)
        isZeroy = IntervalSize.isZeroWidth(env.miny, env.maxy)
        if isZerox or isZeroy:
            node = tree.find(env)
        else:
            node = tree.getNode(env)
        node.add(item)

    def insert(self, env, item) -> None:
        """
         * Insert an item into the quadtree this is the root of.
        """
        index = NodeBase.getSubnodeIndex(env, self.origin)
        if index == -1:
            self.add(item)
            return

        node = self.subnode[index]

        if node is None or not node.env.contains(env):
            largerNode = Node.createExpanded(node, env)
            self.subnode[index] = largerNode

        self.insertContained(self.subnode[index], env, item)

    def isSearchMatch(self, env) -> bool:
        return True
