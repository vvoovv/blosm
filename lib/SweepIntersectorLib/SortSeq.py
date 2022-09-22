# Implementation in pure Python of a sweep line algorithm for line-segment
# intersection, based on the algorithmof described in the paper:
#
# Mehlhorn, K., Näher, S.(1994). Implementation of a sweep line algorithm
# for the Straight Line Segment Intersection Problem (MPI-I-94-160).
# Saarbrücken: Max-Planck-Institut für Informatik.
# https://pure.mpg.de/pubman/faces/ViewItemOverviewPage.jsp?itemId=item_1834220#
# 
# Implementation of the class <SortSeq>.
# This class emulates the LEDA class <sortseq> using our own code.
#

from .SkipList import SkipList, SkipNode

class SortSeq(SkipList):
    def insert(self, key, inf):
        """
        If there is a node <key,inf> in the structure, then inf is replaced by
        <inf> , otherwise a new node <key,inf> is added to the structure.
        In both cases the node is returned.
        """
        from .Segment import Segment
        node, _ = self._scan(key)
        if node:
            node.data = inf
            return node
        else:
            node = self._insert(key, inf)
            return node

    def succ(self, node):
        """
        Returns the successor node of <node> in the sequence
        containing <node>, None if there is no such node.
        """
        node = node.succ[0]
        if node == self._tail:
            return None
        else:
            return node

    def pred(self, node):
        """
        Returns the predecessor node of <node> in the sequence
        containing <node>, None if there is no such node.
        """
        node = node.prev[0]
        if node == self._head:
            return None
        else:
            return node

    def changeInf(self, node, inf):
        """
        Makes <inf> be the data of <node>.
        """
        from .Segment import Segment
        node.data = inf

    def lookup(self, key):
        """
        Returns the node with key <key>, None if there is no such item.
        """
        node, _ = self._scan(key)
        return node

    def locate(self,key):
        """
        Returns the node (key',inf) in SortSeq such that key' is minimal
        with key' >= key. None if no such node exists
        """
        _, update = self._scan(key)
        return update[0].succ[0]
 
    @staticmethod
    def key(node):
        """
        Returns the key of <node>.
        Precondition: <node> is a node in SortSeq
        """
        return node.key

    @staticmethod
    def inf(node):
        """
        Returns the element of <node>.
        Precondition: <node> is a node in SortSeq
        """
        return node.data

    def delete(self,key):
        """
        Removes the node with the key <key> from SortSeq.
        No operation if no such key exists.
        """
        self._remove(key)

    def empty(self):
        return len(self) == 0

    def min(self):
        if self.empty():
            return None
        else:
            return self.head.succ[0]

    # -----------------------------------------------------------------------
    # The following methods change the order of the structure so that it no
    # longer remains sorted. Therefore they are implemented here and
    # not in the base class <SkipList>.

    def insertAt(self, node, key, inf):
        """
        Like insert(key,inf), the node <node> gives the position of the
        node <key,inf> in the sequence.
        Precondition: <node> is a node in SortSeq with either key(node)
        is maximal with key(node) <= <key> or key(node) is minimal with 
        key(node) >= <key>.
        """
        from .Segment import Segment
        if key == node.key:
            node.data = inf
            return node

        # Not often used, so we insert with the same height as <node>
        prevNode = node if key > node.key else node.prev[0]
        succNode = prevNode.succ[0]
        newSucc = [s for s in prevNode.succ if s == succNode]
        newPrev = [s for s in succNode.prev if s == prevNode]
        new_node = SkipNode(key, inf, newSucc, newPrev)
        self._size += 1
        return new_node

    def delItem(self, node):
        """
        Removes the <node> from SortSeq containing it.
        Precondition: <node> is a node in SortSeq.
        """
        for level in range(len(node.succ)):
            node.prev[level].succ[level] = node.succ[level]
            node.succ[level].prev[level] = node.prev[level]
        # trim not used head pointers
        for i in reversed(range(len(self.head.succ))):
            if self.head.succ[i] != self.tail:
                break
            elif i > 0:  # at least one pointer
                head_node = self.head.succ.pop()
                del head_node
        del node
        self._size -= 1

    def reverseItems(self, a, b):
        """
        The subsequence of SortSeq from nodes <a> to <b> is reversed.
	    Precondition: Node <a> appears before <b> in SortSeq.
        NOTE: This operation destroys the order in the SortSeq!
        """
        while a != b:
            c = a
            a = a.succ[0]
            self.delItem(c)

            # insert c after b
            predNode = b
            succNode = b.succ[0]
            c.succ = [s for s in predNode.succ if s == succNode]
            c.prev = [s for s in succNode.prev if s == predNode]
            for level in range(len(c.prev)):
                c.prev[level].succ[level] = c.succ[level].prev[level] = c
            self._size += 1
