# Implementation in pure Python of a sweep line algorithm for line-segment
# intersection, based on the algorithmof described in the paper:
#
# Mehlhorn, K., Näher, S.(1994). Implementation of a sweep line algorithm
# for the Straight Line Segment Intersection Problem (MPI-I-94-160).
# Saarbrücken: Max-Planck-Institut für Informatik.
# https://pure.mpg.de/pubman/faces/ViewItemOverviewPage.jsp?itemId=item_1834220#
# 
# Implementation of the class <SkipList>.
#
# Adapted and simplified from py-skiplist by Alexander Zhukov,
# at https://github.com/ZhukovAlexander/py-skiplist
# He published the code under the "Do What The F*ck You Want To Public License",
# the easiest license out there. It gives the user permissions to do whatever
# they want with your code. We say "thank you"!

from itertools import chain, dropwhile, count, repeat
import random

def geometric(p):
    return (next(dropwhile(lambda _: random.randint(1, int(1. / p)) == 1, count())) for _ in repeat(1))

class NIL(object):
    """Sentinel object that always compares greater than another object"""
    __slots__ = ()

    def __cmp__(self, other):
        # None is always greater than the other
        return 1

    def __lt__(self, other):
        return False

    def __le__(self, other):
        return False

    def __ge__(self, other):
        return True

    def __str__(self):
        return 'NIL'

    def __nonzero__(self):
        return False

    def __bool__(self):
        return False

class SkipNode():
    def __init__(self, key, data, succ, prev):
        self.key = key
        self.data = data
        self.succ = succ
        self.prev = prev

        for level in range(len(prev)):
            prev[level].succ[level] = self.succ[level].prev[level] = self

class SkipList():
    distribution = geometric(0.5)
    def __init__(self, **kwargs):

        self._tail = SkipNode(NIL(), None, [], [])
        self._head = SkipNode(None, 'HEAD', [self.tail], [])
        self._tail.prev.extend([self.head])

        self._size = 0

        for k, v in kwargs.items():
            self[k] = v

    @property
    def head(self):
        return self._head

    @property
    def tail(self):
        return self._tail

    def _height(self):
        return len(self.head.succ)

    def _level(self, start=None, level=0):
        node = start or self.head.succ[level]
        while node is not self.tail:
            yield node
            node = node.succ[level]

    def _scan(self, key):
        return_value = None
        height = len(self.head.succ)
        prevs = [self.head] * height
        node = self.head.succ[-1]
        for level in reversed(range(height)):
            node = next(
                dropwhile(
                    lambda node_: node_.succ[level].key <= key,
                    chain([self.head], self._level(node, level))
                )
            )
            if node.key == key:
                return_value = node
            else:
                prevs[level] = node

        return return_value, prevs

    def _insert(self, key, data):
            # Inserts data into appropriate position.

            node, update = self._scan(key)

            if node:
                node.data = data
                return node

            node_height = next(self.distribution) + 1 
            # because height should be positive non-zero
            # if node's height is greater than number of levels
            # then add new levels, if not do nothing
            height = len(self.head.succ)

            update.extend([self.head for _ in range(height, node_height)])

            self.head.succ.extend([self.tail for _ in range(height, node_height)])

            self.tail.prev.extend([self.head for _ in range(height, node_height)])

            new_node = SkipNode(key, data, [update[l].succ[l] for l in range(node_height)], [update[l] for l in range(node_height)])
            self._size += 1
            return new_node

    def _remove(self, key):
        # Removes node with given data. No operation if data is not in list.

        node, update = self._scan(key)
        if not node:
            return

        for level in range(len(node.succ)):
            update[level].succ[level] = node.succ[level]

        # trim not used head pointers
        for i in reversed(range(len(self.head.succ))):
            if self.head.succ[i] != self.tail:
                break
            elif i > 0:  # at least one pointer
                head_node = self.head.succ.pop()
                del head_node

        del node
        self._size -= 1

    def __len__(self):
        return self._size

    def __repr__(self):
        return 'skiplist({{{}}})'.format(
            ', '.join('{key}: {value}'.format(key=node.key, value=node.data) for node in self._level())
        )

    def __getitem__(self, key):
        # Returns item with given index
        node, _ = self._scan(key)
        if node is None:
            return None
        return node.data

    def __setitem__(self, key, value):
        return self._insert(key, value)

    def __delitem__(self, key):
        self._remove(key)

    def __iter__(self):
        # Iterate over keys in sorted order
        return (node.key for node in self._level())

    def iteritems(self):
        return ((node.key, node.data) for node in self._level())

    def iterkeys(self):
        return (item[0] for item in self.iteritems())

    def itervalues(self):
        return (item[1] for item in self.iteritems())
