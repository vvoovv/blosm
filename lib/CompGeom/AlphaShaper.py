# Implementation of an alpha-shape algorithm, either with fixed alpha or
# automatic version similar to algorithm proposed by  Edelsbrunner,
# Kirkpatrick & Seidel (1983). The code does not use scipy nor shapely,
# so that it can be used within Blender. Points are expected as instances
# of mathutils.Vector (Blender) or as a two-dimensional class that
# supports the property <length>.

import sys
from operator import itemgetter

# From BLOSM library:
from lib.CompGeom.delaunay_voronoi import computeDelaunayTriangulation
from lib.CompGeom.centerline import pointInPolygon

EPS = sys.float_info.min


# helper functions ------------------------------------------------------------
def _radiusCircumCircle(a, b, c):
    # Computes the radius of a circumscribed circle of the triangle with the
    # corners a, b, c. The corners are expected to be instances of mathutils.Vector.
    ab = (a - b).length
    bc = (b - c).length
    ca = (c - a).length

    # If a, b and c are collinear, the area of the triangle is zero
    # https://hratliff.com/files/curvature_calculations_and_circle_fitting.pdf
    area = abs((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]))
    if area < 1.0e-5:  # Triangle sides are almost collinear
        return max(ab, bc, ca) / 2.0

    num = ab * bc * ca
    return num / area / 2.0


# -----------------------------------------------------------------------------


class AlphaShaper(object):
    def __init__(self, points):
        # points: list of points, expected to be instances of mathutils.Vector.
        self.points = points
        self.radii = None
        self.simplices = None
        self.faces = []
        self.edgeSet = None

    def alphaShape(self, alpha):
        # Compute alpha shape for points using a given alpha.
        # Returns Python list of rings, that are lists of indices of
        # shape points in <self.points>.
        if len(self.points) < 4:
            return [self.points]

        # Triangulate points
        self.simplices = computeDelaunayTriangulation(self.points)
        # Computes the radiii of the circumscribed circles of the triangles
        self.radii = [
            _radiusCircumCircle(self.points[i0], self.points[i1], self.points[i2])
            for i0, i1, i2 in self.simplices
        ]

        # Apply alpha shape rules and create rings
        rings = self.alphaRings(alpha)

        return rings

    def alphaRings(self, alpha):
        self.faces = []

        # Apply alpha shape rule, select triangle faces of shape
        for radius, (i0, i1, i2) in zip(self.radii, self.simplices):
            if radius < 1.0 / alpha:
                self.faces.extend([(i0, i1), (i1, i2), (i2, i0)])

        # sort face indices along rows and then by first index
        self.faces = sorted(
            [tuple(sorted(row)) for row in self.faces], key=itemgetter(0, 1)
        )

        # Inner edges are now duplicates, remove them and keep boundary faces.
        self.faces = [unique for unique in self.faces if self.faces.count(unique) == 1]

        # Collect and order faces to rings
        rings = self.createRings()
        return rings

    def alphaShapeAuto(self, step=1):
        # Compute alpha shape for points using an automatically computed alpha,
        # similar to algorithm proposed by  Edelsbrunner, Kirkpatrick & Seidel (1983).
        # Returns Python list of rings, that are lists of indices of
        # shape points in <self.points>.
        if len(self.points) < 4:
            return self.points

        # Triangulate points
        self.simplices = computeDelaunayTriangulation(self.points)
        # Computes the radiii of the circumscribed circles of the triangles
        self.radii = [
            _radiusCircumCircle(self.points[i0], self.points[i1], self.points[i2])
            for i0, i1, i2 in self.simplices
        ]

        # Apply alpha shape rules and create rings for an alpha,
        # given the largest triangle.
        prevRings = self.alphaRings(1.0 / max(self.radii) - EPS)

        # reversed argsort for radii, index of largest radius first
        radii_R_i = sorted(
            range(len(self.radii)), key=lambda indx: self.radii[indx], reverse=True
        )  # argsort

        for i in range(0, len(radii_R_i), step):
            # Determine alpha from this ith radius in radii_R_i.
            k = radii_R_i[i]
            alpha = (1 / self.radii[k]) - EPS

            # Apply alpha shape rules and create rings
            rings = self.alphaRings(alpha)

            # Check validity of the new rings.
            noHoles = len(rings) == 1
            poly = [self.points[i] for i in rings[0]]
            noDuplicates = len(poly) == len(set(poly))
            allPointsIn = all(
                pointInPolygon(poly, p) in ("IN", "ON") for p in self.points
            )
            if noHoles and noDuplicates and allPointsIn:
                # Keep valid ring, try with smaller triangle radius.
                prevRings = rings
            else:
                # No mor valid shape, return previous solution.
                break

        # Return valid ring (without holes)
        return prevRings[0]

    def createRings(self):
        # Collect and order faces to rings
        self.edgeSet = self.faces.copy()
        rings = []
        while self.edgeSet:
            ring = []
            e0, e1 = self.edgeSet.pop()
            ring.extend([e0, e1])
            firstP = e0
            while self.edgeSet:
                fwd = [
                    j for (x, j) in self.edgeSet if x == e1
                ]  # forward edge, index e1 is at first position
                rev = [
                    j for (j, x) in self.edgeSet if x == e1
                ]  # reverse edge, index e1 is at last position
                if fwd:  # there is an edge that directly continues ring
                    e0, e1 = e1, fwd[0]
                    self.edgeSet.remove((e0, e1))
                    ring.append(e1)
                elif rev:
                    e1, e0 = rev[0], e1
                    self.edgeSet.remove((e1, e0))
                    ring.append(e1)
                else:  # can't complete ring
                    break

                if e1 == firstP:
                    break

            rings.append(ring[:-1])
        return rings
