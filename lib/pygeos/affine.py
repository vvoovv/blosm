# -*- coding:utf-8 -*-

# Copyright (c) 2007, Sean C. Gillies
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
# * Neither the name of Sean C. Gillies nor the names of
# its contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

# <pep8 compliant>


from .shared import Coordinate, GeomTypeId


def affine_transform(geom, matrix):
    """Returns a transformed geometry using an affine transformation matrix.
    The coefficient matrix is provided as a list or tuple with 6 or 12 items
    for 2D or 3D transformations, respectively.
    For 2D affine transformations, the 6 parameter matrix is.
        [a, b, d, e, xoff, yoff]
    which represents the augmented matrix.
                            / a  b xoff \ 
        [x' y' 1] = [x y 1] | d  e yoff |
                            \ 0  0   1  /
    or the equations for the transformed coordinates.
        x' = a * x + b * y + xoff
        y' = d * x + e * y + yoff
    For 3D affine transformations, the 12 parameter matrix is.
        [a, b, c, d, e, f, g, h, i, xoff, yoff, zoff]
    which represents the augmented matrix.
                                 / a  b  c xoff \ 
        [x' y' z' 1] = [x y z 1] | d  e  f yoff |
                                 | g  h  i zoff |
                                 \ 0  0  0   1  /
    or the equations for the transformed coordinates.
        x' = a * x + b * y + c * z + xoff
        y' = d * x + e * y + f * z + yoff
        z' = g * x + h * y + i * z + zoff
    """
    if geom.is_empty:
        return geom
    if len(matrix) == 6:
        ndim = 2
        a, b, d, e, xoff, yoff = matrix
        if geom.has_z:
            ndim = 3
            i = 1.0
            c = f = g = h = zoff = 0.0
            matrix = a, b, c, d, e, f, g, h, i, xoff, yoff, zoff
    elif len(matrix) == 12:
        ndim = 3
        a, b, c, d, e, f, g, h, i, xoff, yoff, zoff = matrix
        if not geom.has_z:
            ndim = 2
            matrix = a, b, d, e, xoff, yoff
    else:
        raise ValueError("'matrix' expects either 6 or 12 coefficients")

    def affine_pts(pts):
        """Internal function to yield affine transform of coordinate"""
        if ndim == 2:
            for pt in pts:
                x, y = pt.x, pt.y
                xp = a * x + b * y + xoff
                yp = d * x + e * y + yoff
                yield Coordinate(xp, yp)
        elif ndim == 3:
            for pt in pts:
                x, y, z = pt.x, pt.y, pt.z
                xp = a * x + b * y + c * z + xoff
                yp = d * x + e * y + f * z + yoff
                zp = g * x + h * y + i * z + zoff
                yield Coordinate(xp, yp, zp)

    # Process coordinates from each supported geometry type
    if geom.type_id == GeomTypeId.GEOS_POINT:
        return geom._factory.createPoint(list(affine_pts(geom.coords)))
    
    elif geom.type_id == GeomTypeId.GEOS_LINESTRING:
        return geom._factory.createLineString(list(affine_pts(geom.coords)))
    
    elif geom.type_id == GeomTypeId.GEOS_LINEARRING:
        return geom._factory.createLinearRing(list(affine_pts(geom.coords)))
        
    elif geom.type_id == GeomTypeId.GEOS_POLYGON:
        ring = geom.exterior
        shell = geom._factory.createLinearRing(list(affine_pts(ring.coords)))
        holes = list(geom.interiors)
        for pos, ring in enumerate(holes):
            holes[pos] = geom._factory.createLinearRing(list(affine_pts(ring.coords)))
        return geom._factory.createPolygon(shell, holes)
    
    elif geom.type_id in [
            GeomTypeId.GEOS_MULTIPOINT,
            GeomTypeId.GEOS_MULTILINESTRING,
            GeomTypeId.GEOS_MULTIPOLYGON,
            GeomTypeId.GEOS_GEOMETRYCOLLECTION
            ]:
        return geom._factory.buildGeometry([affine_transform(part, matrix)
                           for part in geom.geoms])
    else:
        raise ValueError('Type %r not recognized' % geom.geom_id)
