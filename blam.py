'''
blam - Blender Camera Calibration Tools
Copyright (C) 2012  Per Gantelius

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see http://www.gnu.org/licenses/
'''
import bpy
import mathutils
import math, cmath 

bl_info = { \
    'name': 'BLAM - The Blender camera calibration toolkit',
    'author': 'Per Gantelius',
    'version': (0, 0, 5),
    'blender': (2, 6, 0),
    'api': 41524,
    'location': 'Move Clip Editor > Tools Panel > Static Camera Calibration and 3D View > Tools Panel > Photo Modeling Tools',
    'description': 'Reconstruction of 3D geometry and estimation of camera orientation and focal length based on photographs.',
    'tracker_url': 'https://github.com/stuffmatic/blam/issues',
    'wiki_url': 'https://github.com/stuffmatic/blam/wiki',
    'support': 'COMMUNITY',
    'category': '3D View'}

'''
Public domain pure python linear algebra
stuff from http://users.rcn.com/python/download/python.htm
'''
import operator, math, random
from functools import reduce
NPRE, NPOST = 0, 0                    # Disables pre and post condition checks

def iszero(z):  return abs(z) < .000001
def getreal(z):
    try:
        return z.real
    except AttributeError:
        return z
def getimag(z):
    try:
        return z.imag
    except AttributeError:
        return 0
def getconj(z):
    try:
        return z.conjugate()
    except AttributeError:
        return z


separator = [ '', '\t', '\n', '\n----------\n', '\n===========\n' ]

class Table(list):
    dim = 1
    concat = list.__add__      # A substitute for the overridden __add__ method
    def __getslice__( self, i, j ):
        return self.__class__( list.__getslice__(self,i,j) )
    def __init__( self, elems ):
        elems = list(elems)
        list.__init__( self, elems )
        if len(elems) and hasattr(elems[0], 'dim'): self.dim = elems[0].dim + 1
    def __str__( self ):
        return separator[self.dim].join( map(str, self) )
    def map( self, op, rhs=None ):
        '''Apply a unary operator to every element in the matrix or a binary operator to corresponding
        elements in two arrays.  If the dimensions are different, broadcast the smaller dimension over
        the larger (i.e. match a scalar to every element in a vector or a vector to a matrix).'''
        if rhs is None:                                                 # Unary case
            return self.dim==1 and self.__class__( map(op, self) ) or self.__class__( [elem.map(op) for elem in self] )
        elif not hasattr(rhs,'dim'):                                    # List / Scalar op
            return self.__class__( [op(e,rhs) for e in self] )
        elif self.dim == rhs.dim:                                       # Same level Vec / Vec or Matrix / Matrix
            assert NPRE or len(self) == len(rhs), 'Table operation requires len sizes to agree'
            return self.__class__( map(op, self, rhs) )
        elif self.dim < rhs.dim:                                        # Vec / Matrix
            return self.__class__( [op(self,e) for e in rhs]  )
        return self.__class__( [op(e,rhs) for e in self] )         # Matrix / Vec
    def __mul__( self, rhs ):  return self.map( operator.mul, rhs )
    def __div__( self, rhs ):  return self.map( operator.div, rhs )
    def __sub__( self, rhs ):  return self.map( operator.sub, rhs )
    def __add__( self, rhs ):  return self.map( operator.add, rhs )
    def __rmul__( self, lhs ):  return self*lhs
    #def __rdiv__( self, lhs ):  return self*(1.0/lhs)
    def __rsub__( self, lhs ):  return -(self-lhs)
    def __radd__( self, lhs ):  return self+lhs
    def __abs__( self ): return self.map( abs )
    def __neg__( self ): return self.map( operator.neg )
    def conjugate( self ): return self.map( getconj )
    def real( self ): return self.map( getreal  )
    def imag( self ): return self.map( getimag )
    def flatten( self ):
        if self.dim == 1: return self
        return reduce( lambda cum, e: e.flatten().concat(cum), self, [] )
    def prod( self ):  return reduce(operator.mul, self.flatten(), 1.0)
    def sum( self ):  return reduce(operator.add, self.flatten(), 0.0)
    def exists( self, predicate ):
        for elem in self.flatten():
            if predicate(elem):
                return 1
        return 0
    def forall( self, predicate ):
        for elem in self.flatten():
            if not predicate(elem):
                return 0
        return 1
    def __eq__( self, rhs ):  return (self - rhs).forall( iszero )

class Vec(Table):
    def dot( self, otherVec ):  return reduce(operator.add, map(operator.mul, self, otherVec), 0.0)
    def norm( self ):  return math.sqrt(abs( self.dot(self.conjugate()) ))
    def normalize( self ):  return self * (1.0 / self.norm())
    def outer( self, otherVec ):  return Mat([otherVec*x for x in self])
    def cross( self, otherVec ):
        'Compute a Vector or Cross Product with another vector'
        assert len(self) == len(otherVec) == 3, 'Cross product only defined for 3-D vectors'
        u, v = self, otherVec
        return Vec([ u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0] ])
    def house( self, index ):
        'Compute a Householder vector which zeroes all but the index element after a reflection'
        v = Vec( Table([0]*index).concat(self[index:]) ).normalize()
        t = v[index]
        sigma = 1.0 - t**2
        if sigma != 0.0:
            t = v[index] = t<=0 and t-1.0 or -sigma / (t + 1.0)
            v = v * (1.0/ t)
        return v, 2.0 * t**2 / (sigma + t**2)
    def polyval( self, x ):
        'Vec([6,3,4]).polyval(5) evaluates to 6*x**2 + 3*x + 4 at x=5'
        return reduce( lambda cum,c: cum*x+c, self, 0.0 )
    def ratval( self, x ):
        'Vec([10,20,30,40,50]).ratfit(5) evaluates to (10*x**2 + 20*x + 30) / (40*x**2 + 50*x + 1) at x=5.'
        degree = len(self) / 2
        num, den = self[:degree+1], self[degree+1:] + [1]
        return num.polyval(x) / den.polyval(x)

class Matrix(Table):
    __slots__ = ['size', 'rows', 'cols']
    def __init__( self, elems ):
        'Form a matrix from a list of lists or a list of Vecs'
        elems = list(elems)
        Table.__init__( self, hasattr(elems[0], 'dot') and elems or map(Vec,map(tuple,elems)) )
        self.size = self.rows, self.cols = len(elems), len(elems[0])
    def tr( self ):
        'Tranpose elements so that Transposed[i][j] = Original[j][i]'
        return Mat(zip(*self))
    def star( self ):
        'Return the Hermetian adjoint so that Star[i][j] = Original[j][i].conjugate()'
        return self.tr().conjugate()
    def diag( self ):
        'Return a vector composed of elements on the matrix diagonal'
        return Vec( [self[i][i] for i in range(min(self.size))] )
    def trace( self ): return self.diag().sum()
    def mmul( self, other ):
        'Matrix multiply by another matrix or a column vector '
        if other.dim==2: return Mat( map(self.mmul, other.tr()) ).tr()
        assert NPRE or self.cols == len(other)
        return Vec( map(other.dot, self) )
    def augment( self, otherMat ):
        'Make a new matrix with the two original matrices laid side by side'
        assert self.rows == otherMat.rows, 'Size mismatch: %s * %s' % (self.size, otherMat.size)
        return Mat( map(Table.concat, self, otherMat) )
    def qr( self, ROnly=0 ):
        'QR decomposition using Householder reflections: Q*R==self, Q.tr()*Q==I(n), R upper triangular'
        R = self
        m, n = R.size
        for i in range(min(m,n)):
            v, beta = R.tr()[i].house(i)
            R -= v.outer( R.tr().mmul(v)*beta )
        for i in range(1,min(n,m)): R[i][:i] = [0] * i
        R = Mat(R[:n])
        if ROnly: return R
        Q = R.tr().solve(self.tr()).tr()       # Rt Qt = At    nn  nm  = nm
        self.qr = lambda r=0, c=self: not r and c==self and (Q,R) or Matrix.qr(self,r) #Cache result
        assert NPOST or m>=n and Q.size==(m,n) and isinstance(R,UpperTri) or m<n and Q.size==(m,m) and R.size==(m,n)
        assert NPOST or Q.mmul(R)==self and Q.tr().mmul(Q)==eye(min(m,n))
        return Q, R
    def _solve( self, b ):
        '''General matrices (incuding) are solved using the QR composition.
        For inconsistent cases, returns the least squares solution'''
        Q, R = self.qr()
        return R.solve( Q.tr().mmul(b) )
    def solve( self, b ):
        'Divide matrix into a column vector or matrix and iterate to improve the solution'
        if b.dim==2: return Mat( map(self.solve, b.tr()) ).tr()
        assert NPRE or self.rows == len(b), 'Matrix row count %d must match vector length %d' % (self.rows, len(b))
        x = self._solve( b )
        diff = b - self.mmul(x)
        maxdiff = diff.dot(diff)
        for i in range(10):
            xnew = x + self._solve( diff )
            diffnew = b - self.mmul(xnew)
            maxdiffnew = diffnew.dot(diffnew)
            if maxdiffnew >= maxdiff:  break
            x, diff, maxdiff = xnew, diffnew, maxdiffnew
            #print >> sys.stderr, i+1, maxdiff
        assert NPOST or self.rows!=self.cols or self.mmul(x) == b
        return x
    def rank( self ):  return Vec([ not row.forall(iszero) for row in self.qr(ROnly=1) ]).sum()

class Square(Matrix):
    def lu( self ):
        'Factor a square matrix into lower and upper triangular form such that L.mmul(U)==A'
        n = self.rows
        L, U = eye(n), Mat(self[:])
        for i in range(n):
            for j in range(i+1,U.rows):
                assert U[i][i] != 0.0, 'LU requires non-zero elements on the diagonal'
                L[j][i] = m = 1.0 * U[j][i] / U[i][i]
                U[j] -= U[i] * m
        assert NPOST or isinstance(L,LowerTri) and isinstance(U,UpperTri) and L*U==self
        return L, U
    def __pow__( self, exp ):
        'Raise a square matrix to an integer power (i.e. A**3 is the same as A.mmul(A.mmul(A))'
        assert NPRE or exp==int(exp) and exp>0, 'Matrix powers only defined for positive integers not %s' % exp
        if exp == 1: return self
        if exp&1: return self.mmul(self ** (exp-1))
        sqrme = self ** (exp/2)
        return sqrme.mmul(sqrme)
    def det( self ):  return self.qr( ROnly=1 ).det()
    def inverse( self ):  return self.solve( eye(self.rows) )
    def hessenberg( self ):
        '''Householder reduction to Hessenberg Form (zeroes below the diagonal)
        while keeping the same eigenvalues as self.'''
        for i in range(self.cols-2):
            v, beta = self.tr()[i].house(i+1)
            self -= v.outer( self.tr().mmul(v)*beta )
            self -= self.mmul(v).outer(v*beta)
        return self
    def eigs( self ):
        'Estimate principal eigenvalues using the QR with shifts method'
        origTrace, origDet = self.trace(), self.det()
        self = self.hessenberg()
        eigvals = Vec([])
        for i in range(self.rows-1,0,-1):
            while not self[i][:i].forall(iszero):
                shift = eye(i+1) * self[i][i]
                q, r = (self - shift).qr()
                self = r.mmul(q) + shift
            eigvals.append( self[i][i] )
            self = Mat( [self[r][:i] for r in range(i)] )
        eigvals.append( self[0][0] )
        assert NPOST or iszero( (abs(origDet) - abs(eigvals.prod())) / 1000.0 )
        assert NPOST or iszero( origTrace - eigvals.sum() )
        return Vec(eigvals)

class Triangular(Square):
    def eigs( self ):  return self.diag()
    def det( self ):  return self.diag().prod()

class UpperTri(Triangular):
    def _solve( self, b ):
        'Solve an upper triangular matrix using backward substitution'
        x = Vec([])
        for i in range(self.rows-1, -1, -1):
            assert NPRE or self[i][i], 'Backsub requires non-zero elements on the diagonal'
            x.insert(0, (b[i] - x.dot(self[i][i+1:])) / self[i][i] )
        return x

class LowerTri(Triangular):
    def _solve( self, b ):
        'Solve a lower triangular matrix using forward substitution'
        x = Vec([])
        for i in range(self.rows):
            assert NPRE or self[i][i], 'Forward sub requires non-zero elements on the diagonal'
            x.append( (b[i] - x.dot(self[i][:i])) / self[i][i] )
        return x

def Mat( elems ):
    'Factory function to create a new matrix.'
    elems = list(elems)
    m, n = len(elems), len(elems[0])
    if m != n: return Matrix(elems)
    if n <= 1: return Square(elems)
    for i in range(1, len(elems)):
        if not iszero( max(map(abs, elems[i][:i])) ):
            break
    else: return UpperTri(elems)
    for i in range(0, len(elems)-1):
        if not iszero( max(map(abs, elems[i][i+1:])) ):
            return Square(elems)
    return LowerTri(elems)


def funToVec( tgtfun, low=-1, high=1, steps=40, EqualSpacing=0 ):
    '''Compute x,y points from evaluating a target function over an interval (low to high)
    at evenly spaces points or with Chebyshev abscissa spacing (default) '''
    if EqualSpacing:
        h = (0.0+high-low)/steps
        xvec = [low+h/2.0+h*i for i in range(steps)]
    else:
        scale, base = (0.0+high-low)/2.0, (0.0+high+low)/2.0
        xvec = [base+scale*math.cos(((2*steps-1-2*i)*math.pi)/(2*steps)) for i in range(steps)]
    yvec = map(tgtfun, xvec)
    return Mat( [xvec, yvec] )

def funfit(xvec, yvec, basisfuns ):
    'Solves design matrix for approximating to basis functions'
    return Mat([ map(form,xvec) for form in basisfuns ]).tr().solve(Vec(yvec))

def polyfit(xvec, yvec, degree=2 ):
    'Solves Vandermonde design matrix for approximating polynomial coefficients'
    return Mat([ [x**n for n in range(degree,-1,-1)] for x in xvec ]).solve(Vec(yvec))

def ratfit(xvec, yvec, degree=2 ):
    'Solves design matrix for approximating rational polynomial coefficients (a*x**2 + b*x + c)/(d*x**2 + e*x + 1)'
    return Mat([[x**n for n in range(degree,-1,-1)]+[-y*x**n for n in range(degree,0,-1)] for x,y in zip(xvec,yvec)]).solve(Vec(yvec))

def genmat(m, n, func):
    if not n: n=m
    return Mat([ [func(i,j) for i in range(n)] for j in range(m) ])

def zeroes(m=1, n=None):
    'Zero matrix with side length m-by-m or m-by-n.'
    return genmat(m,n, lambda i,j: 0)

def eye(m=1, n=None):
    'Identity matrix with side length m-by-m or m-by-n'
    return genmat(m,n, lambda i,j: int(i==j))

def hilb(m=1, n=None):
    'Hilbert matrix with side length m-by-m or m-by-n.  Elem[i][j]=1/(i+j+1)'
    return genmat(m,n, lambda i,j: 1.0/(i+j+1.0))

def rand(m=1, n=None):
    'Random matrix with side length m-by-m or m-by-n'
    return genmat(m,n, lambda i,j: random.random())

'''
Generic math stuff
'''
def normalize(vec):
    l = length(vec)
    return [x / l for x in vec]

def length(vec):
    return math.sqrt(sum([x * x for x in vec]))

def dot(x, y):
    return sum([x[i] * y[i] for i in range(len(x))])

def cbrt(x): 
    if x >= 0: 
        return math.pow(x, 1.0/3.0) 
    else: 
        return -math.pow(abs(x), 1.0/3.0)
    
def polar(x, y, deg=0): # radian if deg=0; degree if deg=1 
    if deg: 
        return math.hypot(x, y), 180.0 * math.atan2(y, x) / math.pi 
    else: 
        return math.hypot(x, y), math.atan2(y, x)

def quadratic(a, b, c=None): 
    if c: # (ax^2 + bx + c = 0) 
        a, b = b / float(a), c / float(a) 
    t = a / 2.0 
    r = t**2 - b 
    if r >= 0: # real roots 
        y1 = math.sqrt(r) 
    else: # complex roots 
        y1 = cmath.sqrt(r) 
    y2 = -y1 
    return y1 - t, y2 - t    

def solveCubic(a, b, c, d):    
    cIn = [a, b, c, d]
    a, b, c = b / float(a), c / float(a), d / float(a) 
    t = a / 3.0 
    p, q = b - 3 * t**2, c - b * t + 2 * t**3 
    u, v = quadratic(q, -(p/3.0)**3) 
    if type(u) == type(0j): # complex cubic root 
        r, w = polar(u.real, u.imag) 
        y1 = 2 * cbrt(r) * math.cos(w / 3.0) 
    else: # real root 
        y1 = cbrt(u) + cbrt(v) 
    y2, y3 = quadratic(y1, p + y1**2) 
    x1 = y1 - t
    x2 = y2 - t
    x3 = y3 - t
        
    return x1, x2, x3

#helper function to determine if the current version
#of the python api represents matrices as column major
#or row major.
def arePythonMatricesRowMajor():
    v = bpy.app.version
    is262OrGreater = v[0] >= 2 and v[1] >= 62
    is260OrLess = v[0] <= 2 and v[1] <= 60
    is261 = v[0] == 2 and v[1] == 61
    rev = bpy.app.build_revision
    
    if is262OrGreater:
        return True
    
    if is260OrLess:
        return False
    
    #apparently, build_revision is not always just a number:
    #http://code.google.com/p/blam/issues/detail?id=11
    #TODO: find out what the format of bpy.app.build_revision is 
    #for now, remove anything that isn't a digit
    digits = [str(d) for d in range(9)]
    numberString = ''
    for ch in rev:
        if ch in digits:
            numberString = numberString + ch
    
    #do revision check if we're running 2.61
    #matrices are row major starting in revision r42816
    return int(numberString) >= 42816

#helper function that returns the faces of a mesh. in bmesh builds,
#this is a list of polygons, and in pre-bmesh builds this is a list
#of triangles and quads
def getMeshFaces(meshObject):
    try:
        return meshObject.data.faces
    except:
        return meshObject.data.polygons

'''
PROJECTOR CALIBRATION STUFF
'''
'''
class ProjectorCalibrationPanel(bpy.types.Panel):
    bl_label = "Video Projector Calibration"
    bl_space_type = "CLIP_EDITOR"
    bl_region_type = "TOOLS"
    
    def draw(self, context):
        scn = bpy.context.scene
        l = self.layout
        r = l.row()
        r.operator("object.create_proj_calib_win")

        r = l.row()
        r.operator("object.set_calib_window_to_clip")

        r = l.row()
        r.operator("object.set_calib_window_to_view3d")
'''
class CreateProjectorCalibrationWindowOperator(bpy.types.Operator):
    bl_idname = "object.create_proj_calib_win"
    bl_label = "Create calibration window"
    bl_description = "TODO"
    
    def execute(self, context):
        ws = bpy.context.window_manager.windows
        if len(ws) > 1:
            self.report({'ERROR'}, "Other windows exist. Close them and try again.")
            return{'CANCELLED'}
        
        return bpy.ops.screen.area_dupli('INVOKE_DEFAULT')

class SetCalibrationWindowToClipEditor(bpy.types.Operator):
    bl_idname = "object.set_calib_window_to_clip"
    bl_label = "Clip editor"
    bl_description = ""
    
    def execute(self, context):
        windows = bpy.context.window_manager.windows
        if len(windows) > 2:
            self.report({'ERROR'}, "Expected two windows. Found " + str(len(windows)))
            return{'CANCELLED'}
        
        #operate on the window with one area
        window = None
        for w in windows:
            areas = w.screen.areas
            if len(areas) == 1:
                window = w
                break
        
        if not window:
            self.report({'ERROR'}, "Could not find single area window.")
            return{'CANCELLED'}
                    
        area = window.screen.areas[0]
    
        toolsHidden = False
        propsHidden = False
                    
        for i in area.regions:
            print(i.type, i.width, i.height)
            if i.type == 'TOOLS' and i.width <= 1:
                toolsHidden = True
            elif i.type == 'TOOL_PROPS' and i.width <= 1:
                propsHidden = True
    
        area.type = "CLIP_EDITOR"
        override = {'window': window, 'screen': window.screen, 'area': area}
        if not toolsHidden:
            bpy.ops.clip.tools(override)
        if not propsHidden:
            bpy.ops.clip.properties(override)
        bpy.ops.clip.view_all(override)
        bpy.ops.clip.view_zoom_ratio(override, ratio=1)
    
        return{'FINISHED'}

class SetCalibrationWindowToView3D(bpy.types.Operator):
    bl_idname = "object.set_calib_window_to_view3d"
    bl_label = "3D view"
    bl_description = ""
    
    def execute(self, context):
        windows = bpy.context.window_manager.windows
        if len(windows) > 2:
            self.report({'ERROR'}, "Expected two windows. Found " + str(len(windows)))
            return{'CANCELLED'}
        
        #operate on the window with one area
        window = None
        for w in windows:
            areas = w.screen.areas
            if len(areas) == 1:
                window = w
                break
        
        if not window:
            self.report({'ERROR'}, "Could not find single area window.")
            return{'CANCELLED'}
        
        area = window.screen.areas[0]
        
        toolsHidden = False
        propsHidden = False
        
        for i in area.regions:
            print(i.type, i.width, i.height)
            if i.type == 'TOOLS' and i.width <= 1:
                toolsHidden = True
            elif i.type == 'TOOL_PROPS' and i.width <= 1:
                propsHidden = True

        area.type = "VIEW_3D"

        override = {'window': window, 'screen': window.screen, 'area': area}

        if not toolsHidden:
            bpy.ops.view3d.toolshelf(override)
        if not propsHidden:
            bpy.ops.view3d.properties(override)

        s3d = area.spaces.active
        s3d.region_3d.view_camera_offset[0] = 0.0
        s3d.region_3d.view_camera_offset[1] = 0.0

        bpy.ops.view3d.zoom_camera_1_to_1(override)

        return{'FINISHED'}

'''
CAMERA CALIBRATION STUFF
'''

class PhotoModelingToolsPanel(bpy.types.Panel):    
    bl_label = "Photo Modeling Tools"    
    bl_space_type = "VIEW_3D"    
    bl_region_type = "TOOLS"    
    def draw(self, context):
        scn = bpy.context.scene
        l = self.layout
        r = l.row()
        b = r.box()
        b.operator("object.compute_depth_information")
        b.prop(scn, 'separate_faces')
        r = l.row()
        b = r.box()
        
        b.operator("object.project_bg_onto_mesh")
        b.prop(scn, 'projection_method')
        #self.layout.operator("object.make_edge_x")        
        l.operator("object.set_los_scale_pivot")
        l.operator("object.place_vertex_at_cursor")

class SetLineOfSightScalePivot(bpy.types.Operator):
    bl_idname = "object.set_los_scale_pivot"    
    bl_label = "Set line of sight scale pivot"

    bl_description = "Set the pivot to the camera origin, which makes scaling equivalent to translation along the line of sight."
    
    def execute(self, context):
        bpy.ops.object.mode_set(mode='OBJECT')
        
        selStates = []
        objs = bpy.context.scene.objects
        for obj in objs:
            selStates.append(obj.select)
            obj.select = False
        
        #select the camera
        bpy.context.scene.camera.select = True
        
        #snap the cursor to the camer
        bpy.ops.view3d.snap_cursor_to_selected()
        
        #set the cursor to be the pivot
        space = bpy.context.area.spaces.active
        print(space.pivot_point)
        space.pivot_point = 'CURSOR'
        
        for i in range(len(objs)):
            obj = objs[i]
            obj.select = selStates[i]
        
        return{'FINISHED'}

class PlaceVertexAtCursor(bpy.types.Operator):
    bl_idname = "object.place_vertex_at_cursor"
    bl_label = "Place a vertex at the cursor"

    bl_description = "Start a new mesh object by placing a vertex at the cursor"
    
    def execute(self, context):
        # setting active a camera object if there is no active object
        if not context.scene.objects.active:
            for o in bpy.context.scene.objects:
                if o.data == bpy.data.cameras[0]:
                    context.scene.objects.active = o
                    break
        bpy.ops.object.mode_set(mode="OBJECT")
        
        mesh = bpy.data.meshes.new("")
        mesh.from_pydata([bpy.context.scene.cursor_location], [], [])
        mesh.update()
        obj = bpy.data.objects.new("", mesh)
        context.scene.objects.link(obj)
        bpy.ops.object.select_all(action = "DESELECT")
        obj.select = True
        context.scene.objects.active = obj
        bpy.ops.object.mode_set(mode="EDIT")
        return{'FINISHED'}
 
class ProjectBackgroundImageOntoMeshOperator(bpy.types.Operator):
    bl_idname = "object.project_bg_onto_mesh"    
    bl_label = "Project background image onto mesh"
    bl_description = "Projects the current 3D view background image onto a mesh (the active object) from the active camera."    
    
    projectorName = 'tex_projector'
    materialName = 'cam_map_material'
   
    def meshVerticesToNDC(self, cam, mesh):

        #compute a projection matrix transforming
        #points in camera space to points in NDC
        near = cam.data.clip_start
        far = cam.data.clip_end
        rs = bpy.context.scene.render
        rx = rs.resolution_x
        ry = rs.resolution_y
        tall = rx < ry
        if tall:
            fov = cam.data.angle_x
            aspect = rx / float(ry)
            h = math.tan(0.5 * fov)
            w = aspect * h
        else:
            fov = cam.data.angle_x
            aspect = ry / float(rx)
            w = math.tan(0.5 * fov)
            h = aspect * w
        
        pm = mathutils.Matrix()
        pm[0][0] = 1 / w
        pm[1][1] = 1 / h
        pm[2][2] = (near + far) / (near - far)
        pm[2][3] = 2 * near * far / (near - far)
        pm[3][2] = -1.0
        pm[3][3] = 0.0
        
        if not arePythonMatricesRowMajor():
            pm.transpose()
        
        returnVerts = []

        for v in mesh.data.vertices:
            #the vert in local coordinates
            vec = v.co.to_4d()    
            #the vert in world coordinates
            vec = mesh.matrix_world * vec
            #the vert in clip coordinates
            vec = pm * cam.matrix_world.inverted() * vec
            #the vert in normalized device coordinates
            w = vec[3]
            vec = [x / w for x in vec]
            returnVerts.append((vec[0], vec[1], vec[2]))
        
        return returnVerts
   
    def addUVsProjectedFromView(self, mesh):
        cam = bpy.context.scene.camera
        
        #get the mesh vertices in normalized device coordinates
        #as seen through the active camera
        ndcVerts = self.meshVerticesToNDC(cam, mesh)
        
        #create a uv layer
        bpy.ops.object.mode_set(mode='EDIT')
        #projecting from view here, but the current view might not
        #be the camera, so the uvs are computed manually a couple 
        #of lines down
        bpy.ops.uv.project_from_view(scale_to_bounds=True)
        bpy.ops.object.mode_set(mode='OBJECT')
        
        #set uvs to match the vertex x and y components in NDC 
        isBmesh = True
        try:
            f = mesh.data.faces
            isBmesh = False
        except:
            pass
                
        if isBmesh:
            loops = mesh.data.loops
            uvLoops = mesh.data.uv_layers[0].data
            for loop, uvLoop in zip(loops, uvLoops):    
                vIdx = loop.vertex_index
                print("loop", loop, "vertex", loop.vertex_index, "uvLoop", uvLoop)                    
                ndcVert = ndcVerts[vIdx]
                uvLoop.uv[0] = 0.5 * (ndcVert[0] + 1.0)
                uvLoop.uv[1] = 0.5 * (ndcVert[1] + 1.0)
            
        else:
            assert(len(getMeshFaces(mesh)) == len(mesh.data.uv_textures[0].data))
            for meshFace, uvFace in zip(getMeshFaces(mesh), mesh.data.uv_textures[0].data):
                faceVerts = [ndcVerts[i] for i in meshFace.vertices]
            
                for i in range(len(uvFace.uv)):
                    uvFace.uv[i][0] = 0.5 * (faceVerts[i][0] + 1.0)
                    uvFace.uv[i][1] = 0.5 * (faceVerts[i][1] + 1.0)
        
    def performSimpleProjection(self, camera, mesh, img):
        if len(mesh.material_slots) == 0:
            mat = bpy.data.materials.new(self.materialName)
            mesh.data.materials.append(mat)
        else:
            mat = mesh.material_slots[0].material   
        mat.use_shadeless = True
        mat.use_face_texture = True
        
        
                
        self.addUVsProjectedFromView(mesh)
        
        for f in mesh.data.uv_textures[0].data:
            f.image = img
    
    def performHighQualityProjection(self, camera, mesh, img):
        if len(mesh.material_slots) == 0:
            mat = bpy.data.materials.new(self.materialName)
            mesh.data.materials.append(mat)
        else:
            mat = mesh.material_slots[0].material
        mat.use_shadeless = True
        mat.use_face_texture = True 
        
        
        
        #the texture sampling is not perspective correct
        #when directly using sticky UVs or UVs projected from the view
        #this is a pretty messy workaround that gives better looking results
        self.addUVsProjectedFromView(mesh)
        
        #then create an empty object that will serve as a texture projector
        #if the mesh has a child with the name of a texture projector, 
        #reuse it
        reusedProjector = None
        for ch in mesh.children:
            if self.projectorName in ch.name:
                reusedProjector = ch
                break
        
        if reusedProjector:
            projector = reusedProjector
        else:
            bpy.ops.object.camera_add()
            projector = bpy.context.active_object
        
        bpy.context.scene.objects.active = projector
        projector.name = mesh.name + '_' + self.projectorName
        projector.matrix_world = camera.matrix_world
        projector.select = False
        projector.scale = [0.1, 0.1, 0.1]
        projector.data.lens = camera.data.lens
        projector.data.sensor_width = camera.data.sensor_width
        projector.data.sensor_height = camera.data.sensor_height
        projector.data.sensor_fit = camera.data.sensor_fit
        
        #parent the projector to the mesh for convenience
        for obj in bpy.context.scene.objects:
            obj.select = False
        
        projector.select = True
        bpy.context.scene.objects.active = mesh
        #bpy.ops.object.parent_set()
        
        #lock the projector to the mesh
        #bpy.context.scene.objects.active = projector
        #bpy.ops.object.constraint_add(type='COPY_LOCATION')
        #projector.constraints[-1].target = mesh
        
        #create a simple subdivision modifier on the mesh object. 
        #this subdivision is what alleviates the texture sampling
        #artefacts.
        bpy.context.scene.objects.active = mesh
        levels = 3
        bpy.ops.object.modifier_add()
                
        modifier = mesh.modifiers[-1]
        modifier.subdivision_type = 'SIMPLE'
        modifier.levels = levels
        modifier.render_levels = levels
        
        #then create a uv project modifier that will project the
        #image onto the subdivided mesh using our projector object.
        bpy.ops.object.modifier_add(type='UV_PROJECT')
        modifier = mesh.modifiers[-1]
        modifier.aspect_x = bpy.context.scene.render.resolution_x / float(bpy.context.scene.render.resolution_y)
        modifier.image = img
        modifier.use_image_override = True
        modifier.projectors[0].object = projector
        modifier.uv_layer = mesh.data.uv_textures[0].name
                
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.object.mode_set(mode='OBJECT')
    
    def prepareMesh(self, mesh):
        #remove all uv layers
        while len(mesh.data.uv_textures) > 0:
            bpy.ops.mesh.uv_texture_remove()
        
        #remove all modifiers
        for m in mesh.modifiers:
            bpy.ops.object.modifier_remove(modifier=m.name)
    
    def execute(self, context):
        '''
        Get the active object and make sure it is a mesh
        '''
        mesh = bpy.context.active_object

        if mesh == None:
            self.report({'ERROR'}, "There is no active object")
            return{'CANCELLED'}
        elif not 'Mesh' in str(type(mesh.data)):
            self.report({'ERROR'}, "The active object is not a mesh")
            return{'CANCELLED'}
        
        '''
        Get the current camera
        '''
        camera = bpy.context.scene.camera
        if not camera:
            self.report({'ERROR'}, "No active camera.")
            return{'CANCELLED'}    
        
        
        activeSpace = bpy.context.area.spaces.active
        
        if len(activeSpace.background_images) == 0:
            self.report({'ERROR'}, "No backround images of clips found.")
            return{'CANCELLED'}
        
        #check what kind of background we're dealing with
        bg = activeSpace.background_images[0]
        if bg.image != None:
            img = bg.image
        elif bg.clip != None:
            path = bg.clip.filepath
            #create an image texture from the (first) background clip
            try:
                img = bpy.data.images.load(path)
            except:
                self.report({'ERROR'}, "Cannot load image %s" % path)
                return{'CANCELLED'} 
        else:
            #shouldnt end up here
            self.report({'ERROR'}, "Both background clip and image are None")
            return{'CANCELLED'}
       
        #if we made it here, we have a camera, a mesh and an image.
        self.prepareMesh(mesh)
        method = bpy.context.scene.projection_method
        if method == 'hq':
            self.performHighQualityProjection(camera, mesh, img)
        elif method == 'simple':
            self.performSimpleProjection(camera, mesh, img)
        else:
            self.report({'ERROR'}, "Unknown projection method")
            return{'CANCELLED'}
        
        activeSpace.viewport_shade = 'TEXTURED'
        
        return{'FINISHED'}

class Reconstruct3DMeshOperator(bpy.types.Operator):
    bl_idname = "object.compute_depth_information"    
    bl_label = "Reconstruct 3D geometry"
    bl_description = "Reconstructs a 3D mesh with rectangular faces based on a mesh with faces lining up with the corresponding faces in the image. Relies on the active camera being properly calibrated."    
    
    def evalEq17(self, origin, p1, p2):
        a = [x - y for x, y in zip(origin, p1)]
        b = [x - y for x, y in zip(origin, p2)]
        return dot(a, b)
    
    def evalEq27(self, l):
        return self.C4 * l ** 4 + self.C3 * l ** 3 + self.C2 * l ** 2 + self.C1 * l + self.C0

    def evalEq28(self, l):
        return self.B4 * l ** 4 + self.B3 * l ** 3 + self.B2 * l ** 2 + self.B1 * l + self.B0
    
    def evalEq29(self, l):
        return self.D3 * l ** 3 + self.D2 * l ** 2 + self.D1 * l + self.D0
        
    def worldToCameraSpace(self, verts):
        
        ret = []
        for v in verts:
            #the vert in local coordinates
            vec = v.co.to_4d()    
            #the vert in world coordinates
            vec = self.mesh.matrix_world * vec
            #the vert in camera coordinates
            vec = self.camera.matrix_world.inverted() * vec     
            
            ret.append(vec[0:3])
        return ret
    
    def computeCi(self, Qab, Qac, Qad, Qbc, Qbd, Qcd):        
        self.C4 = Qad * Qbc * Qbd - Qac * Qbd ** 2
        self.C3 = Qab * Qad * Qbd * Qcd - Qad ** 2 * Qcd + Qac * Qad * Qbd ** 2 - Qad ** 2 * Qbc * Qbd - Qbc * Qbd + 2 * Qab * Qac * Qbd - Qab * Qad * Qbc
        self.C2 = -Qab * Qbd * Qcd - Qab ** 2 * Qad * Qcd + 2 * Qad * Qcd + Qad * Qbc * Qbd - 3 * Qab * Qac * Qad * Qbd + Qab * Qad ** 2 * Qbc + Qab * Qbc + Qac * Qad ** 2 - Qab ** 2 * Qac
        self.C1 = Qab ** 2 * Qcd - Qcd + Qab * Qac * Qbd - Qab * Qad * Qbc + 2 * Qab ** 2 * Qac * Qad - 2 * Qac * Qad
        self.C0 = Qac - Qab ** 2 * Qac
        
    def computeBi(self, Qab, Qac, Qad, Qbc, Qbd, Qcd):
        self.B4 = Qbd - Qbd * Qcd ** 2
        self.B3 = 2 * Qad * Qbd * Qcd ** 2 + Qab * Qcd ** 2 + Qac * Qbd * Qcd - Qad * Qbc * Qcd - 2 * Qad * Qbd - Qab
        self.B2 = - Qbd * Qcd ** 2 - Qab * Qad * Qcd ** 2 - 3 * Qac * Qad * Qbd * Qcd + Qad ** 2 * Qbc * Qcd + Qbc * Qcd - Qab * Qac * Qcd + Qad ** 2 * Qbd + Qac * Qad * Qbc + 2 * Qab * Qad
        self.B1 = 2 * Qac * Qbd * Qcd - Qad * Qbc * Qcd + Qab * Qac * Qad * Qcd + Qac ** 2 * Qad * Qbd - Qac * Qad ** 2 * Qbc - Qac * Qbc - Qab * Qad ** 2
        self.B0 = Qac * Qad * Qbc - Qac ** 2 * Qbd

    def computeQuadDepthInformation(self, qHatA, qHatB, qHatC, qHatD):

        #print()        
        #print("computeQuadDepthInformation")
        
        '''
        compute the coefficients Qij
        '''
        Qab = dot(qHatA, qHatB)
        Qac = dot(qHatA, qHatC)
        Qad = dot(qHatA, qHatD)
        
        Qba = dot(qHatB, qHatA)
        Qbc = dot(qHatB, qHatC)
        Qbd = dot(qHatB, qHatD)
        
        Qca = dot(qHatC, qHatA)
        Qcb = dot(qHatC, qHatB)
        Qcd = dot(qHatC, qHatD)
        
        #print("Qab", Qab, "Qac", Qac, "Qad", Qad)
        #print("Qba", Qba, "Qbc", Qbc, "Qbd", Qbd)
        #print("Qca", Qca, "Qcb", Qcb, "Qcd", Qcd)
        
        '''
        compute the coefficients Ci of equation (27)
        '''
        self.computeCi(Qab, Qac, Qad, Qbc, Qbd, Qcd)
        
        '''
        compute the coefficients Bi of equation (28)
        '''
        self.computeBi(Qab, Qac, Qad, Qbc, Qbd, Qcd)
        
        '''
        compute the cofficients Di of equation (29)
        '''
        self.D3 = (self.C4 / self.B4) * self.B3 - self.C3
        self.D2 = (self.C4 / self.B4) * self.B2 - self.C2
        self.D1 = (self.C4 / self.B4) * self.B1 - self.C1
        self.D0 = (self.C4 / self.B4) * self.B0 - self.C0
        #print("Di", self.D3, self.D2, self.D1, self.D0)

        '''
        solve eq 29 for lambdaD, i.e the depth in camera space of vertex D.
        '''
        roots = solveCubic(self.D3, self.D2, self.D1, self.D0)
        #print("Eq 29 Roots", roots)
        
        '''
        choose one of the three computed roots. Tan, Sullivan and Baker propose
        choosing a real root that satisfies "(27) or (28)". Since these
        equations are derived from the orthogonality equations (17) and 
        since we're interested in a quad with edges that are "as 
        orthogonal as possible", in this implementation the positive real
        root that minimizes the quad orthogonality error is chosen instead.
        '''
        
        chosenRoot = None
        minError = None
        
        #print()
        #print("Finding root")
        for root in roots:
            
            #print("Root", root)
            
            if type(root) == type(0j):
                #complex root. do nothing
                continue
            elif root <= 0:
                #non-positive root. do nothing
                continue
            
            #compute depth values lambdaA-D based on the current root
            lambdaD = root
            self.lambdaA = 1 #arbitrarily set to 1
            numLambdaA = (Qad * lambdaD - 1.0)
            denLambdaA = (Qbd * lambdaD - Qab)
            self.lambdaB = numLambdaA / denLambdaA
            numLambdaC = (Qad * lambdaD - lambdaD * lambdaD)
            denLambdaC = (Qac - Qcd * lambdaD)
            self.lambdaC = numLambdaC / denLambdaC
            self.lambdaD = lambdaD
            
            #print("lambdaA", numLambdaA, "/", denLambdaA)
            #print("lambdaC", numLambdaC, "/", denLambdaC)
            
            #compute vertex positions
            pA = [x * self.lambdaA for x in qHatA]
            pB = [x * self.lambdaB for x in qHatB]
            pC = [x * self.lambdaC for x in qHatC]
            pD = [x * self.lambdaD for x in qHatD]
            
            #compute the mean orthogonality error for the resulting quad
            meanError, maxError = self.getQuadError(pA, pB, pC, pD)
            
            if minError == None or meanError < minError:
                minError = meanError
                chosenRoot = root
        
        if chosenRoot == None:
            self.report({'ERROR'}, "No appropriate root found.")
            return #TODO cancel properly
        
        #print("Chosen root", chosenRoot)
        
        '''
        compute and return the final vertex positions from equation (16)
        '''
        lambdaD = chosenRoot
        self.lambdaA = 1 #arbitrarily set to 1
        self.lambdaB = (Qad * lambdaD - 1.0) / (Qbd * lambdaD - Qab)
        self.lambdaC = (Qad * lambdaD - lambdaD * lambdaD) / (Qac - Qcd * lambdaD)
        self.lambdaD = lambdaD
            
        pA = [x * self.lambdaA for x in qHatA]
        pB = [x * self.lambdaB for x in qHatB]
        pC = [x * self.lambdaC for x in qHatC]
        pD = [x * self.lambdaD for x in qHatD]
        
        meanError, maxError = self.getQuadError(pA, pB, pC, pD)
        #self.report({'INFO'}, "Error: " + str(meanError) + " (" + str(maxError) + ")")
            
        return [pA, pB, pC, pD]
    
    def getQuadError(self, pA, pB, pC, pD):
        orthABD = self.evalEq17(pA, pB, pD)
        orthABC = self.evalEq17(pB, pA, pC)
        orthBCD = self.evalEq17(pC, pB, pD)
        orthACD = self.evalEq17(pD, pA, pC)
        
        absErrors = [abs(orthABD), abs(orthABC), abs(orthBCD), abs(orthACD)]
        maxError = max(absErrors)
        meanError = 0.25 * sum(absErrors)    
        #print("absErrors", absErrors, "meanError", meanError)
        
        return meanError, maxError
               
    def createMesh(self, inputMesh, computedCoordsByFace, quads):
        '''
        Mesh creation is done in two steps:
        1. adjust the computed depth values so that the 
           quad vertices line up as well as possible
        2. optionally merge each set of computed quad vertices
           that correspond to a single vertex in the input mesh
        '''
        
        '''
        Step 1
        least squares minimize the depth difference
        at each vertex of each shared edge by
        computing depth factors for each quad.
        '''
        quadFacePairsBySharedEdge = {}
                
        def indexOfFace(fcs, face):
                i = 0
                for f in fcs:
                    if f == face:
                        return i
                    i = i + 1
                assert(False) #could not find the face. should not end up here...

        #loop over all edges...
        unsharedEdgeCount = 0 #the number of edges beloning to less than two faces
        quadFaces = []
        for e in inputMesh.data.edges:
            ev = e.vertices
            
            #gather all faces containing the current edge
            facesContainingEdge = []
            
            for f in getMeshFaces(inputMesh):
                matchFound = False
                fv = f.vertices
                if len(fv) != 4:
                    #ignore non-quad faces
                    continue
                
                if f not in quadFaces:
                    quadFaces.append(f)

                #if the intersection of the face vertices and the
                #print("fv", fv, "ev", ev, "len(set(fv) & set(ev))", len(set(fv) & set(ev)))
                if len(set(fv) & set(ev)) == 2:
                    matchFound = True
                
                if matchFound:
                    assert(f not in facesContainingEdge)
                    facesContainingEdge.append(f)
            
            #sanity check. an edge can be shared by at most two faces.
            assert(len(facesContainingEdge) <= 2 and len(facesContainingEdge) >= 0)
                
            edgeIsShared = (len(facesContainingEdge) == 2)
            
            if edgeIsShared:
                quadFacePairsBySharedEdge[e] = facesContainingEdge
            else:
                unsharedEdgeCount = unsharedEdgeCount + 1
        numSharedEdges = len(quadFacePairsBySharedEdge.keys())
        numQuadFaces = len(quadFaces)
        #sanity check: the shared and unshared edges are disjoint and should add up to the total number of edges
        assert(unsharedEdgeCount + numSharedEdges  == len(inputMesh.data.edges)) 
        #print("num shared edges", numSharedEdges)
        #print(quadFacePairsBySharedEdge)
        #assert(False)
        
        #each shared edge gives rise to one equation per vertex,
        #so the number of rows in the matrix is 2n, where n is
        #the number of shared edges. the number of columns is m-1
        #where m is the number of faces (the depth factor for the first
        #face is set to 1)
        k1 = 1
        firstFace = getMeshFaces(inputMesh)[0]
        numFaces = len(getMeshFaces(inputMesh))
        faces = [f for f in getMeshFaces(inputMesh)]
        matrixRows = []
        rhRows = [] #rows of the right hand side vector
        vertsToMergeByOriginalIdx = {}
        for e in quadFacePairsBySharedEdge.keys():
            pair = quadFacePairsBySharedEdge[e]
            
            #the two original mesh faces sharing the current edge
            f0 = pair[0]
            f1 = pair[1]
            
            assert(f0 in faces)
            assert(f1 in faces)
            assert(len(f0.vertices) == 4)
            assert(len(f1.vertices) == 4)
            
            #the two computed quads corresponding to the original mesh faces
            c0 = computedCoordsByFace[f0]
            c1 = computedCoordsByFace[f1]
            
            #the indices into the output mesh of the two faces sharing the current edge
            f0Idx = quads.index(c0)#indexOfFace(faces, f0)
            f1Idx = quads.index(c1)#indexOfFace(faces, f1)
            
            def getQuadVertWithMeshIdx(quad, idx):
                #print("idx", idx, "quad", quad)
                i = 0
                for p in quad:
                    if p[-1] == idx:
                        return p, i
                    i = i + 1
                assert(False) #shouldnt end up here
            
            #vij is vertex j of the current edge in face i
            #idxij is the index of vertex j in quad i (0-3)
            v00, idx00 = getQuadVertWithMeshIdx(c0, e.vertices[0])
            v01, idx01 = getQuadVertWithMeshIdx(c0, e.vertices[1])
            
            v10, idx10 = getQuadVertWithMeshIdx(c1, e.vertices[0])
            v11, idx11 = getQuadVertWithMeshIdx(c1, e.vertices[1])
            
            #vert 0 depths
            lambda00 = v00[2]
            lambda10 = v10[2]
            
            #vert 1 depths
            lambda01 = v01[2]
            lambda11 = v11[2]
            #print(faces, f0, f1)
                        
            assert(f0Idx >= 0 and f0Idx < numFaces)
            assert(f1Idx >= 0 and f1Idx < numFaces)
            
            #vert 0
            vert0MatrixRow = [0] * (numQuadFaces - 1)
            vert0RhRow = [0]
            
            if f0Idx == 0:
                vert0RhRow[0] = lambda00
                vert0MatrixRow[f1Idx - 1] = lambda10
            elif f1Idx == 0:
                vert0RhRow[0] = lambda10
                vert0MatrixRow[f10Idx - 1] = lambda00
            else:
                vert0MatrixRow[f0Idx - 1] = lambda00
                vert0MatrixRow[f1Idx - 1] = -lambda10
            
            #vert 1
            vert1MatrixRow = [0] * (numQuadFaces - 1)
            vert1RhRow = [0]
            
            if f0Idx == 0:
                vert1RhRow[0] = lambda01
                vert1MatrixRow[f1Idx - 1] = lambda11
            elif f1Idx == 0:
                vert1RhRow[0] = lambda11
                vert1MatrixRow[f10dx - 1] = lambda01
            else:
                vert1MatrixRow[f0Idx - 1] = lambda01
                vert1MatrixRow[f1Idx - 1] = -lambda11
                
            matrixRows.append(vert0MatrixRow)
            matrixRows.append(vert1MatrixRow)
            
            rhRows.append(vert0RhRow)
            rhRows.append(vert1RhRow)
            
            #store index information for vertex merging in the new mesh
            if e.vertices[0] not in vertsToMergeByOriginalIdx.keys():
                vertsToMergeByOriginalIdx[e.vertices[0]] = []
            if e.vertices[1] not in vertsToMergeByOriginalIdx.keys():
                vertsToMergeByOriginalIdx[e.vertices[1]] = []
             
            l0 = vertsToMergeByOriginalIdx[e.vertices[0]]
            l1 = vertsToMergeByOriginalIdx[e.vertices[1]]
            
            if idx00 + 4 * f0Idx not in l0:
                l0.append(idx00 + 4 * f0Idx)
            if idx10 + 4 * f1Idx not in l0:
                l0.append(idx10 + 4 * f1Idx)
    
            if idx01 + 4 * f0Idx not in l1:
                l1.append(idx01 + 4 * f0Idx)
            if idx11 + 4 * f1Idx not in l1:
                l1.append(idx11 + 4 * f1Idx)
       
        assert(len(matrixRows) == len(rhRows))
        assert(len(matrixRows) == 2 * len(quadFacePairsBySharedEdge))
        #solve for the depth factors 2,3...m
        #print("matrixRows")
        #print(matrixRows)

        #print("rhRows")
        #print(rhRows)
        
        #sanity check: the sets of vertex indices to merge should be disjoint
        for vs in vertsToMergeByOriginalIdx.values():
            
            for idx in vs:
                assert(idx >= 0 and idx < numFaces * 4)
            
            for vsRef in vertsToMergeByOriginalIdx.values():
                if vs != vsRef:
                    #check that the current sets are disjoint
                    s1 = set(vs)
                    s2 = set(vsRef)
                    assert(len(s1 & s2) == 0)
                    
        if numQuadFaces > 2:
            m = Mat(matrixRows)
            b = Mat(rhRows)
            factors = [1] + [f[0] for f in m.solve(b)]
        elif numQuadFaces == 2:
            #TODO: special case to work around a bug
            #in the least squares solver that causes
            #an infinte recursion. should be fixed in
            #the solver ideally
            f = 0.5 * (rhRows[0][0] / matrixRows[0][0] + rhRows[1][0] / matrixRows[1][0])
            factors = [1, f]
        elif numQuadFaces == 1:
            factors = [1]
        
        
        
        #print("factors", factors)
        #multiply depths by the factors computed per face depth factors
        #quads = []
        for face in computedCoordsByFace.keys():
            quad = computedCoordsByFace[face]
            idx = indexOfFace(faces, face)
            depthScale = factors[idx]
            for i in range(4):
                vert = quad[i][:3]
                #print("vert before", vert)
                vert = [x * depthScale for x in vert]
                #print("vert after", vert)
                quad[i] = vert
            #quads.append(quad)
        
        #create the actual blender mesh
        bpy.ops.object.mode_set(mode='OBJECT')
        name = inputMesh.name + '_3D'
        #print("createM  esh", points)
        me = bpy.data.meshes.new(name)    
        ob = bpy.data.objects.new(name, me)    
        ob.show_name = True    
        # Link object to scene    
        bpy.context.scene.objects.link(ob)    
        verts = []
        faces = []
        idx = 0
        for quad in quads:
            quadIdxs = []
            for vert in quad:
                verts.append(vert)
                quadIdxs.append(idx)
                idx = idx + 1
            faces.append(quadIdxs)

        '''
        Step 2:
            optionally merge vertices
        '''
        #print("in faces", [f.vertices[:] for f in self.mesh.data.faces])
        #print("in edges", [e.vertices[:] for e in self.mesh.data.edges])
        #print("out verts", verts)
        #print("out faces", faces)
        #print("out edges", [e.vertices[:] for e in me.edges])
        #print("vertsToMergeByOriginalIdx")
        #print(vertsToMergeByOriginalIdx)
        mergeVertices = not bpy.context.scene.separate_faces
        if mergeVertices:
            for vs in vertsToMergeByOriginalIdx.values():
                print("merging verts", vs)
                #merge at the mean position, which is guaranteed to
                #lie on the line of sight, since all the vertices do
                
                mean = [0, 0, 0]
                for idx in vs:
                    #print("idx", idx)
                    #print("verts", verts)
                    currVert = verts[idx]

                    for i in range(3):
                        mean[i] = mean[i] + currVert[i] / float(len(vs))
                
                for idx in vs:
                    verts[idx] = mean
        print("2")
        # Update mesh with new data    
        me.from_pydata(verts, [], faces)    
        me.update(calc_edges=True)
        ob.select = True
        bpy.context.scene.objects.active = ob

        #finally remove doubles
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.remove_doubles()
        bpy.ops.object.mode_set(mode='OBJECT')
            
        return ob
    
    def getOutputMeshScale(self, camera, inMesh, outMesh):
        inMeanPos = [0.0] * 3
        cmi = camera.matrix_world.inverted()
        mm = inMesh.matrix_world
        for v in inMesh.data.vertices:

            vCamSpace =  cmi * mm * v.co.to_4d()
            for i in range(3):
                inMeanPos[i] = inMeanPos[i] + vCamSpace[i] / float(len(inMesh.data.vertices))
        
        outMeanPos = [0.0] * 3
        for v in outMesh.data.vertices:
            for i in range(3):
                outMeanPos[i] = outMeanPos[i] + v.co[i] / float(len(outMesh.data.vertices))

        inDistance = math.sqrt(sum([x * x for x in inMeanPos]))
        outDistance = math.sqrt(sum([x * x for x in outMeanPos]))
        
        print(inMeanPos, outMeanPos, inDistance, outDistance)        

        if outDistance == 0.0:
            #if we need to handle this case, we probably have bigger problems...
            #anyway, return 1.
            return 1
        
        return inDistance / outDistance
    
    def areAllMeshFacesConnected(self, mesh):
        
        def getFaceNeighbors(faces, face):
            neighbors = []
            for v in face.vertices:
                for otherFace in faces:
                    if otherFace == face:
                        continue
                    if len(set(face.vertices) & set(otherFace.vertices)) == 2:
                        #these faces share an edge
                        if otherFace not in neighbors:
                            neighbors.append(otherFace)
            return neighbors
        
        def visitNeighbors(faces, face, visitedFaces):
            ns = getFaceNeighbors(faces, face)
            for n in ns:
                if n not in visitedFaces:
                    visitedFaces.append(n)
                    visitNeighbors(faces, n, visitedFaces)
        
        
        faces = getMeshFaces(mesh)
        if len(faces) == 1:
            return True
        visitedFaces = []
        visitNeighbors(faces, faces[0], visitedFaces)

        return len(visitedFaces) == len(faces)
        
    def areAllMeshFacesQuads(self, mesh):
        for f in getMeshFaces(mesh):
            if len(f.vertices) != 4:
                return False
        return True
    
    def execute(self, context):
        
        scn = bpy.context.scene
        
        '''
        get the active camera
        '''
        self.camera = bpy.context.scene.camera
        if self.camera == None:
            self.report({'ERROR'}, "There is no active camera")
                    
        '''
        get the mesh containing the quads, assume it's the active object
        '''
        self.mesh = bpy.context.active_object

        if self.mesh == None:
            self.report({'ERROR'}, "There is no active object")
            return{'CANCELLED'}
        if 'Mesh' not in str(type(self.mesh.data)):
            self.report({'ERROR'}, "The active object is not a mesh")
            return{'CANCELLED'}
        if len(getMeshFaces(self.mesh)) == 0:
            self.report({'ERROR'}, "The mesh does not have any faces.")
            return{'CANCELLED'}    
        if not self.areAllMeshFacesQuads(self.mesh):
            self.report({'ERROR'}, "The mesh must consist of quad faces only.")
            return{'CANCELLED'}    
        if not self.areAllMeshFacesConnected(self.mesh):
            self.report({'ERROR'}, "All faces of the input mesh must be connected.")
            return{'CANCELLED'}
        
        '''
        process all quads from the mesh individually, computing vertex depth
        values for each of them.
        '''
        #a dictionary associating computed quads with faces in the original mesh.
        #used later when building the final output mesh
        computedCoordsByFace = {}
        quads = []
        #loop over all faces
        for f in getMeshFaces(self.mesh):
            #if this is a quad face, process it
            if len(f.vertices) == 4:
                assert(f not in computedCoordsByFace.keys())
                
                #gather quad vertices in the local coordinate frame of the
                #mesh
                inputPointsLocalMeshSpace = []
                for idx in f.vertices:
                    inputPointsLocalMeshSpace.append(self.mesh.data.vertices[idx])
                
                #transform vertices to camera space
                inputPointsCameraSpace = self.worldToCameraSpace(inputPointsLocalMeshSpace)
                
                #compute normalized input vectors (eq 16)
                qHats = [normalize(x) for x in inputPointsCameraSpace]

                #run the algorithm to create a quad with depth. coords in camera space
                outputPointsCameraSpace = self.computeQuadDepthInformation(*qHats)
                
                #store the index in the original mesh of the computed quad verts.
                #used later when constructing the output mesh.
                #print("quad")
                for i in range(4):
                   outputPointsCameraSpace[i] = list(outputPointsCameraSpace[i][:])
                   outputPointsCameraSpace[i].append(f.vertices[i])
                
                #remember which original mesh face corresponds to the computed quad       
                computedCoordsByFace[f] = outputPointsCameraSpace
                quads.append(outputPointsCameraSpace)
            else:
                assert(False) #no non-quads allowed. should have been caught earlier
              
        m = self.createMesh(self.mesh, computedCoordsByFace, quads)
        
        #up intil now, coords have been in camera space. transform the final mesh so
        #its transform (and thus origin) conicides with the camera.
        m.matrix_world = bpy.context.scene.camera.matrix_world
        
        #finally apply a uniform scale that matches the distance between
        #the camera and mean point of the two meshes
        uniformScale = self.getOutputMeshScale(self.camera, self.mesh, m)
        m.scale = [uniformScale] * 3
        
        return{'FINISHED'}

class CameraCalibrationPanel(bpy.types.Panel):
    '''
    The GUI for the focal length and camera orientation functionality.
    '''
    bl_label = "Static Camera Calibration"    
    bl_space_type = "CLIP_EDITOR"    
    bl_region_type = "TOOLS"    
    
    def draw(self, context): 
        
        l = self.layout
        scn = context.scene
        l.prop(scn, 'calibration_type')
        row = l.row()        
        box = row.box()
        box.label("Line set 1 (first grease pencil layer)")
        box.prop(scn, 'vp1_axis')
        
        row = l.row()        
        box = row.box()
        singleVp = scn.calibration_type == 'one_vp'
        if singleVp:
            box.label("Horizon line (second grease pencil layer)")
            box.prop(scn, 'use_horizon_segment')
            #box.label("An optional single line segment parallel to the horizon.")
        else:
            box.label("Line set 2 (second grease pencil layer)")
            box.prop(scn, 'vp2_axis')
        row = l.row()        
        #row.enabled = singleVp
        if singleVp:
            row.prop(scn, 'up_axis')
        else:
            row.prop(scn, 'optical_center_type')
        #TODO l.prop(scn, 'vp1_only')

        l.prop(scn, 'set_cambg')
        l.operator("object.estimate_focal_length")  

class CameraCalibrationOperator(bpy.types.Operator):
    '''\brief This operator handles estimation of focal length and camera orientation
    from input line segments. All sections numbers, equations numbers etc
    refer to "Using Vanishing Points for Camera Calibration and Coarse 3D Reconstruction 
    from a Single Image" by E. Guillou, D. Meneveaux, E. Maisel, K. Bouatouch.
    (http://www.irisa.fr/prive/kadi/Reconstruction/paper.ps.gz).
    '''
    
    bl_idname = "object.estimate_focal_length"    
    bl_label = "Calibrate active camera"
    bl_description = "Computes the focal length and orientation of the active camera based on the provided grease pencil strokes."

    def computeSecondVanishingPoint(self, Fu, f, P, horizonDir):
        '''Computes the coordinates of the second vanishing point
        based on the first, a focal length, the center of projection and 
        the desired horizon tilt angle. The equations here are derived from 
        section 3.2 "Determining the focal length from a single image".
        param Fu the first vanishing point in normalized image coordinates.
        param f the relative focal length.
        param P the center of projection in normalized image coordinates.
        param horizonDir The desired horizon direction.
        return The coordinates of the second vanishing point.
        '''
        
        #find the second vanishing point
        #TODO 1: take principal point into account here
        #TODO 2: if the first vanishing point coincides with the image center,
        #        these lines won't work, but this case should be handled somehow.
        k = -(Fu[0] ** 2 + Fu[1] ** 2 + f ** 2) / (Fu[0] * horizonDir[0] + Fu[1] * horizonDir[1])
        Fv = [Fu[i] + k * horizonDir[i] for i in range(2)]
        
        return Fv

    def computeFocalLength(self, Fu, Fv, P):
        '''Computes the focal length based on two vanishing points and a center of projection.
        See 3.2 "Determining the focal length from a single image"
        \param Fu the first vanishing point in normalized image coordinates.
        \param Fv the second vanishing point in normalized image coordinates.
        \param P the center of projection in normalized image coordinates.
        \return The relative focal length.
        '''
        
        #compute Puv, the orthogonal projection of P onto FuFv
        dirFuFv = normalize([x - y for x, y in zip(Fu, Fv)])
        FvP = [x - y for x, y in zip(P, Fv)]
        proj = dot(dirFuFv, FvP)
        Puv = [proj * x + y for x, y in zip(dirFuFv, Fv)]
        
        PPuv = length([x - y for x, y in zip(P, Puv)])
        
        FvPuv = length([x - y for x, y in zip(Fv, Puv)])
        FuPuv = length([x - y for x, y in zip(Fu, Puv)])
        FuFv = length([x - y for x, y in zip(Fu, Fv)])
        #print("FuFv", FuFv, "FvPuv + FuPuv", FvPuv + FuPuv)
        
        fSq = FvPuv * FuPuv - PPuv * PPuv
        #print("FuPuv", FuPuv, "FvPuv", FvPuv, "PPuv", PPuv, "OPuv", FvPuv * FuPuv)
        #print("fSq = ", fSq, " = ", FvPuv * FuPuv, " - ", PPuv * PPuv)
        if fSq < 0:
            return None    
        f = math.sqrt(fSq)
        #print("dot 1:", dot(normalize(Fu + [f]), normalize(Fv + [f])))
        
        return f
    
    def computeCameraRotationMatrix(self, Fu, Fv, f, P):
        '''Computes the camera rotation matrix based on two vanishing points 
        and a focal length as in section 3.3 "Computing the rotation matrix".
        \param Fu the first vanishing point in normalized image coordinates.
        \param Fv the second vanishing point in normalized image coordinates.
        \param f the relative focal length.
        \return The matrix Moc
        '''
        Fu[0] -= P[0]
        Fu[1] -= P[1]
        
        Fv[0] -= P[0]
        Fv[1] -= P[1]
        
        OFu = [Fu[0], Fu[1], f]
        OFv = [Fv[0], Fv[1], f]
        
        #print("matrix dot", dot(OFu, OFv))
        
        s1 = length(OFu)
        upRc = normalize(OFu)
        
        s2 = length(OFv)
        vpRc = normalize(OFv)
        
        wpRc = [upRc[1] * vpRc[2] - upRc[2] * vpRc[1], upRc[2] * vpRc[0] - upRc[0] * vpRc[2], upRc[0] * vpRc[1] - upRc[1] * vpRc[0]] 
        
        M = mathutils.Matrix()
        M[0][0] = Fu[0] / s1
        M[0][1] = Fv[0] / s2
        M[0][2] = wpRc[0]
        
        M[1][0] = Fu[1] / s1
        M[1][1] = Fv[1] / s2
        M[1][2] = wpRc[1]
        
        M[2][0] = f / s1
        M[2][1] = f / s2
        M[2][2] = wpRc[2]
        
        if arePythonMatricesRowMajor():
            M.transpose()
       
        return M
    
    def alignCoordinateAxes(self, M, ax1, ax2):
        '''
        Modifies the original camera transform to make the coordinate axes line 
        up as specified.
        \param M the original camera rotation matrix
        \ax1 The index of the axis to align with the first layer segments.
        \ax2 The index of the axis to align with the second layer segments.
        \return The final camera rotation matrix.
        '''
        #line up the axes as specified in the ui
        x180Rot = mathutils.Euler((math.radians(180.0), 0, 0), 'XYZ').to_matrix().to_4x4()
        z180Rot = mathutils.Euler((0, 0, math.radians(180.0)), 'XYZ').to_matrix().to_4x4()
        z90Rot = mathutils.Euler((0, 0, math.radians(90.0)), 'XYZ').to_matrix().to_4x4()
        zn90Rot = mathutils.Euler((0, 0, math.radians(-90.0)), 'XYZ').to_matrix().to_4x4()
        yn90Rot = mathutils.Euler((0, math.radians(-90.0), 0), 'XYZ').to_matrix().to_4x4()
        xn90Rot = mathutils.Euler((math.radians(-90.0), 0, 0), 'XYZ').to_matrix().to_4x4()
        
        M =  x180Rot * M * z180Rot
        
        if ax1 == 0 and ax2 == 1:
            #print("x,y")
            pass
        elif ax1 == 1 and ax2 == 0:
            #print("y, x")
            M = z90Rot * M
        elif ax1 == 0 and ax2 == 2:
            #print("x, z")
            M = xn90Rot * M
        elif ax1 == 2 and ax2 == 0:
            #print("z, x")
            M = xn90Rot * zn90Rot * M
        elif ax1 == 1 and ax2 == 2:
            #print("y, z")
            M = yn90Rot * z90Rot * M
        elif ax1 == 2 and ax2 == 1:
            #print("z, y")
            M = yn90Rot * M
        
        return M
    
    def gatherGreasePencilSegments(self):
        '''Collects and returns line segments in normalized image coordinates
        from the first two grease pencil layers.
        \return A list of line segment sets. [i][j][k][l] is coordinate l of point k 
        in segment j from layer i. 
        '''
        
        gp = bpy.context.area.spaces.active.clip.grease_pencil
        gpl = gp.layers

        #loop over grease pencil layers and gather line segments
        vpLineSets = []
        for i in range(len(gpl)):
            l = gpl[i]
            if not l.active_frame:
                #ignore empty layers
                continue
            strokes = l.active_frame.strokes
            lines = []
            for s in strokes:
                if len(s.points) == 2:
                    #this is a line segment. add it.
                    line = [p.co[0:2] for p in s.points]
                    lines.append(line)
                    
            vpLineSets.append(lines)
        return vpLineSets    
    
    def computeIntersectionPointForLineSegments(self, lineSet):
        '''Computes the intersection point in a least squares sense of 
        a collection of line segments.
        '''
        matrixRows = []
        rhsRows = []
        
        for line in lineSet:
            #a point on the line
            p = line[0]        
            #a unit vector parallel to the line
            dir = normalize([x - y for x, y in zip(line[1], line[0])])
            #a unit vector perpendicular to the line
            n = [dir[1], -dir[0]]
            matrixRows.append([n[0], n[1]])
            rhsRows.append([p[0] * n[0] + p[1] * n[1]])
            
        m = Mat(matrixRows)
        b = Mat(rhsRows)
        vp = [f[0] for f in m.solve(b)]
        return vp
    
    def computeTriangleOrthocenter(self, verts):
        #print("verts", verts)
        assert(len(verts) == 3)
        
        A = verts[0]
        B = verts[1]
        C = verts[2]
        
        #print("A, B, C", A, B, C)
        
        a = A[0]
        b = A[1]
        c = B[0]
        d = B[1]
        e = C[0]
        f = C[1]
        
        N = b * c+ d * e + f * a - c * f - b * e - a * d
        x = ((d-f) * b * b + (f-b) * d * d + (b-d) * f * f + a * b * (c-e) + c * d * (e-a) + e * f * (a-c) ) / N
        y = ((e-c) * a * a + (a-e) * c * c + (c-a) * e * e + a * b * (f-d) + c * d * (b-f) + e * f * (d-b) ) / N
       
        return (x, y)
        
    
    def relImgCoords2ImgPlaneCoords(self, pt, imageWidth, imageHeight):
        ratio = imageWidth / float(imageHeight)
        sw = ratio
        sh = 1
        return [sw * (pt[0] - 0.5), sh * (pt[1] - 0.5)]
        
    def execute(self, context):
        '''Executes the operator.
        \param context The context in which the operator was executed.
        '''
        scn = bpy.context.scene
        singleVp = scn.calibration_type == 'one_vp'
        useHorizonSegment = scn.use_horizon_segment
        setBgImg = scn.set_cambg
        
        '''
        get the active camera
        '''
        cam = scn.camera     
        if not cam:
            self.report({'ERROR'}, "No active camera.")
            return{'CANCELLED'}
        
        '''
        check settings
        '''
        if singleVp:
            upAxisIndex = ['x', 'y', 'z'].index(scn.up_axis)
            vp1AxisIndex = ['x', 'y', 'z'].index(scn.vp1_axis)
            
            if upAxisIndex == vp1AxisIndex:
                self.report({'ERROR'}, "The up axis cannot be parallel to the axis of the line set.")
                return{'CANCELLED'}    
            vp2AxisIndex = (set([0, 1, 2]) ^ set([upAxisIndex, vp1AxisIndex])).pop()
            vpAxisIndices = [vp1AxisIndex, vp2AxisIndex]
        else:
            vp1AxisIndex = ['x', 'y', 'z'].index(scn.vp1_axis)
            vp2AxisIndex = ['x', 'y', 'z'].index(scn.vp2_axis)
            vpAxisIndices = [vp1AxisIndex, vp2AxisIndex]
            setBgImg = scn.set_cambg
            
            if vpAxisIndices[0] == vpAxisIndices[1]:
                self.report({'ERROR'}, "The two line sets cannot be parallel to the same axis.")
                return{'CANCELLED'}
        
        '''
        gather lines for each vanishing point
        '''        
        activeSpace = bpy.context.area.spaces.active
        
        if not activeSpace.clip:
            self.report({'ERROR'}, "There is no active movie clip.")
            return{'CANCELLED'}
        
        #check that we have the number of layers we need
        if not activeSpace.clip.grease_pencil:
            self.report({'ERROR'}, "There is no grease pencil datablock.")
            return{'CANCELLED'}
        gpl = activeSpace.clip.grease_pencil.layers
        if len(gpl) == 0:
            self.report({'ERROR'}, "There are no grease pencil layers.")
            return{'CANCELLED'}
        if len(gpl) < 2 and not singleVp:
            self.report({'ERROR'}, "Calibration using two vanishing points requires two grease pencil layers.")
            return{'CANCELLED'}
        if len(gpl) < 2 and singleVp and useHorizonSegment:
            self.report({'ERROR'}, "Single vanishing point calibration with a custom horizon line requires two grease pencil layers")
            return{'CANCELLED'}
       
        vpLineSets = self.gatherGreasePencilSegments()
        
        #check that we have the expected number of line segment strokes
        if len(vpLineSets[0]) < 2:
            self.report({'ERROR'}, "The first grease pencil layer must contain at least two line segment strokes.")
            return{'CANCELLED'}
        if not singleVp and len(vpLineSets[1]) < 2:
            self.report({'ERROR'}, "The second grease pencil layer must contain at least two line segment strokes.")
            return{'CANCELLED'}
        if singleVp and useHorizonSegment and len(vpLineSets[1]) != 1:
            self.report({'ERROR'}, "The second grease pencil layer must contain exactly one line segment stroke (the horizon line).")
            return{'CANCELLED'}    
         
        '''
        get the principal point P in image plane coordinates
        TODO: get the value from the camera data panel, 
        currently always using the image center
        '''
        imageWidth = activeSpace.clip.size[0]
        imageHeight = activeSpace.clip.size[1]
        
        #principal point in image plane coordinates. 
        #in the middle of the image by default
        P = [0, 0]
        
        if singleVp:
            '''
            calibration using a single vanishing point
            '''
            imgAspect = imageWidth / float(imageHeight)
            
            #compute the horizon direction
            horizDir = normalize([1.0, 0.0]) #flat horizon by default
            if useHorizonSegment:
                xHorizDir = imgAspect * (vpLineSets[1][0][1][0] - vpLineSets[1][0][0][0])
                yHorizDir = vpLineSets[1][0][1][1] - vpLineSets[1][0][0][1]
                horizDir = normalize([-xHorizDir, -yHorizDir])
            #print("horizDir", horizDir)
            
            #compute the vanishing point location
            vp1 = self.computeIntersectionPointForLineSegments(vpLineSets[0])
            
            #get the current relative focal length
            fAbs = activeSpace.clip.tracking.camera.focal_length
            sensorWidth = activeSpace.clip.tracking.camera.sensor_width
            
            f = fAbs / sensorWidth * imgAspect
            #print("fAbs", fAbs, "f rel", f)
            Fu = self.relImgCoords2ImgPlaneCoords(vp1, imageWidth, imageHeight)
            Fv = self.computeSecondVanishingPoint(Fu, f, P, horizDir)
        else:
            '''
            calibration using two vanishing points
            '''
            if scn.optical_center_type == 'camdata':
                #get the principal point location from camera data
                P = [x for x in  activeSpace.clip.tracking.camera.principal]
                #print("camera data optical center", P[:])
                P[0] /= imageWidth
                P[1] /= imageHeight
                #print("normlz. optical center", P[:])
                P = self.relImgCoords2ImgPlaneCoords(P, imageWidth, imageHeight)
            elif scn.optical_center_type == 'compute':
                if len(vpLineSets) < 3:
                    self.report({'ERROR'}, "A third grease pencil layer is needed to compute the optical center.")
                    return{'CANCELLED'}
                #compute the principal point using a vanishing point from a third gp layer.
                #this computation does not rely on the order of the line sets
                vps = [self.computeIntersectionPointForLineSegments(vpLineSets[i]) for i in range(len(vpLineSets))]
                vps = [self.relImgCoords2ImgPlaneCoords(vps[i], imageWidth, imageHeight) for i in range(len(vps))]
                P = self.computeTriangleOrthocenter(vps)
            else:
                #assume optical center in image midpoint
                pass
            
            #compute the two vanishing points
            vps = [self.computeIntersectionPointForLineSegments(vpLineSets[i]) for i in range(2)]    
        
            #order vanishing points along the image x axis
            if vps[1][0] < vps[0][0]:
                vps.reverse()
                vpLineSets.reverse()
                vpAxisIndices.reverse()            
        
            '''
            compute focal length
            '''
            Fu = self.relImgCoords2ImgPlaneCoords(vps[0], imageWidth, imageHeight)
            Fv = self.relImgCoords2ImgPlaneCoords(vps[1], imageWidth, imageHeight)
            
            f = self.computeFocalLength(Fu, Fv, P)
            
            if f == None:
                self.report({'ERROR'}, "Failed to compute focal length. Invalid vanishing point constellation.")
                return{'CANCELLED'}
        
        '''
        compute camera orientation
        '''
        print(Fu, Fv, f)
        #initial orientation based on the vanishing points and focal length
        M = self.computeCameraRotationMatrix(Fu, Fv, f, P)
        
        #sanity check: M should be a pure rotation matrix, 
        #so its determinant should be 1
        eps = 0.00001
        if 1.0 - M.determinant() < -eps or 1.0 - M.determinant() > eps:
            self.report({'ERROR'}, "Non unit rotation matrix determinant: " + str(M.determinant()))    
            #return{'CANCELLED'} 
        
        #align the camera to the coordinate axes as specified
        M = self.alignCoordinateAxes(M, vpAxisIndices[0], vpAxisIndices[1])
        #apply the transform to the camera
        cam.matrix_world = M
        
        '''
        move the camera an arbitrary distance away from the ground plane
        TODO: focus on the origin or something
        '''
        cam.location = (0, 0, 2)
    
        #compute an absolute focal length in mm based 
        #on the current camera settings
        #TODO: make sure this works for all combinations of
        #image dimensions and camera sensor settings
        if imageWidth >= imageHeight:
            fMm = cam.data.sensor_height * f
        else:
            fMm = cam.data.sensor_width * f
        cam.data.lens = fMm
        self.report({'INFO'}, "Camera focal length set to " + str(fMm))
        
        #move principal point of the blender camera
        r = imageWidth / float(imageHeight)
        cam.data.shift_x = -1 * P[0] / r
        cam.data.shift_y = -1 * P[1] / r
        
        '''
        set the camera background image
        '''
        bpy.context.scene.render.resolution_x = imageWidth
        bpy.context.scene.render.resolution_y = imageHeight
        
        if setBgImg:
            bpy.ops.clip.set_viewport_background()
            
        return{'FINISHED'}    
    
    
        

def initSceneProps():
    '''
    Focal length and orientation estimation stuff
    '''
    desc = "The type of calibration method to use."
    bpy.types.Scene.calibration_type = bpy.props.EnumProperty(items = [('one_vp','one vanishing point','Estimates the camera orientation using a known focal length, a single vanishing point and an optional horizon tilt angle.'),('two_vp','two vanishing points','Estimates the camera focal length and orientation from two vanishing points')],name = "Method", description=desc, default = ('two_vp'))
    
    desc = "The axis to which the line segments from the first layer are parallel."
    bpy.types.Scene.vp1_axis = bpy.props.EnumProperty(items = [('x','x axis','xd'),('y','y axis','yd'),('z','z axis','zd')],name = "Parallel to the", description=desc, default = ('x'))
    desc = "The axis to which the line segments from the second layer are parallel."
    bpy.types.Scene.vp2_axis = bpy.props.EnumProperty(items = [('x','x axis','xd'),('y','y axis','yd'),('z','z axis','zd')],name = "Parallel to the", description=desc, default = ('y'))
    desc = "The up axis for single vanishing point calibration."
    bpy.types.Scene.up_axis = bpy.props.EnumProperty(items = [('x','x axis','xd'),('y','y axis','yd'),('z','z axis','zd')],name = "Up axis", description=desc, default = ('z'))
    
    desc = "How the optical center is computed for calibration using two vanishing points."
    bpy.types.Scene.optical_center_type = bpy.props.EnumProperty(items = [('mid','Image midpoint','Assume the optical center coincides with the image midpoint (reasonable in most cases).'),('camdata','From camera data','Get a known optical center from the current camera data.'),('compute','From 3rd vanishing point','Computes the optical center using a third vanishing point from grease pencil layer 3.')],name = "Optical center", description=desc, default = ('mid'))
    
    bpy.types.Scene.vp1_only = bpy.props.BoolProperty(name="Only use first line set", description=desc, default=False)
    desc = "Automatically set the current movie clip as the camera background image when performing camera calibration (works only when a 3D view-port is visible)."
    bpy.types.Scene.set_cambg = bpy.props.BoolProperty(name="Set background image", description=desc, default=True)
    desc = "Extract the horizon angle from a single line segment in the second grease pencil layer. If unchecked, the horizon angle is set to 0."
    bpy.types.Scene.use_horizon_segment = bpy.props.BoolProperty(name="Compute from grease pencil stroke", description=desc, default=True)
    '''
    3D reconstruction stuff
    '''
    desc = "Do not join the faces in the reconstructed mesh. Useful for finding problematic faces."
    bpy.types.Scene.separate_faces = bpy.props.BoolProperty(name="Separate faces", description=desc, default=False)
    
    desc = "The method to use to project the image onto the mesh."
    bpy.types.Scene.projection_method = bpy.props.EnumProperty(items = [('simple','simple','Uses UV coordinates projected from the camera view. May give warping on large faces.'),('hq','high quality','Uses a UV Project modifier combined with a simple subdivision modifier.')],name = "Method", description=desc, default = ('hq'))

def register():
    initSceneProps()
    bpy.utils.register_module(__name__)
    
 
def unregister():
    bpy.utils.unregister_module(__name__)
 
if __name__ == "__main__":
    register()
