import numpy as np
from math import ceil
from mathutils import Vector
from lib.CompGeom.piecewise import piecewise
from lib.CompGeom.centerline import centerlineOf
from lib.CompGeom.PolyLine import PolyLine

from osmPlot import *

def clipParallelPartold(lines):
    line1 = lines[0]    # left line
    line2 = lines[-1]   # right line
    reference = PolyLine(centerlineOf(line1[:],line2[:]))
    length1 = line1.length()
    length2 = line2.length()
    notSwapped = length1>length2
    longer, shorter = (line1.clone(),line2.clone()) if notSwapped else (line2.clone(),line1.clone())

    # Eventually, the problem comes from different line lengths.
    if abs(length1-length2) > 10.:
        _,t =longer.orthoProj(shorter[-1])
        longer = longer.trimmed(0.,t)
        longer[-1].freeze()
        if notSwapped:
            lines[0] = longer
        else:
            lines[-1] = longer
        result = (
            lines,
            'clipL' if notSwapped else 'clipR'
        )
        return result
    else:
        # Subsample reference line in pieces of approximately 5 m.
        length = reference.length()
        sampleCount = ceil(length / 5.)
        x = np.linspace(0.,length,sampleCount)
        # Compute distance of these samples to shorter line.
        ys = [shorter.distTo(reference.d2v(xi))[1] for xi in x]
        yl = [longer.distTo(reference.d2v(xi))[1] for xi in x]
        y = [abs(d1)+abs(d2) for d1,d2 in zip(ys,yl)]

        # Now, a piecewise linear regression should deliver a
        # separate segment at the end.
        n = [i for i in range(sampleCount)]
        model = piecewise(n,y)

        # import matplotlib.pyplot as plt
        # plt.plot(n,y)
        # for segment in model.segments:
        #     xx = np.linspace(segment.start_t,segment.end_t,5)
        #     yy = [segment.coeffs[1]*x + segment.coeffs[0] for x in xx]
        #     plt.plot(xx,yy,'r')
        # plt.show()

        if len(model.segments)>1:
            breakIndx = int(model.segments[-1].start_t)-1
            # Trim the lines at the distance x given by this index
            t = reference.d2t(x[breakIndx])
            reference = reference.trimmed(0.,t)
            p0,p1 = reference[-1],reference[-2]
            v = p1 - p0
            perp = Vector((-v[1],v[0]))
            p2 = p0 + perp

            # for line in lines:
            #     line.plot('k')
            #     p = line[-1]
            #     plt.plot(p[0],p[1],'gx')
            #     q,t = line.intersectWithLine(p0, p2)
            #     plt.plot(q[0],q[1],'cx')
            #     new = line.trimmed(0.,t)
            #     new.plot('k',3)
            # reference.plot('g')
            # plt.plot(p0[0],p0[1],'ro')
            # plt.plot([p0[0],p2[0]],[p0[1],p2[1]],'r')
            # plotEnd()

            for i,line in enumerate(lines):
                _,t = lines[i].intersectWithLine(p0, p2)
                lines[i] = line.trimmed(0.,t)
                lines[i][-1].freeze()

            # for line in lines:
            #     line.plot('k')
            # reference.plot('g')
            # plotEnd()
       # else only one segment, don't trim
        result = (
            lines,
            'clipB'
        )
        return result

def clipParallelPart(lines):
    line1 = lines[0]    # left line
    line2 = lines[-1]   # right line
    length1 = line1.length()
    length2 = line2.length()
    notSwapped = length1>length2
    longer, shorter = (line1.clone(),line2.clone()) if notSwapped else (line2.clone(),line1.clone())

    # Eventually, the problem comes from different line lengths.
    if abs(length1-length2) > 10.:
        _,t =longer.orthoProj(shorter[-1])
        longer = longer.trimmed(0.,t)
        longer[-1].freeze()
        if notSwapped:
            lines[0] = longer
        else:
            lines[-1] = longer
        result = (
            lines,
            'clipL' if notSwapped else 'clipR'
        )
        return result
    else:
        # Subsample longer line in pieces of approximately 5 m.
        length = longer.length()
        sampleCount = ceil(length / 5.)
        x = np.linspace(0.,length,sampleCount)
        # Compute distance of these samples to shorter line.
        y = [shorter.distTo(longer.d2v(xi))[1] for xi in x]

        # Now, a piecewise linear regression should deliver a
        # separate segment at the end.
        n = [i for i in range(sampleCount)]
        model = piecewise(n,y)

        # import matplotlib.pyplot as plt
        # plt.plot(n,y)
        # for segment in model.segments:
        #     xx = np.linspace(segment.start_t,segment.end_t,sampleCount)
        #     yy = [segment.coeffs[1]*x + segment.coeffs[0] for x in xx]
        #     plt.plot(xx,yy,'r')
        # plt.show()

        if len(model.segments)>1:
            breakIndx = int(model.segments[-1].start_t)-5
            # Trim the lines at the distance x given by this index
            for i,line in enumerate(lines):
                lines[i] = line.trimmed(0.,line.d2t(x[breakIndx]))
                lines[i][-1].freeze()
       # else only one segment, don't trim
        result = (
            lines,
            'clipB'
        )
        return result

