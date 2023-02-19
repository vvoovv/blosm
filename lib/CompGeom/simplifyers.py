def sqrDistance(a,b):
    d = a-b
    return d.dot(d)

# Distance from point p to segment p1-p2
def sqrDistanceSeg(p,p1,p2):
    x,y = p1[0],p1[1]
    d = p2-p1
    dx, dy = d[0],d[1]
    t = ((p[0] - x) * dx + (p[1] - y) * dy) / (dx * dx + dy * dy)

    if t > 1:
        x = p2[0]
        y = p2[1]

    elif t > 0:
        x += dx * t;
        y += dy * t;

    dx = p[0] - x
    dy = p[1] - y
    return dx * dx + dy * dy

# Distance based simplification
def simplifyRadialDist(line,tolerance):
    tolerance2 = tolerance*tolerance
    i = 0
    while i < len(line)-1:
        a, b = line[i], line[i+1]
        if sqrDistance(a,b) < tolerance2:
            del line[i]
        else:
            i += 1
    return line

# Simplify short ends of line (important when line is expanded)
def simplifyEnds(line,tolerance):
    tolerance2 = tolerance*tolerance
    if sqrDistance(line[0],line[-1]) < tolerance2:
        return line
    if sqrDistance(line[0],line[1]) < tolerance2:
        del line[1]
    if sqrDistance(line[-2],line[-1]) < tolerance2:
        del line[-2]
    return line

def RDPStep(points, first, last, sqTolerance, simplified):
    maxSqDist = sqTolerance

    for i in range(first+1,last):
        sqDist = sqrDistanceSeg(points[i], points[first], points[last])
        if sqDist > maxSqDist:
            index = i;
            maxSqDist = sqDist

    if maxSqDist > sqTolerance:
        if index - first > 1: RDPStep(points, first, index, sqTolerance, simplified)
        simplified.append(points[index])
        if last - index > 1: RDPStep(points, index, last, sqTolerance, simplified)

# Simplification using Ramer-Douglas-Peucker algorithm
def simplifyRDP(line, tolerance):
    sqTolerance = tolerance * tolerance
    
    last = len(line)-1
    simplified = [line[0]]
    RDPStep(line, 0, last, sqTolerance, simplified)
    simplified.append(line[last])

    return simplified