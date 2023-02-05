# A simple implementation of a density based scan, used to classify 
# neighbor cluster endpoints by their neighborhood.
def dbClusterScan(clusterEnds, eps, minP):
    # clusterEnds:  A list of endpoints, where the first element
    #               of an endpoint is its position.
    # eps:          Threshold distance
    # minP:         Minimal number of points in cluster
    # return:       A Python list of lists, everyone containing neighbor
    #               endpoints or a single endpoint, when no neighbors.
 
    labels = [0]*len(clusterEnds) # -1: noise point, 0: not yet processed
    curLabel = 0            # index of current label
    
    # Find seed point
    for ip, p in enumerate(clusterEnds):
        # Skip if already processed (labels[ip] == 0 else)
        if labels[ip]:
           continue
        
        # Find all of <p>'s neighboring points.
        neighbors = findNeighbors(clusterEnds, p, eps)
        
        # Noise, if not enough neighbors, ...
        if len(neighbors) < minP:
            labels[ip] = -1

        # ... else we have a seed point for a new cluster.    
        else: 
           curLabel += 1
           # Create the new cluster
           createCluster(clusterEnds, labels, ip, neighbors, curLabel, eps, minP)
             
    # combine endpoints and labels
    combEndLabel = [(end,label) for end,label in zip(clusterEnds,labels)]
    clusterGroups = [[y[0] for y in combEndLabel if y[1]==x] for x in set(labels) if x != -1]
    # Add single endpoints
    clusterGroups.extend( [ [y[0]] for y in combEndLabel if y[1]==-1])
    return clusterGroups


def createCluster(clusterEnds, labels, ip, neighbors, curLabel, eps, minP):
    labels[ip] = curLabel
    
    # From here, use <neighbors> as FIFO queue
    i = 0
    while i < len(neighbors):          
        # Get the next point from the queue.        
        ipn = neighbors[i]
       
        if labels[ipn] == -1:
           labels[ipn] = curLabel
        
        # Otherwise, if <pn> isn't already claimed, claim it as part of <curLabel>.
        elif labels[ipn] == 0:
            labels[ipn] = curLabel
            
            # Find all the neighbors of <pn>
            pn_neighbors = findNeighbors(clusterEnds, clusterEnds[ipn], eps)
            
            # If <pn> has at least <minP> neighbors, it's a branch point!
            # Add all of its neighbors to the FIFO queue to be searched. 
            if len(pn_neighbors) >= minP:
                neighbors = neighbors + pn_neighbors

            # If <pn> doesn't have enough neighbors, then it's a leaf point, do nothing.        
        i += 1        

def findNeighbors(clusterEnds, p, eps):
    # Find all points in <vects> within distance <eps> of point <p>.
    neighbors = []
    eps2 = eps*eps  # we use the square of the distance
    for ipn,pn in enumerate(clusterEnds):
        dp = pn[0] - p[0]
        if dp.dot(dp) < eps2:
           neighbors.append(ipn)           
    return neighbors