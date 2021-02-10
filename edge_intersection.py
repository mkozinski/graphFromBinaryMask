from enum import Enum
import itertools

class Direction(Enum):
    collinear=0
    anticlockwise=1
    clockwise=2
    
def direction(a,b,c):
    v = (b[1]-a[1])*(c[0]-b[0])-(b[0]-a[0])*(c[1]-b[1])
    if (v == 0):
        return Direction.collinear
    elif (v < 0):
        return Direction.anticlockwise
    return Direction.clockwise

def point_in_interval(i1,i2,p):
    return i1<=p<=i2 or i1>=p>=i2

def point_inside_interval(i1,i2,p):
    #insdie but not at the end
    return i1<p<i2 or i1>p>i2

def point_inside_rect(c1,c2,p):
    return point_inside_interval(c1[0],c2[0],p[0]) and point_inside_interval(c1[1],c2[1],p[1])

def point_in_rect(c1,c2,p):
    return point_in_interval(c1[0],c2[0],p[0]) and point_in_interval(c1[1],c2[1],p[1])

def point_in_section(s1,s2,p):
    return direction(s1,s2,p)==Direction.collinear and point_in_rect(s1,s2,p)

def edges_intersect(p1,p2,q1,q2):
    # end point lying on the section is not considered an intersection
    # coincidence of end points is not considered an intersection
    
    dir1 = direction(p1, p2, q1)
    dir2 = direction(p1, p2, q2)
    dir3 = direction(q1, q2, p1)
    dir4 = direction(q1, q2, p2)

    if (dir1==Direction.clockwise and dir2==Direction.anticlockwise or dir1==Direction.anticlockwise and dir2==Direction.clockwise) and \
       (dir3==Direction.clockwise and dir4==Direction.anticlockwise or dir3==Direction.anticlockwise and dir4==Direction.clockwise) :
            return True
    if dir1 == Direction.collinear and point_inside_rect(p1,p2,q1): #q1 on the p line 
        return True
    if dir2 == Direction.collinear and point_inside_rect(p1,p2,q2): #q2 on the p line 
        return True
    if dir3 == Direction.collinear and point_inside_rect(q1,q2,p1): #p1 on the q line
        return True
    if dir4 == Direction.collinear and point_inside_rect(q1,q2,p2): #p2 on the q line 
        return True
    
    return False

def intersection_point(p1,p2,q1,q2):
    a=((p2[1]-q2[1])*(-q1[0]+q2[0])+(-q1[1]+q2[1])*(-p2[0]+q2[0]))/((p1[1]-p2[1])*(q1[0]-q2[0])+(-q1[1]+q2[1])*(p1[0]-p2[0]))
    return (p1[0]*a+p2[0]*(1-a),p1[1]*a+p2[1]*(1-a))

def move_graph(g,correction):
    for n in g.nodes:
        g.nodes[n]['pos']=(g.nodes[n]['pos'][0]-correction[0],g.nodes[n]['pos'][1]-correction[1])
    return g

def crop_graph(g,crop):
    nodeinds=[n for n in g.nodes]
    maxind=max(nodeinds)
    newind=maxind+1
    # insert new nodes
    edges2remove=[]
    edges2add=[]
    nodes2add=[]
    for e in g.edges:
        p1=g.nodes[e[0]]['pos']
        p2=g.nodes[e[1]]['pos']
        q1=(crop[0][0],crop[1][0])
        q2=(crop[0][0],crop[1][1])
        q3=(crop[0][1],crop[1][1])
        q4=(crop[0][1],crop[1][0])
        for p,q in itertools.product([(p1,p2)],[(q1,q2),(q2,q3),(q3,q4),(q4,q1)]):
            if edges_intersect(p[0],p[1],q[0],q[1]):
                ip=intersection_point(p[0],p[1],q[0],q[1])
                edges2add.append((e[0],newind))
                edges2add.append((e[1],newind))
                nodes2add.append((newind,ip))
                edges2remove.append(e)
                newind+=1
                break
    for n in nodes2add:
        g.add_node(n[0],pos=n[1])
    for e in edges2add:
        g.add_edge(e[0],e[1])
    for e in edges2remove:
        g.remove_edge(e[0],e[1])
            
    nodes2remove=[]
    for n in g.nodes:
        if crop[0][0]<=g.nodes[n]['pos'][0]<=crop[0][1] and crop[1][0]<=g.nodes[n]['pos'][1]<=crop[1][1]:
            pass
        else:
            nodes2remove.append(n)
    for n in nodes2remove:
        g.remove_node(n)
        
    return g
