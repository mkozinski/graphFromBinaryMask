import networkx as nx
import torch
import torch.nn as nn
from rdp import rdp
import argparse
import numpy as np
import itertools
from edge_intersection import edges_intersect, intersection_point
from skimage.morphology import skeletonize

def crop_graph(g,crop):
    # insert new nodes
    edges2remove=[]
    edges2add=[]
    nodes2add=[]
    for e in g.edges:
        p1=e[0]
        p2=e[1]
        q1=(crop[0][0],crop[1][0])
        q2=(crop[0][0],crop[1][1])
        q3=(crop[0][1],crop[1][1])
        q4=(crop[0][1],crop[1][0])
        for p,q in itertools.product([(p1,p2)],[(q1,q2),(q2,q3),(q3,q4),(q4,q1)]):
            if edges_intersect(p[0],p[1],q[0],q[1]):
                ip=intersection_point(p[0],p[1],q[0],q[1])
                edges2add.append((e[0],(ip[0],ip[1])))
                edges2add.append((e[1],(ip[0],ip[1])))
                nodes2add.append((ip[0],ip[1]))
                edges2remove.append(e)
                break
    for n in nodes2add:
        g.add_node(n)
    for e in edges2add:
        g.add_edge(e[0],e[1])
    for e in edges2remove:
        g.remove_edge(e[0],e[1])
            
    nodes2remove=[]
    for n in g.nodes:
        if crop[0][0]<=n[0]<=crop[0][1] and crop[1][0]<=n[1]<=crop[1][1]:
            pass
        else:
            nodes2remove.append(n)
    for n in nodes2remove:
        g.remove_node(n)
        
    return g

def get_full_graph(mask):
    # get nx.Graph from a binary mask np.array
    
    g=nx.Graph()
    
    # nonzero pixels are graph nodes
    indsy,indsx=np.nonzero(mask)
    nodes=[(x,y) for y,x in zip(indsy,indsx)]
    g.add_nodes_from(nodes)
    
    # filters for detecting vertical and horizontal edges
    fltrs_vh=np.array([
        [[0,1,0],
         [0,1,0],
         [0,0,0]],
        [[0,0,0],
         [0,1,0],
         [0,1,0]],
        [[0,0,0],
         [1,1,0],
         [0,0,0]],
        [[0,0,0],
         [0,1,1],
         [0,0,0]],])
    direction_vh=  [(-1, 0),( 1, 0),( 0,-1),( 0, 1)]
    
    # find vertical and horizontal edges
    c=nn.Conv2d(1,4,3,1,1,bias=False)
    c.weight.data=torch.from_numpy(fltrs_vh).reshape_as(c.weight.data).type_as(c.weight.data)
    conn_vh=(c.forward(torch.from_numpy(mask[np.newaxis,np.newaxis,:,:]).type(torch.float32))==2)
    
    # add vertical and horizontal edges to the graph
    for ind in range(4):
        indsy,indsx=np.nonzero(conn_vh[0][ind].numpy())
        for y,x in zip(indsy,indsx):
            cand_neighb=(x+direction_vh[ind][1],y+direction_vh[ind][0],)
            g.add_edge((x,y,),cand_neighb)

    # diagonal connectivity
    fltrs_diag=np.array([
        [[1,0,0],
         [0,1,0],
         [0,0,0]],
        [[0,0,1],
         [0,1,0],
         [0,0,0]],
        [[0,0,0],
         [0,1,0],
         [1,0,0]],
        [[0,0,0],
         [0,1,0],
         [0,0,1]]])
    connections_diag=[(-1,-1),(-1, 1),( 1,-1),( 1, 1)]
    
    # find diagonal edges
    c=nn.Conv2d(1,4,3,1,1,bias=False)
    c.weight.data=torch.from_numpy(fltrs_diag).reshape_as(c.weight.data).type_as(c.weight.data)
    conn_diag=(c.forward(torch.from_numpy(mask[np.newaxis,np.newaxis,:,:]).type(torch.float32))==2)
    
    # add diagonal edges to the graph, but avoid forming three-edge loops
    # involving previously added vertical and horizontal edges
    
    def connected(g,n1,n2): # nodes are connected over two-edge path
        neighbs2=[n for m in g.neighbors(n1) for n in g.neighbors(m)]
        return n2 in neighbs2

    for ind in range(4):
        indsy,indsx=np.nonzero(conn_diag[0][ind].numpy())
        for y,x in zip(indsy,indsx):
            cand_neighb=(x+connections_diag[ind][1],y+connections_diag[ind][0],)
            if not connected(g,(x,y,),cand_neighb):
                g.add_edge((x,y,),cand_neighb)
                
    return g

def remove_small_ccs(g,numnodes):
    ccs2remove=[cc for cc in nx.connected_components(g) if len(cc)<numnodes ]
    for ccs in ccs2remove:
        g.remove_nodes_from(ccs)

def remove_short_dead_ends(g,length):
    # g is a nx.Graph
    # 
    
    junctions=[n for n in g.nodes if len(list(g.neighbors(n)))>2]
    
    # to identify the chains to be removed, we remove all the junctions from the graph
    # the connected components of the resulting graph form chains 
    # each cc is guaranteed to only contain nodes of cardinality at most 2
    h=g.copy()
    h.remove_nodes_from(junctions)
    nodes2remove=[]
    for cc in nx.connected_components(h):
        ccg=h.subgraph(cc).copy()
        assert len(ccg)>0
        # interf contains nodes from outside of cc that connect to cc
        interf=[n for n in junctions if len(set(g[n])&set(ccg.nodes))>0 ]
        assert 0<=len(interf)<=2 # a chain can have at most 2 end points
        
        # case 0: disconnected comp
        # case 1: either junction-node-...-node, or junction-node or loop on a stem
        if len(interf)<=1:
            if len(ccg)<length:
                nodes2remove+=list(ccg.nodes) 
                
    g.remove_nodes_from(nodes2remove)

    return len(nodes2remove)

def simplify_chain(chaing,endpoints,epsilon):
    ordered_chain=nx.shortest_path(chaing,endpoints[0],endpoints[1])
    simplified_chain=rdp(ordered_chain,epsilon=epsilon)
    g=nx.Graph()
    cciter=iter(simplified_chain)
    prev=next(cciter)
    for node in cciter:
        g.add_edge((prev[0],prev[1]),(node[0],node[1]))
        prev=node
    return g

def simplify_loop(loopg,epsilon):
    for n in loopg.nodes:
        assert len(loopg[n])==2
    m=next(iter(loopg.nodes))
    n=next(iter(loopg[m]))
    lg=loopg.copy()
    lg.remove_edge(m,n)
    simplified_loop=simplify_chain(lg,[m,n],epsilon)
    simplified_loop.add_edge(m,n)
    return simplified_loop

def simplify_graph(g,epsilon=1):
    # epsilon is a parameter of the RDP polyline simplification algorithm
    # g is a nx.Graph
    # 
    # simplify chains connecting junctions/end-points
    # do not move junctions/end points
    # do not change topology
    # our assumption is that loops in g, if they exist, have at least 4 edges
    # this is guaranteed when g is output by get_full_graph
    
    junctions=[n for n in g.nodes if len(list(g.neighbors(n)))>2]
    
    simplified_g=nx.Graph()
    
    # edges between two junction nodes cannot be simplified
    junction_only_edges=[e for e in g.edges() if e[0] in junctions and e[1] in junctions]
    simplified_g.add_edges_from(junction_only_edges)
    
    # to identify the chains to be simplified, we remove all the junctions from the graph
    # the connected components of the resulting graph form chains that can be simplified 
    # each cc is guaranteed to only contain nodes of cardinality at most 2
    h=g.copy()
    h.remove_nodes_from(junctions)
    for cc in nx.connected_components(h):
        ccg=h.subgraph(cc).copy()
        assert len(ccg)>0
        # interf contains nodes from outside of cc that connect to cc
        interf=[n for n in junctions if len(set(g[n])&set(ccg.nodes))>0 ]
        assert 0<=len(interf)<=2 # a chain can have at most 2 end points
        
        # case 0: disconnected comp
        if len(interf)==0:
            endps=[n for n in ccg.nodes if len(ccg[n])<2]
            assert 0<=len(endps)<=2
            # single node
            if len(endps)==1:
                assert len(ccg.nodes)==1
                simplified_g.add_node(next(iter(ccg.nodes)))
            # a chain
            elif len(endps)==2:
                chain=simplify_chain(ccg,endps,epsilon)
                simplified_g.add_edges_from(chain.edges)
            # a loop
            elif len(endps)==0:
                for n in ccg.nodes():
                  assert len(ccg[n])==2
                simplified_loop=simplify_loop(ccg,epsilon)
                simplified_g.add_edges_from(simplified_loop.edges)
                
        # case 1: either junction-node-...-node, or junction-node or loop on a stem
        elif len(interf)==1:
            n=interf[0]
            interf_inner=[m for m in ccg.nodes if m in g[n]]
            assert 0<len(interf_inner)<=2
            # chain
            if len(interf_inner)==1:
                m=interf_inner[0] # one node attaches to the outside of the cc
                eps=[m for m in ccg.nodes if len(g[m])==1] 
                assert len(eps)==1 
                ep=eps[0] # another node is the end point of the graph
                ccg.add_edge(n,m)
                simplified_chain=simplify_chain(ccg,[n,ep],epsilon)
                simplified_g.add_edges_from(simplified_chain.edges)
            # loop
            elif len(interf_inner)==2:
                ccg.add_edge(n,interf_inner[0])
                ccg.add_edge(n,interf_inner[1])
                simplified_loop=simplify_loop(ccg,epsilon)
                simplified_g.add_edges_from(simplified_loop.edges)
                    
        # case 2: junction-node-node-....-junction
        elif len(interf)==2:
            n1,n2=interf
            n1n=[n for n in g.neighbors(n1) if n in ccg]
            n2n=[n for n in g.neighbors(n2) if n in ccg]
            assert len(n1n)==1
            assert len(n2n)==1
            m1=n1n[0]
            m2=n2n[0]
            assert m1!=m2 or len(ccg)==1
            ccg.add_edge(n1,m1)
            ccg.add_edge(n2,m2)
            simplified_chain=simplify_chain(ccg,[n1,n2],epsilon)
            simplified_g.add_edges_from(simplified_chain.edges)
        else:
            raise ValueError()
            
                
    return simplified_g


def main(ifname,ofname,threshold,crop,small_disc_thr,deadend_thr,epsilon):
        pre=np.load(ifname)
        pre_skel=skeletonize(pre>threshold)
        fullgraph=get_full_graph((pre_skel).astype(np.float32))
        if crop:
            croppedgraph=crop_graph(fullgraph,crop) #((19,1281),(19,1281)))
        
        # two heuristic filters
        # they only work on dense pixel-wise graphs, before simplification
        while remove_short_dead_ends(croppedgraph,deadend_thr):
            pass
        remove_small_ccs(croppedgraph,small_disc_thr)
        
        simplifiedgraph=simplify_graph(croppedgraph,epsilon=epsilon)
        
        h=convertGraph(simplifiedgraph)

        nx.write_gpickle(h,ofname)

def convertGraph(g):
    # convert graph to the `standard' format
    h=nx.Graph()
    node_map={}
    nodeind=1
    for n in g.nodes():
        h.add_node(nodeind,pos=n)
        node_map[n]=nodeind
        nodeind+=1
    for e in g.edges():
        h.add_edge(node_map[e[0]],node_map[e[1]])
    
    return h

if __name__ == "__main__":
    threshold_help="the threshold used to binarize the input probability map"
    crop_help="top bottom left right in image pixel coordinates"
    small_disconnected_threshold_help="disconnected subgraphs with less than this number of pixels are removed"
    deadend_threshold_help="dead-ending roads with less than this number of pixels are removed"
    epsilon_rdp_help="a parameter of the RDP algorithm, typically set to 3? 5? pixels"
    parser = argparse.ArgumentParser()   
    parser.add_argument("input_file",  type=str)
    parser.add_argument("output_file", type=str)
    parser.add_argument("--threshold", "-t", type=float, default=0.5, help=threshold_help)
    parser.add_argument("--crop", "-c", type=int, nargs=4, required=True, default=None, help=crop_help)
    parser.add_argument("--small_disconnected_threshold", "-s", type=int, default=0, help=small_disconnected_threshold_help)
    parser.add_argument("--deadend_threshold", "-d", type=int, default=0, help=deadend_threshold_help)
    parser.add_argument("--epsilon_rdp", "-e", type=int, default=0, help=epsilon_rdp_help)

    args = parser.parse_args()

    if args.crop:
        crop=((args.crop[0],args.crop[1]),(args.crop[2],args.crop[3]))
    else:
        crop=None
    main(args.input_file,args.output_file,args.threshold,crop,
         args.small_disconnected_threshold,args.deadend_threshold,args.epsilon_rdp)

   #python graphFromBinaryMask.py "log_ce_v2/output_epoch_220/SN3_roads_train_AOI_2_Vegas_PS-RGB_img494.png.npy" g.graph -t 0.5 -c 19 1281 19 1281 -s 100 -d 20 -e 3
