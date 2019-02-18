# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 15:33:51 2019
this module implements algorithm for maximum clique detection.
@author: le3
"""
#import planarity
import networkx as nx
import numpy as np
from scipy.spatial.distance import squareform
from collections import defaultdict

def sort_graph_edges(G):
    # rank links based on weight (absolute value) 
    sorted_edges = []
    for source, dest, data in sorted(G.edges(data=True),
                                     key=lambda x: abs(x[2]['weight'])):
        sorted_edges.append({'source': source,
                             'dest': dest,
                             'weight': data['weight']})
    return sorted_edges

def compute_PMFG(sorted_edges, nb_nodes):
    # remove weakest links
    '''
    References
    Tumminello, M., Aste, T., Di Matteo, T., & Mantegna, R. N. (2005). A tool for filtering information in complex systems. Proceedings of the National Academy of Sciences of the United States of America, 102(30), 10421-10426.
    https://gmarti.gitlab.io/networks/2018/06/03/pmfg-algorithm.html
    https://github.com/hagberg/planarity
    http://jgaa.info/accepted/2004/BoyerMyrvold2004.8.3.pdf
    https://networkx.github.io/documentation/stable/index.html
    '''
    PMFG = nx.Graph()
    for edge in sorted_edges:
        PMFG.add_edge(edge['source'], edge['dest'])
        if not planarity.is_planar(PMFG):
            PMFG.remove_edge(edge['source'], edge['dest'])
            
        if len(PMFG.edges()) == 3*(nb_nodes-2):
            break
    
    return PMFG
    
def planar_maximally_filter(G):
    #fn = 'corr_nokia.gexf'
    #G = nx.read_gexf(fn)
    sorted_edges = sort_graph_edges(G)
    G2 = compute_PMFG(sorted_edges, len(G.nodes))  
    return G2

def simplify_using_spanningTree(G):
    T = nx.minimum_spanning_tree(G)
    return T
    
def testcase_simulate_graph():
    nb_nodes = 1000
    distances = squareform(np.random.uniform(
        size=int(nb_nodes * (nb_nodes - 1) / 2)))
    distances[np.diag_indices(nb_nodes)] = np.ones(nb_nodes)
    complete_graph = nx.Graph()
    for i in range(nb_nodes):
        for j in range(i+1, nb_nodes):
            complete_graph.add_edge(i, j, weight=distances[i,j])
    
    sorted_edges = sort_graph_edges(complete_graph)
    
    result = compute_PMFG(sorted_edges, len(complete_graph.nodes))  
    return result

def simplification_minimum_lost_connectivity(G,gama):
    # Zhou, Fang, Sebastien Malher, and Hannu Toivonen. 
    # "Network simplification with minimal loss of connectivity." Data Mining (ICDM), 2010 IEEE 10th International Conference on. IEEE, 2010.
    #  The algorithm is a naive approach, simply pruning a fraction of the weakest edges by sorting edges according to the edge weight
    n = gama*(len(G.edges())-len(G.node())+1)
    be_removed = []
    count = 0
    # print 'length', len(G.edges())
    # novel idea: remove edges whose weight close to 0 (less positve and less negative)
    for source, dest, data in sorted(G.edges(data=True),
                                     key=lambda x: abs(x[2]['weight'])):
        G.remove_edge(source,dest)
        # print 'length', len(G.edges())
        if (nx.is_connected(G) and count<n):
            be_removed.append((source, dest))
            count = count+1
        else:
            print 'break for loop'
            break 
    print 'length', len(G.edges())
    return G
    
def network_from_sinduja_data():
    # generate network from published data:
    # https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0198807
    # Ranganathan, Sindhuja, Mikko Kivelä, and Juho Kanniainen. 
    #"Dynamics of investor spanning trees around dot-com bubble." PloS one 13.6 (2018): e0198807
    import glob
    files = glob.glob('./financial_institution/*')
    #files = glob.glob('./non_financial_institution/*')
    #files = glob.glob('./household/*')
    #data = pd.read_csv(files[0], sep=" ", header=None)
    links = []
    with open(files[0], "r") as ins:
        for line in ins:
            a=line.split(',')
            links.append((a[0],a[1],float(a[2])))
    G = nx.Graph()   
    #G2.add_nodes_from(investors)
    G.add_weighted_edges_from(links)
    '''
    cliques=nx.find_cliques(G)
    for clique in cliques:
        print len(clique)
    G1 = testCase_investor_network(G)
    cliques2=nx.find_cliques(G1)
    for clique in cliques2:
        print len(clique)
    '''
    #G1 = planar_maximally_filter(G)
    # G2 = simplification_minimum_lost_connectivity(G,0.8)
    return G 



def degeneracy_ordering(G):
    de= nx.core_number(G)
    nn= [(u,de[u]) for u in de]
    o = [a for a,b in sorted(nn, key=lambda x: x[1], reverse=False)]
    return o
    
def retrieve(G,clique):
    clq = []
    for u in clique:
        nodesAt_u = [x for x,y in G.nodes(data=True) if y['count']==u]
        clq.append(nodesAt_u[0])
    return clq    
 
def is_clique(C,G):
    b = False
    if len(C)<2:
        b = True 
    else:
        H = G.subgraph(C)
        e = len(list(H.edges()))
        v = len(list(H.nodes()))
        if e == v*(v-1)/2:
            b = True
    return b

def fast_heuristic_clique(G):
    nx.set_node_attributes(G, nx.core_number(G), name='core')
    clique = []
    maxSize = 0
    for v, data in sorted(G.nodes(data=True), key=lambda x: x[1]['core'], reverse=True):
        if G.node[v]['core'] >= maxSize:
            C = []
            for u in sorted(list(G.neighbors(v)), key=lambda x: G.node[x]['core'], reverse=True):
                C.append(u)
                if (is_clique(C,G)== True):
                    if maxSize < len(C):
                        clique = C
                        maxSize = len(C)
                else:
                    C.pop()
    print clique 
    return clique

def expand(C,P,list_cliques_found,currentSize,list_listNodes):
    for u in range( len(P) - 1, -1, -1):
        if len(P)+len(C)<=currentSize[len(currentSize)-1]: #  pruning 4
            return
        v=P[u]
        C.append(v)
        neighbourHood = list_listNodes[v]
        newP = []
        for w in P:
            if w in neighbourHood and len(list_listNodes[w])>= currentSize[len(currentSize)-1]-1:
                newP.append(w)
        ######################################################################
        if len(newP)==0 and len(C) > currentSize[len(currentSize)-1]:
            # currentSize = update_size(C)
            currentSize.append(len(C[:]))
            list_cliques_found.append(C[:])
        if len(newP) > 0:
            expand(C,newP,list_cliques_found,currentSize,list_listNodes)
        C.pop()    
        P.pop()


def find_a_maximum_clique(G3):
    # G = network_from_sinduja_data()
    # G3 = simplification_minimum_lost_connectivity(G,0.8)
    edges1 =[('60','100'),('52','100'),('50','100'),('38','100'),('7','100')]
    # for test and validate result: just to be sure network have only one maximum clique 
    G3.add_edges_from(edges1)
    #currentSize = 2
    list_cliques_found = [] # clique found during the recursive
    listNodes_original = []
    listNodes = []# from 0 to n-1. Node j is at position j
    count = 0 
    list_listNodes = []
    # another way to store graph, element j is the list neighbors of node j
    atr = {}
    for n, d in sorted(G3.degree, key=lambda x: x[1], reverse=True): 
        listNodes_original.append(n)
        listNodes.append(count)
        atr.update({n:count}) 
        count = count + 1
    
    ###############################################################################
    # simulate the original network by generating another network node count from 0
    # to n 
    nx.set_node_attributes(G3, atr, name='count')
    
    for n, d in sorted(G3.degree, key=lambda x: x[1], reverse=True):
        nb = []
        for nn in G3.neighbors(n):
            nb.append(G3.node[nn]['count'])
        list_listNodes.append(nb) 
            
    list_cliques_found = []
    #len(fast_heuristic_clique(G3))
    minSize = len(fast_heuristic_clique(G3))
    currentSize = []
    currentSize.append(minSize)
    
    for i in listNodes:
    # top = len(G3)# consider only top nodes
    # for i in range(top):# pruning 1
    # In addition we incorporate novel pruning techniques based on the investor behavior context. 
    # Over-expression of specific investor attributes
        N = list_listNodes[i]
        if len(N) >= minSize-1:# pruning 2
            # a large clique in the network during bull market, the bubble propagates, or  by setting a high input value of minimum size
            C = [] # current clique
            C.append(i)
            P = [] # nodes could be added to form clique
            for j in N:
                if j > i: # pruning 3
                    if len(list_listNodes[j]) >= minSize-1:
                        P.append(j)
            #######################################################################
            if len(P) > 0:
                expand(C,P,list_cliques_found,currentSize,list_listNodes)
            
            # retun here
    maximum_clique = []
    if len(list_cliques_found) > 0:
        pos= len(list_cliques_found)-1
        maximum_clique = retrieve(G3,list_cliques_found[pos])
    return maximum_clique


def test():
    G = network_from_sinduja_data()
    G3 = simplification_minimum_lost_connectivity(G,0.8)
    clq2 = list(nx.find_cliques(G3))
    maxcl = clq2[0]
    size = len(maxcl)
    for cl in clq2:
        if len(cl) > size:
            maxcl = cl
            size = len(cl)
               
    maximum_clique = find_a_maximum_clique(G3)
    clq2 = list(nx.find_cliques(G3))
    for cl in clq2:
        if len(cl)==len(maximum_clique):
            print cl
test()