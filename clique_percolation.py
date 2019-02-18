# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 15:33:51 2019
This module implements algorithm for clique percolation.
@author: le3
"""
# import planarity
import networkx as nx
import numpy as np
from scipy.spatial.distance import squareform

def simplification_minimum_lost_connectivity(G,gama):
    # Zhou, Fang, Sebastien Malher, and Hannu Toivonen. 
    # "Network simplification with minimal loss of connectivity." Data Mining (ICDM), 2010 IEEE 10th International Conference on. IEEE, 2010.
    #  The algorithm is a naive approach, simply pruning a fraction of the weakest edges by sorting edges according to the edge weight
    n = gama*(len(G.edges())-len(G.node())+1)
    be_removed = []
    count = 0
    print 'length', len(G.edges())
    # novel idea: remove edges whose weight close to 0 (less positve and less negative)
    for source, dest, data in sorted(G.edges(data=True),
                                     key=lambda x: abs(x[2]['weight'])):
        G.remove_edge(source,dest)
        print 'length', len(G.edges())
        if (nx.is_connected(G) and count<n):
            be_removed.append((source, dest))
            count = count+1
        else:
            print 'break for loop'
            break 
    print 'length', len(G.edges())
    return G

def remove_negative_weight_links(G):
    for source, dest, data in G.edges(data=True):
        if data['weight'] < 0:
            G.remove_edge(source,dest)
    return G

def network_from_sinduja_data():
    # generate network from published data:
    # https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0198807
    # Ranganathan, Sindhuja, Mikko Kivelä, and Juho Kanniainen. 
    #"Dynamics of investor spanning trees around dot-com bubble." PloS one 13.6 (2018): e0198807
    import glob
    files = glob.glob('./financial_institution/*')
    # files = glob.glob('./non_financial_institution/*')
    # files = glob.glob('./household/*')
    # data = pd.read_csv(files[0], sep=" ", header=None)
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
    # G1 = planar_maximally_filter(G)
    # G2 = simplification_minimum_lost_connectivity(G,0.8)
    return G 

from collections import defaultdict
# reference: http://sociograph.blogspot.com/2011/11/clique-percolation-in-few-lines-of.html
          
def get_percolated_cliques(G,cliques,k):
    # source code from networkx
    # general routine for clique percolation
    perc_graph = nx.Graph()
    # cliques = [frozenset(c) for c in nx.find_cliques(G) if len(c) >= k]
    perc_graph.add_nodes_from(cliques)

    # First index which nodes are in which cliques
    membership_dict = defaultdict(list)
    for clique in cliques:
        for node in clique:
            membership_dict[node].append(clique)

    # For each clique, see which adjacent cliques percolate
    for clique in cliques:
        for adj_clique in get_adjacent_cliques(clique, membership_dict):
            if len(clique.intersection(adj_clique)) >= (k - 1):
                perc_graph.add_edge(clique, adj_clique)

    # Connected components of clique graph with perc edges
    # are the percolated cliques
    for component in nx.connected_components(perc_graph):
        yield(frozenset.union(*component))

def get_adjacent_cliques(clique, membership_dict):
    adjacent_cliques = set()
    for n in clique:
        for adj_clique in membership_dict[n]:
            if clique != adj_clique:
                adjacent_cliques.add(adj_clique)
    return adjacent_cliques


def algorithm_1(G):
    # http://hal.elte.hu/cfinder/wiki/papers/communitylettm.pdf
    cls = list(nx.find_cliques(G))
    # [frozenset(c) for c in cls if len(c) >= k]
    k = 2
    b = True
    result = []
    while b == True and k < 7:
        list_cliques = [frozenset(c) for c in cls if len(c) >= k]
        cms = list(get_percolated_cliques(G,list_cliques,k))
        cm = sorted(cms, key=lambda x: len(x), reverse=False)
        if len(cm) >= 2:
            if len(cm[0])< 1.5*len(cm[1]):
                b = False 
                result = cm
                print k
                print cm
        k = k+1
    return k, result


def algorithm_2(G):
    # http://cfinder.org/wiki/papers/WeightedNetworkModules-Farkas-07-NewJPhys.pdf
    cls = list(nx.find_cliques(G))
    ws = [weight1 for u1,v1,weight1 in G.edges(data = 'weight', default = 1)]
    ws = sorted(ws)
    k = 0
    b = True
    result = []
    while b == True and k < len(ws):
        list_cliques = [frozenset(c) for c in cls if intensity(c,G) >= ws[k] and len(c) > 2]
        cms = list(get_percolated_cliques(G,list_cliques,k))
        cm = sorted(cms, key=lambda x: len(x), reverse=False)
        if len(cm) >= 2:
            if len(cm[0])< 2*len(cm[1]):
                b = False 
                result = cm
                print k
                print cm
        k = k+1
    return k, result

def geo_mean(iterable):
    a = np.array(iterable)
    return a.prod()**(1.0/len(a))


def intensity(clique,G):
    clq = G.subgraph(clique)
    w1 = [weight1 for u1,v1,weight1 in clq.edges(data = 'weight', default = 1)]
    return geo_mean(w1)
def test():    
    G = network_from_sinduja_data()   
    G3 = simplification_minimum_lost_connectivity(G,0.8)
    G4 = remove_negative_weight_links(G3)
    # k, result = algorithm_1(G4)
    k, result = algorithm_2(G4)
    print result
    print k
    
test()