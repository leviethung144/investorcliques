# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 15:33:51 2019
This module implements algorithm for maximal clique enumeration.
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
    # G1 = planar_maximally_filter(G)
    # G2 = simplification_minimum_lost_connectivity(G,0.8)
    return G 

import concurrent.futures

def find_cliques_node_ordering_parallel(graph):
    cliques = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_results = []
        p = set(graph.keys())
        r = set()
        # x = set()
        for v in range(len(p)):
        # for v in degeneracy_ordering(graph):
            n_v = graph[v]
            p_v = set([u for u in n_v if u>v])
            x_v = set([t for t in n_v if t<v])
            future_results.append(executor.submit(find_cliques_pivot(graph, r.union([v]), p_v, x_v, cliques)))
            # p.remove(v)
            # x.add(v)
        concurrent.futures.wait(future_results)
        # executor.shutdown(wait=True)
    return cliques

def find_cliques_node_ordering(graph):
    p = set(graph.keys())
    r = set()
    # x = set()
    cliques = []
    for v in range(len(p)):
    # for v in degeneracy_ordering(graph):
        n_v = graph[v]
        p_v = set([u for u in n_v if u>v])
        x_v = set([t for t in n_v if t<v])
        find_cliques_pivot(graph, r.union([v]), p_v, x_v, cliques)
        # p.remove(v)
        # x.add(v)
    return cliques

def find_cliques(graph):
    p = set(graph.keys())
    r = set()
    x = set()
    cliques = []
    for v in degeneracy_ordering2(graph):
        neighs = graph[v]
        find_cliques_pivot(graph, r.union([v]), p.intersection(neighs), x.intersection(neighs), cliques)
        p.remove(v)
        x.add(v)
    return cliques

def find_cliques_pivot(graph, r, p, x, cliques):
    if len(p) == 0 and len(x) == 0:
        cliques.append(r)
    else:
        # u = iter(p.union(x)).next()
        # u = random.choice(list(p.union(x)))
        u = max(list(p.union(x)), key=lambda j: len(p.intersection(graph[j])))
        for v in p.difference(graph[u]):
            neighs = graph[v]
            find_cliques_pivot(graph, r.union([v]), p.intersection(neighs), x.intersection(neighs), cliques)
            p.remove(v)
            x.add(v)
from collections import defaultdict

def degeneracy_ordering2(graph):
  ordering = []
  ordering_set = set()
  degrees = defaultdict(lambda : 0)
  degen = defaultdict(list)
  max_deg = -1
  for v in graph:
    deg = len(graph[v])
    degen[deg].append(v)
    degrees[v] = deg
    if deg > max_deg:
      max_deg = deg

  while True:
    i = 0
    while i <= max_deg:
      if len(degen[i]) != 0:
        break
      i += 1
    else:
      break
    v = degen[i].pop()
    ordering.append(v)
    ordering_set.add(v)
    for w in graph[v]:
      if w not in ordering_set:
        deg = degrees[w]
        degen[deg].remove(w)
        if deg > 0:
          degrees[w] -= 1
          degen[deg - 1].append(w)

  # ordering.reverse()
  return ordering            


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
def count(u,G):
    c=0
    for i in G[u]:
        if i>u:
            c = c+1
    return c
def reindex(G):
    graph = {node: list(G.neighbors(node)) for node in list(G.nodes())}
    order = degeneracy_ordering2(graph)
    # order = degeneracy_ordering(G)
    atr = {}
    for i in range (len(order)):
        atr.update({order[i]:i})
    nx.set_node_attributes(G, atr, name='count')
    graph2 = {} 
    for i in range (len(order)): 
        nb = []
        for nn in G.neighbors(order[i]):
            nb.append(G.node[nn]['count'])
        graph2.update({i:nb})
    return graph2



#################################################################
# compare paralell to sequencial 
def enumerate_all_maximal_cliques_parallel(G):
    graph = reindex(G)
    cliques1 = list(find_cliques_node_ordering_parallel(graph))
    # cliques2 = list(find_cliques_node_ordering(graph2))
    results = []
    for cl in cliques1:
        results.append(retrieve(G,cl))
        # print results
    
    return results

# print len(cliques2)
#################################################################  

def test (results,clq2):
    bl = []     
    for c1 in results:
        b = False
        for c2 in list(clq2):
            if set(c1) == set(c2):
                b = True
                break
        bl.append(b)
    for bb in bl:
        if(bb==False):
            print bb
    print bl
def test2():
    G = network_from_sinduja_data()   
    G3 = simplification_minimum_lost_connectivity(G,0.8)
    results =  enumerate_all_maximal_cliques_parallel(G3)   
    clq2 = list(nx.find_cliques(G3))
    test(results,clq2)
test2()