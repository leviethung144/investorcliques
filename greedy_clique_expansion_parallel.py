# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 15:33:51 2019
This module implements parallel greedy clique expansion.
@author: le3
"""
# import planarity
import networkx as nx
from collections import defaultdict


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
    # print 'length', len(G.edges())
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

def cliques_enumerate_using_networkx():  
    G1,G2 = network_from_sinduja_data()
    cliques = nx.find_cliques(G1)
    for clique in cliques:
        print len(clique)
    cliques2 = nx.find_cliques(G2)
    for clique in cliques2:
        print len(clique)


def find_all_cliques_for_a_subgraph(G,minSize):
    cls1 = find_cliques_recursive(G)
    res1=[]
    for cl in cls1:
        #cl.sort(reverse = True)
        if len(cl) >= minSize:
            res1.append(cl)
            print cl
    # result = set(tuple(clq) for clq in res1)
    #print len(result)
    #print len(res1)
    return res1


def find_cliques_recursive(G):
    # https://www.sciencedirect.com/science/article/pii/S0304397506003586?via%3Dihub
    # this famous algorithm is implemented in networkx:
    # https://pelegm-networkx.readthedocs.io/en/latest/reference/generated/networkx.algorithms.clique.find_cliques.html
    if len(G) == 0:
        return iter([])
    adj = {u: {v for v in G[u] if v != u} for u in G}
    Q = []

    def expand(subg, cand):
        u = max(subg, key=lambda u: len(cand & adj[u]))
        for q in cand - adj[u]:
            cand.remove(q)
            Q.append(q)
            adj_q = adj[q]
            subg_q = subg & adj_q
            if not subg_q:
                yield Q[:]
            else:
                cand_q = cand & adj_q
                if cand_q:
                    for clique in expand(subg_q, cand_q):
                        yield clique
            Q.pop()

    return expand(set(G), set(G))


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

def internal_degree(cm,G):
    cmt = G.subgraph(cm)
    return 2*len(cmt.edges())

def external_degree(cm,G):
    not_cm = [n for n in G if n not in cm]
    count = 0
    # frontier = []
    for n in cm:
        for m in not_cm:
            if G.has_edge(n,m):
                count = count +1
                # frontier.append(m)
    return count

def candidate(cm,G):
    not_cm = [n for n in G if n not in cm]
    # count = 0
    frontier = []
    for n in cm:
        for m in not_cm:
            if G.has_edge(n,m):
                # count = count +1
                frontier.append(m)
    ###############
    return frontier
 
def fitness(cm,G,alpha):
    count = external_degree(cm,G)
    internal = float(internal_degree(cm,G))
    return internal/float((count + internal)**alpha)

def fitness_node(node,cm,G,alpha):
    cm1 = cm[:]
    cm1.append(node)
    return  fitness(cm1,G,alpha)

def smaller(c1,c2):
    a = len(c1)
    if len(c1) > len(c2):
        a = len(c2)
    return a
      
        
def similarity(c1,c2):
    intersection = [a for a in c1 if a in c2]
    embed = len(intersection)/float(smaller(c1,c2))
    # print embed
    return embed
    
def greedy_expansion_2(cm,G,alpha,cms):
    b = True
    # print 'before', cm
    while b == True:
        frontier = candidate(cm,G)
        # print len(frontier)
        if len(frontier) > 0:
            u = max(frontier, key=lambda node: fitness_node(node,cm,G,alpha))
            if fitness_node(u,cm,G,alpha)> fitness(cm,G,alpha):
                cm.append(u)
            else: # stop
                b = False
        else:
            b = False
    # print 'after', cm
    cms.append(cm)
    
def greedy_expansion(cm,G,alpha):
    b = True
    # print 'before', cm
    while b == True:
        frontier = candidate(cm,G)
        # print len(frontier)
        if len(frontier) > 0:
            u = max(frontier, key=lambda node: fitness_node(node,cm,G,alpha))
            if fitness_node(u,cm,G,alpha)> fitness(cm,G,alpha):
                cm.append(u)
            else: # stop
                b = False
        else:
            b = False
    # print 'after', cm
    return cm

def paralell_greedy_expansion(G, cls,alpha):
    # return list of partitions. 
    cms = []
    future_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        for cm in cls:
            future_results.append(executor.submit(greedy_expansion_2(cm,G,alpha,cms)))
        concurrent.futures.wait(future_results)
    # pick the best one with the highest modularity
    # result = max(cms, key=lambda j: modularity(G,j))
    return cms

def remove_duplicate(items):
    found = []
    keep = []
    for item in items:
        if contain(found,item):
            pass
        else:
            found.append(item)
            keep.append(item)
    return keep

def contain(cms,cm0):
    b = False
    for cm in cms:
        if similarity(cm0,cm) > 0.8:
            b = True
            break
            
    return b

def modularity(G, communities):
    m = G.size(weight='weight')
    out_degree = dict(G.degree(weight='weight'))
    in_degree = out_degree
    norm = 1 / (2 * m)
    norm2 = 1 / (4 * m)
    # weight = nx.get_edge_attributes(G,'weight')
    def val(u, v):
        w = 0.0
        if G.has_edge(u,v):
            w = G[u][v].get('weight', 1)
        # w=weight[(u,v)]
        if u == v:
            w *= 2
        return w - in_degree[u] * out_degree[v] * norm

    Q = 0 
    for c in communities:
        for u in c:
            for v in c:
                Q = Q + val(u, v)
    # Q = sum(val(u, v) for c in communities for u, v in product(c, repeat=2))
    return Q * norm2

def _naive_greedy_modularity_communities(G):
    """Find communities in graph using the greedy modularity maximization.
    This implementation is O(n^4), much slower than alternatives, but it is
    provided as an easy-to-understand reference implementation.
    Source code from networkx
    """
    # First create one community for each node
    communities = list([frozenset([u]) for u in G.nodes()])
    # Track merges
    merges = []
    # Greedily merge communities until no improvement is possible
    old_modularity = None
    new_modularity = modularity(G, communities)
    while old_modularity is None or new_modularity > old_modularity:
        # Save modularity for comparison
        old_modularity = new_modularity
        # Find best pair to merge
        trial_communities = list(communities)
        to_merge = None
        for i, u in enumerate(communities):
            for j, v in enumerate(communities):
                # Skip i=j and empty communities
                if j <= i or len(u) == 0 or len(v) == 0:
                    continue
                # Merge communities u and v
                trial_communities[j] = u | v
                trial_communities[i] = frozenset([])
                trial_modularity = modularity(G, trial_communities)
                if trial_modularity >= new_modularity:
                    # Check if strictly better or tie
                    if trial_modularity > new_modularity:
                        # Found new best, save modularity and group indexes
                        new_modularity = trial_modularity
                        to_merge = (i, j, new_modularity - old_modularity)
                    elif (
                        to_merge and
                        min(i, j) < min(to_merge[0], to_merge[1])
                    ):
                        # Break ties by choosing pair with lowest min id
                        new_modularity = trial_modularity
                        to_merge = (i, j, new_modularity - old_modularity)
                # Un-merge
                trial_communities[i] = u
                trial_communities[j] = v
        if to_merge is not None:
            # If the best merge improves modularity, use it
            merges.append(to_merge)
            i, j, dq = to_merge
            u, v = communities[i], communities[j]
            communities[j] = u | v
            communities[i] = frozenset([])
    # Remove empty communities and sort
    communities = [c for c in communities if len(c) > 0]
    for com in sorted(communities, key=lambda x: len(x), reverse=True):
        yield com




def test2():    
    B = simplification_minimum_lost_connectivity(network_from_sinduja_data(),0.8)
    A = remove_negative_weight_links(B)
    # G4 = nx.fast_gnp_random_graph(100,0.2)
    clqs = enumerate_all_maximal_cliques_parallel(A)
    # sequencial
    # cms =  [greedy_expansion(c[:],A,3.0) for c in clqs if len(c)>3]
    cls = [c for c in clqs if len(c)>3] 
    cms = paralell_greedy_expansion(A, cls,3.0)
    results = remove_duplicate(cms)
    # print results
    print modularity(A, results)
    ################################################################
    cms2 =  [greedy_expansion(c[:],A,1.0) for c in clqs if len(c)>3]  
    results2 = remove_duplicate(cms2)
    # print results2
    # from networkx.algorithms.community.quality import modularity as md
    # print modularity(A, list(_naive_greedy_modularity_communities(A)))
    # print 2*md(A, list(_naive_greedy_modularity_communities(A)))
    print modularity(A, results2)
    
test2()