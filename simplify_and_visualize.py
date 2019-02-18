# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 15:33:51 2019

@author: le3
"""
import planarity
import networkx as nx
import numpy as np
from scipy.spatial.distance import squareform
import plotly as py
import plotly.graph_objs as go
import plotly.io as pio


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
        PMFG.add_edge(edge['source'], edge['dest'],weight = edge['weight'])
        if not planarity.is_planar(PMFG):
            PMFG.remove_edge(edge['source'], edge['dest'])
            
        if len(PMFG.edges()) == 3*(nb_nodes-2):
            break
    
    return PMFG
    
def planar_maximally_filter(G):
    #fn = 'corr_nokia.gexf'
    #G = nx.read_gexf(fn)
    for source, dest, data in sorted(G.edges(data=True),
                                 key=lambda x: abs(x[2]['weight'])):
        if data <= 0.1:
            G.remove_edge(source,dest)
    sorted_edges = sort_graph_edges(G)
    G2 = compute_PMFG(sorted_edges, len(G.nodes))  
    return G2

def simplify_using_spanningTree(G):
    minT = nx.minimum_spanning_tree(G,weight='weight')
    maxT = nx.maximum_spanning_tree(G,weight='weight')
    return minT,maxT
    
def testcase_simulate_graph():
    nb_nodes = 150
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
    G2 = G
    print 'length', len(G.edges())
    # novel idea: remove edges whose weight close to 0 (less positve and less negative)
    for source, dest, data in sorted(G.edges(data=True),
                                     key=lambda x: abs(x[2]['weight'])):
        G.remove_edge(source,dest)
        if (nx.is_connected(G) and count<n):
            be_removed.append((source, dest))
            count = count+1
        else:
            print 'break for loop'
            break 
    G2.remove_edges_from(be_removed)
    # print 'length', len(G2.edges())
    return G2

def remove_insignificant_links(G):

    # novel idea: remove edges whose weight close to 0 (less positve and less negative)
    for source, dest, data in sorted(G.edges(data = True),
                                     key=lambda x: abs(x[2]['weight'])):
        if data <= 0.1:
            G.remove_edge(source,dest)
    return G
    
def network_from_sinduja_data(k):
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
    with open(files[k], "r") as ins:
        for line in ins:
            a=line.split(',')
            links.append((a[0],a[1],float(a[2])))
    G = nx.Graph()   
    #G2.add_nodes_from(investors)
    G.add_weighted_edges_from(links)
    return G
    '''
    cliques=nx.find_cliques(G)
    for clique in cliques:
        print len(clique)
    G1 = testCase_investor_network(G)
    cliques2=nx.find_cliques(G1)
    for clique in cliques2:
        print len(clique)
    '''
     
def mainf():
    G = network_from_sinduja_data(0)
    G1 = planar_maximally_filter(G)
    G2 = simplification_minimum_lost_connectivity(G,0.8)
    
    for u,v,weight in G.edges(data='weight', default=1):
        print u
        print v
        print weight
    # list(G.edges_iter(data='weight', default=1))
    # w1 = nx.is_weighted(G, weight='weight') 
    # print w1
    return G1,G2

def compare_result(gama):
    import glob
    import numpy as np
    import pandas as pd
    files = glob.glob('./financial_institution2/*')
    mdate = []
    mG1 = []
    mG2 = []
    mG3 = []
    mG4 = []
    
    for k in range (len(files)):
        mdate.append(files[k][-15:-4])
        G = network_from_sinduja_data(k)
        G1 = planar_maximally_filter(G)
        G2 = simplification_minimum_lost_connectivity(G,gama)
        minT = nx.minimum_spanning_tree(G,weight = 'weight')
        maxT = nx.maximum_spanning_tree(G,weight = 'weight')
        w1 = [weight1 for u1,v1,weight1 in G1.edges(data = 'weight', default = 1)]
        w2 = [weight2 for u2,v2,weight2 in G2.edges(data = 'weight', default = 1)]
        w3 = [weight3 for u3,v3,weight3 in minT.edges(data = 'weight', default = 1)]
        w4 = [weight4 for u4,v4,weight4 in maxT.edges(data = 'weight', default = 1)]
        print len(w1)
        print len(w2)
        mG1.append(np.mean(w1))
        mG2.append(np.mean(w2))
        mG3.append(np.mean(w3))
        mG4.append(np.mean(w4))
        
    df = pd.DataFrame({'date':mdate,'PMFG':mG1,'minimum_lost':mG2,'min_tree':mG3,'max_tree':mG4},columns=['date','PMFG','minimum_lost','min_tree','max_tree'])
    df['date'] =pd.to_datetime(df.date)
    df = df.sort_values(by='date')
    #print df.groupby(df.date.dt.year)['PMFG', 'minimum_lost'].transform('mean')
    
    #df = df.groupby([df.date.dt.strftime('%Y')])['PMFG','minimum_lost','min_tree','max_tree'].mean()
    #df.index.name = 'date'
    #df.reset_index(level=0, inplace=True)
    return df
    #df.index=isin_list
    #df.index.name = 'date'
    #df.reset_index(level=0, inplace=True)
    #df=df[df['bowtie']==True]
    #df2=df.sort_values(by='pscc', ascending=False)     
    
def visualize(df):
    trace1 = go.Scatter(
                    x=df.date,
                    y=df['PMFG'],
                    name = "PMFG",
                    line = dict(color = 'rgb(244, 116, 66)'),
                    opacity = 0.8)
    
    trace2 = go.Scatter(
                    x=df.date,
                    y=df['minimum_lost'],
                    name = "Minimum Lost Connectivity",
                    line = dict(color = 'rgb(116, 244, 65)'),
                    opacity = 0.8)
    
    trace3 = go.Scatter(
                    x=df.date,
                    y=df['min_tree'],
                    name = "Minimum Spanning Tree",
                    line = dict(color = 'rgb(244, 241, 65)'),
                    opacity = 0.8)
    
    trace4 = go.Scatter(
                    x=df.date,
                    y=df['max_tree'],
                    name = "Maximum Spanning Tree",
                    line = dict(color = 'rgb(169, 65, 244)'),
                    opacity = 0.8)
    
    data = [trace1,trace2,trace3,trace4]
    
    layout = dict(
        title = "compare network simplification techniques"
    )
    
    fig = go.Figure(data=data, layout=layout)
    py.offline.plot(fig)
    pio.write_image(fig, 'comparison.pdf')


    
df = compare_result(0.8) 
print df  
visualize(df)