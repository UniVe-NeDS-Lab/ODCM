import networkx as nx
import numpy as np
import pandas as pd
from utils import *
from tqdm.contrib.concurrent import process_map  # or thread_map


def average_path_length(graph: nx.Graph):
    gws = [n for n in graph if 'type' in graph.nodes[n] and graph.nodes[n]['type'] == 'gateway']
    plenghts = nx.multi_source_dijkstra_path_length(graph, gws)
    lengths = np.array(list(plenghts.values()))

    return lengths.mean(), lengths.max()

def get_clusters(graph: nx.Graph):
    cc = nx.connected_components(graph)
    sizes = []
    for c in cc:
        size = 0
        has_gw = False
        for n in c:
            if graph.nodes[n].get('type')=='gateway':
                has_gw = True
            size += graph.nodes[n]['subscriptions']
        if has_gw:
            sizes.append(size)
    return sizes

def compute_simple_metricses(graphs):
    data = process_map(_compute_simple_metrics, graphs, max_workers=16, chunksize=100)
    df = pd.DataFrame(data) 
    return df

def _compute_simple_metrics(graph):
    data = []
    (p, mgb, w_g,f_g, n_subs)  = graph
    area, ratio, cluster_size, algo, n_gw, time, random_seed = p
    measures = {}
    measures['nodes'] = len(w_g)
    measures['n_gw'] = n_gw
    measures['algo'] = algo
    measures['subscriptions'] = n_subs
    measures['edges'] = len(w_g.edges())
    measures['avg_pathl'], measures['max_pathl'] = average_path_length(w_g)
    measures['area'] = area
    measures['ratio'] = int(ratio)
    measures['gateways'] = len([n for n,att in w_g.nodes(data=True) if att.get('type')=='gateway'])
    measures['leaves'] = len([n for n in w_g.nodes() if w_g.degree()[n]==1])
    measures['relays'] = len([n for n in w_g.nodes() if w_g.degree()[n]>1]) - measures['gateways']
    measures['relays_ratio'] =  measures['relays']/n_subs
    unc = sum([d['subscriptions'] for n, d in w_g.nodes(data=True) if w_g.degree()[n] == 0])
    all_sub = sum([d['subscriptions'] for n, d in w_g.nodes(data=True)])
    measures['unc'] = unc
    measures['unconnected'] = unc/all_sub
    measures['cluster_size'] = int(cluster_size)
    measures['connected_components'] = get_clusters(w_g)
    measures['time'] = time
    measures['seed'] = random_seed
    return measures



def compute_n_ants(graphs):
    from multiprocessing import Pool
    print("Start n_ant compute")
    with Pool(16) as p:
        data = p.map(_compute_n_ant, graphs)
    print("End n_ant compute")
    #data = process_map(_compute_n_ant, graphs, max_workers=16, chunksize=100)
    ant_df = pd.DataFrame(flatten(data)) 
    return ant_df

def _compute_n_ant(graph):
    data = []
    (p, mgb, w_g,f_g, n_subs)  = graph
    area, ratio, cluster_size, algo, n_gw, time, random_seed = p

    gws = [n for n in w_g if 'type' in w_g.nodes[n] and w_g.nodes[n]['type'] == 'gateway']
    
    for n in w_g.nodes():
        if w_g.degree()[n] > 0:
            run = {}
            run['area'] = area
            run['cluster_size'] = cluster_size
            run['ratio'] =  ratio
            if n in gws:
                run['type'] = 'gw'
            elif w_g.degree()[n]==1:
                run['type'] = 'leaf'
            else:
                run['type'] = 'relay'
            run['antennae'] =  w_g.nodes[n]['n_ant']
            run['degree'] = w_g.degree()[n]
            run['n_gw'] = n_gw
            run['algo'] = algo
            run['area'] =  area
            run['mgb'] = mgb
            data.append(run)
    return data
