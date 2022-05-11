import configargparse
import networkx as nx
import numpy as np
import pandas as pd
from Topology import Topology
import geopandas as gpd
from diskcache import Cache
import matplotlib.pyplot as plt
import math as m
import os
import time
import random
from sqlalchemy import create_engine
import metis
import tqdm
from misc import copy_graph
from collections import Counter
import osmnx as ox
import json
ox.config(use_cache=True, log_console=False)
# set the folder for the cache
cache = Cache(".cache")

@cache.memoize()
def read_graphml(dataset: str) -> tuple[gpd.GeoDataFrame | pd.Series, nx.Graph, nx.MultiGraph]:
    graph = nx.read_graphml(f"data/{dataset}.graphml.gz",node_type=int)
    nodes = pd.DataFrame.from_dict(graph.nodes, orient='index')
    nodes = nodes[nodes.households > 0]
    graph = nx.subgraph(graph, nodes.index)
    if dataset == 'casciana terme':
        #Casciana Terme changed name since last census, so OSM has a different name
        dataset='casciana terme lari'
    osmg = ox.graph_from_place(f'{dataset}, Italy')
    posmg = ox.project_graph(osmg, 'EPSG:3003')
    osm_road = ox.get_undirected(posmg)
    return nodes, graph,osm_road

# @cache.memoize()
# def read_data(base_folder: str, dataset: str) -> tuple[gpd.GeoDataFrame | pd.Series, nx.Graph, nx.MultiGraph]:
#     # Reads and join nodes and heights csv
#     nodes = pd.read_csv(f"{base_folder}/vg/{dataset}/best_p.csv", sep=',', header=0, names=['id', 'x', 'y'], dtype=int).set_index('id', drop=False)
#     nodes = gpd.GeoDataFrame(nodes, geometry=gpd.points_from_xy(nodes.x, nodes.y))
#     try:
#         heights = pd.read_csv(f"{base_folder}/{dataset}_2_2/heights.csv", sep=',', names=['id', 'h'], dtype={'id': int, 'h': float}).set_index('id')
#         nodes = nodes.join(heights)
#     except FileNotFoundError:
#         print("heights.csv file not found, skipping it")

#     hdf = pd.read_csv(f'{base_folder}/sociecon/{dataset}.csv',
#                       dtype={'id': int,
#                              'volume': float,
#                              'height': float,
#                              'area': float,
#                              'population': float,
#                              'households': float}
#                       ).set_index('id', drop=False)
#     nodes = nodes.join(hdf.drop('id', axis=1))  # Should drop id_rp
#     nodes = nodes[nodes.households > 0]
#     # Read graph from edgelist
#     graph = nx.read_edgelist(
#         f"{base_folder}/vg/{dataset}/distance.edgelist",
#         delimiter=" ",
#         nodetype=int,
#         data=[('dist', float)]
#     )
#     graph = nx.subgraph(graph, nodes.index)
#     nx.set_node_attributes(graph, nodes.drop('geometry', axis=1).to_dict('index'))
    
#     #Get road graph from OSM
#     if dataset == 'casciana terme':
#         #Casciana Terme changed name since last census, so OSM has a different name
#         dataset='casciana terme lari'
#     osmg = ox.graph_from_place(f'{dataset}, Italy')
#     posmg = ox.project_graph(osmg, 'EPSG:3003')
#     osm_road = ox.get_undirected(posmg)

#     return nodes, graph, osm_road


class Simulator():
    def __init__(self, args):
        self.dataset = args.dataset.lower()
        self.random_seed = args.seed
        self.random_state = np.random.RandomState(self.random_seed)
        self.nodes, self.graph, self.osm_road = read_graphml(self.dataset)
        self.total_households = int(self.graph.graph['total_households'])
        print(f"Total households : {self.total_households}")
        with open('fiber_pop.json') as fr:
            self.fiber_pop = json.load(fr)[self.dataset]

    def filter_nodes_households(self, subscribers_ratio):
        self.subscribers_ratio = subscribers_ratio
        target_households = int(self.total_households*self.subscribers_ratio)
        selected_nodes = self.nodes.sample(target_households, weights=self.nodes.households, replace=True).index
        subscriptions = Counter(selected_nodes)
        self.mynodes = self.nodes.loc[subscriptions.keys()]
        self.mynodes['subscriptions'] = pd.Series(subscriptions)
        assert(self.mynodes.subscriptions.sum() == target_households)
        self.vg_filtered = nx.subgraph(self.graph, self.mynodes.index)
        self.cgraph = self.vg_filtered# nx.subgraph(self.graph, max(nx.connected_components(self.vg_filtered), key=len))

    def clusterize_metis(self, cluster_size):
        self.cluster_size =  cluster_size
        self.n_clusters = m.ceil(self.mynodes.subscriptions.sum()/cluster_size)
        #Transform weights to integers round(dist*100)
        nx.set_edge_attributes(self.vg_filtered, {e: round(wt * 1e2) for e, wt in nx.get_edge_attributes(self.vg_filtered, "dist").items()}, "int_dist")
        nx.set_node_attributes(self.vg_filtered, self.mynodes.subscriptions.to_dict(),"subscriptions")
        #Apply metis algorithm for graph partitioning
        self.vg_filtered.graph['edge_weight_attr'] = 'int_dist'
        self.vg_filtered.graph['node_weight_attr'] = 'subscriptions'
        if self.n_clusters == 1:
            #Don't clusterize if I want only one cluster
            self.mynodes['cluster'] = 0
        else:
            edgecuts, clusters = metis.part_graph(self.vg_filtered, self.n_clusters)
            self.mynodes['cluster'] = pd.Series({n:clusters[ndx] for ndx, n in enumerate(self.vg_filtered.nodes())})
               

    def plot_clusters(self):
        plt.figure(figsize=(20,20))
        for idx, c in enumerate(self.mynodes.cluster.unique()):
            clnodes = self.mynodes[self.mynodes.cluster == c]
            print(len(clnodes))
            ax = plt.subplot(4,4,int(idx+1))
            ax.scatter(clnodes.x, clnodes.y)
            ax.set_title(f'cluster {c}')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim([self.nodes.x.min(), self.nodes.x.max()])
            ax.set_ylim([self.nodes.y.min(), self.nodes.y.max()])
        # plt.scatter(self.kmeans.cluster_centers_[:, 0], self.kmeans.cluster_centers_[:, 1], color='black')
        plt.savefig('results/clusters.pdf')
        plt.tight_layout()
        exit(0)

    def generate_topologies(self):
        base_dir = f'results/{self.dataset}_{(self.subscribers_ratio*100):.0f}_{self.cluster_size}/'
        os.makedirs(base_dir, exist_ok=True)
        TG = Topology(self.graph, self.mynodes, self.n_clusters)
        TG.extract_graph()
        TG.fiber_backhaul(self.osm_road, self.fiber_pop)
        TG.save_graph(f'{base_dir}/{time.time()*10:.0f}_{self.random_seed}')


def main():
    parser = configargparse.ArgumentParser(description='Generate reliable backhaul infrastructures', default_config_files=['sim.yaml'])
    parser.add_argument('-D', '--dataset', help='dataset')
    parser.add_argument('--cluster_size', type=int, action='append')
    parser.add_argument('--subscribers_ratio', type=float, action='append')
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--seed', help='random seed', default=int(random.random()*100))
    args = parser.parse_args()
    s = Simulator(args)
    pbar = tqdm.tqdm(total=len(args.cluster_size)*len(args.subscribers_ratio)*args.runs)
    for cs in args.cluster_size:
        for sr in args.subscribers_ratio:
            for run in range(args.runs):
                s.filter_nodes_households(sr)
                s.clusterize_metis(cs)
                s.generate_topologies()
                pbar.update(1)
    pbar.close()
            
        


if __name__ == '__main__':
    main()
