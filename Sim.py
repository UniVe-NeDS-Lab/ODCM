import configargparse
import networkx as nx
import Topology
import numpy as np
import pandas as pd
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
# set the folder for the cache
cache = Cache(".cache")

DSN = "postgresql://dbreader:dBReader23#@34.76.18.0/TrueNets"


@cache.memoize()
def read_data(base_folder: str, dataset: str) -> tuple[gpd.GeoDataFrame, nx.Graph]:
    # Reads and join nodes and heights csv
    nodes = pd.read_csv(f"{base_folder}/vg/{dataset}/best_p.csv", sep=',', header=0, names=['id', 'x', 'y'], dtype=int).set_index('id', drop=False)
    nodes = gpd.GeoDataFrame(nodes, geometry=gpd.points_from_xy(nodes.x, nodes.y))
    try:
        heights = pd.read_csv(f"{base_folder}/{dataset}_2_2/heights.csv", sep=',', names=['id', 'h'], dtype={'id': int, 'h': float}).set_index('id')
        nodes = nodes.join(heights)
    except FileNotFoundError:
        print("heights.csv file not found, skipping it")

    hdf = pd.read_csv(f'{base_folder}/sociecon/{dataset}.csv',
                      dtype={'id': int,
                             'volume': float,
                             'height': float,
                             'area': float,
                             'population': float,
                             'households': float}
                      ).set_index('id', drop=False)
    nodes = nodes.join(hdf.drop('id', axis=1))  # Should drop id_rp
    nodes = nodes[nodes.households > 0]
    # Read graph from edgelist
    graph = nx.read_edgelist(
        #f"{base_folder}/vg/{dataset}/best_p_intervisibility.edgelist",
        f"{base_folder}/vg/{dataset}/distance.edgelist",
        delimiter=" ",
        nodetype=int,
        data=[('dist', float)]
    )
    graph = nx.subgraph(graph, nodes.index)
    nx.set_node_attributes(graph, nodes.drop('geometry', axis=1).to_dict('index'))
    return nodes, graph


class Simulator():
    def __init__(self, args):
        self.base_folder = args.base_folder
        self.dataset = args.dataset.lower()
        engine = create_engine(DSN)
        istat = gpd.read_postgis(f"""SELECT "PF1", geom, sez2011_tx::bigint as sez2011 FROM istat_agcom WHERE lower("COMUNE") = 'magliano in toscana'""", engine).set_index('sez2011', drop=False)
        self.random_seed = args.seed
        
        self.random_state = np.random.RandomState(self.random_seed)
        self.nodes, self.graph = read_data(self.base_folder, self.dataset)
        self.total_households = int(istat['PF1'].sum())
        #print(f"Total households : {self.total_households}")
        nx.set_node_attributes(self.graph, self.nodes.drop('geometry', axis=1).to_dict('index'))
        self.gw_strategy_name = args.gw_strategy
        self.topo_strategy_name = args.topo_strategy
        self.sub_area_nodes = self.nodes[self.nodes['households'] > 0]

    def filter_nodes_households(self, pop_ratio):
        self.pop_ratio = pop_ratio
        target_households = int(self.total_households*self.pop_ratio)
        selected_nodes = self.nodes.sample(target_households, weights=self.nodes.households, replace=True).index
        subscriptions = Counter(selected_nodes)
        self.mynodes = self.nodes.loc[subscriptions.keys()]
        self.mynodes['subscriptions'] = pd.Series(subscriptions)
        assert(self.mynodes.subscriptions.sum() == target_households)
        #print(f"Got {len(self.mynodes)} buildings")
        self.vg_filtered = nx.subgraph(self.graph, self.mynodes.index)
        #print(f"Graph has size {len(self.vg_filtered)}")
        self.cgraph = nx.subgraph(self.graph, max(nx.connected_components(self.vg_filtered), key=len))
        #print(f"Connected Graph has size {len(self.cgraph)}")

    def clusterize_metis(self, cluster_size):
        self.cluster_size =  cluster_size
        self.n_clusters = m.ceil(len(self.mynodes)/cluster_size)
    
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

    def generate_topologies(self, gw_strategy_name, topo_strategy_name):
        base_dir = f'results/{self.dataset}_{(self.pop_ratio*100):.0f}_{self.cluster_size}/'
        os.makedirs(base_dir, exist_ok=True)

        TG_class = Topology.get_topology_generator(topo_strategy_name, gw_strategy_name, self.n_clusters)
        TG = TG_class(self.graph, self.mynodes)
        TG.extract_graph()
        TG.save_graph(f'{base_dir}/{time.time()*10:.0f}_{self.random_seed}_{topo_strategy_name}_{gw_strategy_name}.graphml.gz')


def main():
    parser = configargparse.ArgumentParser(description='Generate reliable backhaul infrastructures', default_config_files=['sim.yaml'])
    parser.add_argument('--base_folder', help='path of the directory containing the visibility graphs')
    parser.add_argument('-D', '--dataset', help='dataset')
    parser.add_argument('--cluster_size', type=int, action='append')
    parser.add_argument('--pop_ratio', type=float, action='append')
    parser.add_argument('--gw_strategy', type=str, action='append')
    parser.add_argument('--topo_strategy', type=str, action='append')
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--seed', help='random seed', default=int(random.random()*100))
    args = parser.parse_args()
    s = Simulator(args)
    pbar = tqdm.tqdm(total=len(args.cluster_size)*len(args.pop_ratio)*args.runs)
    for cs in args.cluster_size:
        for pr in args.pop_ratio:
            for run in range(args.runs):
                s.filter_nodes_households(pr)
                s.clusterize_metis(cs)
                for gs in args.gw_strategy:
                    for ts in args.topo_strategy:
                        s.generate_topologies(gs, ts)
                pbar.update(1)
    pbar.close()
            
        


if __name__ == '__main__':
    main()
