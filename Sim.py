import pdb
from typing import Type
import configargparse
import networkx as nx
import Topology
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from diskcache import Cache
from k_means_constrained import KMeansConstrained
import matplotlib.pyplot as plt
import os
import time
import random
import pdb
import networkit as nk
from sqlalchemy import create_engine
import metis
from misc import copy_graph
# set the folder for the cache
cache = Cache(".cache")

DSN = "postgresql://dbreader:dBReader23@@34.76.18.0/TrueNets"


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
        self.cluster_size = args.cluster_size
        engine = create_engine(DSN)
        istat = gpd.read_postgis(f"""SELECT "PF1", geom, sez2011_tx::bigint as sez2011
                                FROM istat_agcom
                                WHERE lower("COMUNE") = '{self.dataset}'""", engine).set_index('sez2011', drop=False)
        self.random_seed = args.seed
        
        self.random_state = np.random.RandomState(self.random_seed)
        self.nodes, self.graph = read_data(self.base_folder, self.dataset)
        self.total_households = int(istat['PF1'].sum())
        print(f"Total households : {self.total_households}")
        nx.set_node_attributes(self.graph, self.nodes.drop('geometry', axis=1).to_dict('index'))
        self.gw_strategy_name = args.gw_strategy
        self.topo_strategy_name = args.topo_strategy
        self.sub_area_nodes = self.nodes[self.nodes['households'] > 0]

    def filter_nodes_households(self, pop_ratio):
        self.pop_ratio = pop_ratio
        target_households = int(self.total_households*self.pop_ratio)
        self.mynodes = self.nodes.sample(target_households, weights=self.nodes.households, replace=True)
        print(f"Got {len(self.mynodes)} buildings")
        self.vg_filtered = nx.subgraph(self.graph, self.mynodes.index)
        print(f"Graph has size {len(self.vg_filtered)}")
        #We  have less nodes than 0.2*target_households because many nodes are repeated
        #TODO: save the repeated nodes into the graph so that we can analyzie that later
        self.cgraph = nx.subgraph(self.graph, max(nx.connected_components(self.vg_filtered), key=len))
        print(f"Connected Graph has size {len(self.cgraph)}")

    def clusterize_louvain(self):
        #Cant set size
        self.n_clusters = len(self.mynodes)//self.cluster_size
        clusters = nx.community.louvain_communities(self.vg_filtered, resolution=1.1)
        print([len(x) for x in clusters])

    def clusterize_groupcentr(self):
        self.n_clusters = len(self.mynodes)//self.cluster_size
        nkg = nk.nxadapter.nx2nk(self.vg_filtered)
        cent = nk.centrality.GroupCloseness(nkg, k=self.n_clusters)
        cent.run()
        nkgws = cent.groupMaxCloseness()
        gateways = [list(self.vg_filtered.nodes())[nd] for nd in nkgws]
        paths = nx.multi_source_dijkstra_path(self.vg_filtered, sources=gateways, weight='dist')
        for node, path in paths.items():
            self.mynodes.loc[node, 'cluster'] = gateways.index(path[0])

    def clusterize_metis(self, cluster_size):
        self.n_clusters = len(self.mynodes)//cluster_size
        #Transform weights to integers round(dist*100)
        nx.set_edge_attributes(self.vg_filtered, {e: round(wt * 1e2) for e, wt in nx.get_edge_attributes(self.vg_filtered, "dist").items()}, "int_dist")
        #Apply metis algorithm for graph partitioning
        self.vg_filtered.graph['edge_weight_attr'] = 'int_dist'
        edgecuts, clusters = metis.part_graph(self.vg_filtered, self.n_clusters)
        self.mynodes['cluster'] = pd.Series({n:clusters[ndx] for ndx, n in enumerate(self.vg_filtered.nodes())})
        

    def clusterize_kmeans(self, cluster_size):
        ##Spatial clustering
        self.n_clusters = len(self.mynodes)//cluster_size
        print(self.n_clusters)
        self.kmeans = KMeansConstrained(
            init="random",
            n_clusters=self.n_clusters,
            size_max=cluster_size*1.1,
            size_min=cluster_size*0.9,
            random_state=self.random_state
        )
        clusters = self.kmeans.fit_predict(self.mynodes[['x', 'y']])
        self.mynodes['cluster'] = clusters
        #Compute closeness for each node and find the highest in the spatial cluster
        self.mynodes['closeness'] = 0
        centrality = pd.Series(nx.centrality.closeness_centrality(self.vg_filtered))
        gateways = []
        for c in clusters:
            try:
                nodes = self.mynodes[self.mynodes.cluster == c]
                gw = centrality[centrality.index.isin(nodes.index)].idxmax()
                gateways.append(gw)
            except ValueError:
                print(f'{c} is empty')
                #Filter out unconnected nodes
                pass 
        paths = nx.multi_source_dijkstra_path(self.vg_filtered, sources=gateways, weight='dist')
        for node, path in paths.items():
            self.mynodes.loc[node, 'cluster'] = gateways.index(path[0])
            
        

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

    def generate_topologies_old(self, gw_strategy_name, topo_strategy_name):
        base_dir = f'results/{self.dataset}_{(self.density_nodes):.0f}_{self.area_radius:.0f}_{self.area_center[0]:.0f}_{self.area_center[1]:.0f}/'
        os.makedirs(base_dir, exist_ok=True)
        for i in range(self.n_clusters):
            mydf = self.mynodes[self.mynodes.cluster == i]
            cluster_vg = copy_graph(self.graph.subgraph(mydf.id))
            TG_class = Topology.get_topology_generator(topo_strategy_name, gw_strategy_name)
            TG = TG_class(cluster_vg)
            TG.extract_graph()
            TG.save_graph(f'{base_dir}/{time.time()*10:.0f}_{self.random_seed}_{topo_strategy_name}_{gw_strategy_name}_{i}.graphml.gz')

    def generate_topologies(self, gw_strategy_name, topo_strategy_name):
        base_dir = f'results/{self.dataset}_{(self.pop_ratio*100):.0f}/'
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
    for cs in args.cluster_size:
        for pr in args.pop_ratio:
            for run in range(args.runs):
                s.filter_nodes_households(pr)
                s.clusterize_metis(cs)
                #s.plot_clusters()
                for gs in args.gw_strategy:
                    for ts in args.topo_strategy:
                        s.generate_topologies(gs, ts)
            
        


if __name__ == '__main__':
    main()
