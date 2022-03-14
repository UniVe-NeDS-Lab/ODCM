import pdb
import configargparse
import networkx as nx
import Topology
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from sklearn.cluster import KMeans
from diskcache import Cache
from k_means_constrained import KMeansConstrained
import matplotlib.pyplot as plt
import os
import time
# set the folder for the cache
cache = Cache(".cache")


@cache.memoize()
def read_data(base_folder, dataset):
    # Reads and join nodes and heights csv
    nodes = pd.read_csv(f"{base_folder}/{dataset}_2_2/best_p.csv", sep=',', header=0, names=['id', 'x', 'y'], dtype=int).set_index('id', drop=False)
    nodes = gpd.GeoDataFrame(nodes, geometry=gpd.points_from_xy(nodes.x, nodes.y))
    heights = pd.read_csv(f"{base_folder}/{dataset}_2_2/heights.csv", sep=',', names=['id', 'h'], dtype={'id': int, 'h': float}).set_index('id')
    nodes = nodes.join(heights)
    # Read graph from edgelist
    graph = nx.read_edgelist(
        f"{base_folder}/{dataset}_2_2/distance.edgelist",
        delimiter=" ",
        nodetype=int,
        data=[('dist', float)]
    )
    return nodes, graph


class Simulator():
    def __init__(self, args):
        self.base_folder = args.base_folder
        self.dataset = args.dataset
        self.cluster_size = args.cluster_size
        self.area_radius = args.area_radius
        self.density_nodes = args.density_nodes
        self.nodes, self.graph = read_data(self.base_folder, self.dataset)

        if not args.area_center:
            self.centroid = self.nodes.dissolve().centroid[0]
            self.area_center = [self.centroid.x, self.centroid.y]
        else:
            self.area_center = list(map(float, args.area_center.split(',')))
            self.centroid = Point(self.area_center[0], self.area_center[1])

        self.area = self.centroid.buffer(self.area_radius)

    def filter_nodes(self):
        sub_area_nodes = self.nodes[self.nodes.intersects(self.area)]
        target_nodes = int(self.area.area*1e-6 * self.density_nodes)
        self.mynodes = sub_area_nodes.sample(target_nodes)

    def clusterize(self):
        self.n_clusters = len(self.mynodes)//self.cluster_size
        self.kmeans = KMeansConstrained(
            init="random",
            n_clusters=self.n_clusters,
            size_max=self.cluster_size*1.1,
            size_min=self.cluster_size*0.9,
            random_state=42
        )
        clusters = self.kmeans.fit_predict(self.mynodes[['x', 'y']])
        self.mynodes['cluster'] = clusters

    def plot_clusters(self):
        for i in range(self.kmeans.n_clusters):
            plt.scatter(self.mynodes[self.mynodes.cluster == i].x, self.mynodes[self.mynodes.cluster == i].y)
        plt.scatter(self.kmeans.cluster_centers_[:, 0], self.kmeans.cluster_centers_[:, 1], color='black')
        plt.savefig('results/clusters.pdf')

    def generate_topologies(self):
        topology_strategy_name = 'SimpleBackhaul'
        gw_strategy_name = 'BetweennessGW'
        base_dir = f'results/{self.dataset}_{(self.density_nodes):.0f}_{self.area_radius:.0f}_{self.area_center[0]:.0f}_{self.area_center[1]:.0f}/{time.time():.0f}/'
        os.makedirs(base_dir, exist_ok=True)
        for i in range(self.n_clusters):
            mydf = self.mynodes[self.mynodes.cluster == i]
            print(len(mydf))
            cluster_vg = self.graph.subgraph(mydf.id)
            TG_class = Topology.get_topology_generator(topology_strategy_name, gw_strategy_name)
            TG = TG_class(cluster_vg)
            TG.extract_graph()
            TG.save_graph(f'{base_dir}/{topology_strategy_name}_{gw_strategy_name}_{i}.graphml.gz')


def main():
    parser = configargparse.ArgumentParser(description='Generate reliable backhaul infrastructures', default_config_files=['sim.yaml'])
    parser.add_argument('--base_folder', help='path of the directory', default='../TrueNets_GPU/results/')
    parser.add_argument('-D', '--dataset', help='dataset')
    parser.add_argument('--cluster_size', type=int)
    parser.add_argument('--area_radius', type=float)
    parser.add_argument('--density_nodes', type=float)
    parser.add_argument('--area_center')
    parser.add_argument('--seed', help='random seed')
    args = parser.parse_args()
    s = Simulator(args)
    s.filter_nodes()
    s.clusterize()
    s.plot_clusters()
    s.generate_topologies()


if __name__ == '__main__':
    main()
