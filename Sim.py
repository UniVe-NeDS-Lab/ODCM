# MIT License

# Copyright (c) [2022] [Gabriele Gemmi gabriele.gemmi@unive.it]

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import configargparse
import networkx as nx
import numpy as np
import pandas as pd
from Topology import Topology
import geopandas as gpd
from diskcache import Cache
import math as m
import os
import time
import random
import metis
import tqdm
from collections import Counter
import osmnx as ox
import json
ox.settings.log_console=False
ox.settings.use_cache=True
#ox.config(use_cache=True, log_console=False)
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

    def generate_topologies(self, t):
        base_dir = f'results/{self.dataset}_{(self.subscribers_ratio*100):.0f}_{self.cluster_size}_{t}/'
        algo, n_gws = t.split('_')
        os.makedirs(base_dir, exist_ok=True)
        TG = Topology(self.graph, self.mynodes, self.n_clusters)
        TG.extract_graph(n_gws=int(n_gws), algo=algo)
        TG.fiber_backhaul(self.osm_road, self.fiber_pop)
        TG.save_graph(f'{base_dir}/{time.time()*10:.0f}_{self.random_seed}')


def main():
    parser = configargparse.ArgumentParser(description='Generate reliable backhaul infrastructures', default_config_files=['sim.yaml'])
    parser.add_argument('-D', '--dataset', help='dataset')
    parser.add_argument('--cluster_size', type=int, action='append')
    parser.add_argument('--subscribers_ratio', type=float, action='append')
    parser.add_argument('--types', type=str, action='append')
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
                for t in args.types:
                    s.generate_topologies(t)
                pbar.update(1)
    pbar.close()
            
        


if __name__ == '__main__':
    main()
