import networkx as nx

from misc import copy_graph, giant_component


def get_topology_generator(gw_strategy_name: str, topology_strategy_name: str):
    # Get class using string as classname
    gw_strategy = globals()[gw_strategy_name]
    topology_strategy = globals()[topology_strategy_name]

    class TopologyGenerator(gw_strategy, topology_strategy):
        def __init__(self, vg):
            self.vg = vg
            self.phig = nx.Graph()
            self.phig.add_nodes_from(self.vg.nodes(data=True))

    return TopologyGenerator


class Backhaul():
    def extract_graph(self, vg):
        raise NotImplementedError

    def save_graph(self, file):
        nx.write_graphml(self.phig, file)


class SimpleBackhaul(Backhaul):
    def extract_graph(self):
        self.select_gateways(1)
        p = nx.shortest_path(self.vg, self.gateways[0])
        for path in p.values():
            for i in range(len(path)-1):
                self.phig.add_edge(path[i], path[i+1])


class VisGraph(Backhaul):
    def extract_graph(self):
        self.phig.add_edges_from(giant_component(self.vg).edges())


class K2EdgeAugmentedST(SimpleBackhaul):
    def extract_graph(self):
        super().extract_graph()
        for src, dst in nx.k_edge_augmentation(self.phig,
                                               k=2,
                                               avail=giant_component(self.vg).edges(),
                                               partial=True):
            self.phig.add_edge(src, dst)


class MultiGWBackhaul(Backhaul):
    def extract_graph(self):
        self.select_gateways(3)
        for gw in self.gateways[:3]:
            p = nx.shortest_path(self.vg, gw)
            for path in p.values():
                for i in range(len(path)-1):
                    self.phig.add_edge(path[i], path[i+1])


class MultiGWBackhaulDisjoint(Backhaul):
    def extract_graph(self):
        self.select_gateways(3)
        myvg = copy_graph(self.vg)
        for gw in self.gateways[:3]:
            p = nx.shortest_path(myvg, gw)
            for path in p.values():
                for i in range(len(path)-1):
                    self.phig.add_edge(path[i], path[i+1])
                    try:
                        myvg.remove_edge(path[i], path[i+1])
                    except nx.exception.NetworkXError:
                        pass


class Gateway():
    def rank_gateways(self):
        self.sorted_nodes = {}
        # Abstract function

    def select_gateways(self, n=1):
        self.rank_gateways()
        sorted_gw = sorted(self.sorted_nodes.items(), key=lambda x: x[1], reverse=True)
        self.gateways = list(map(lambda x: x[0], sorted_gw[:n]))
        for gw in self.gateways:
            self.phig.nodes[gw]['type'] = 'gateway'


class PageRankGW(Gateway):
    def rank_gateways(self):
        self.sorted_nodes = nx.link_analysis.pagerank(self.vg)


class BetweennessGW(Gateway):
    def rank_gateways(self):
        self.sorted_nodes = nx.centrality.betweenness_centrality(self.vg)


class ClosenessGW(Gateway):
    def rank_gateways(self):
        self.sorted_nodes = nx.centrality.closeness_centrality(self.vg)
