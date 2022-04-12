import networkx as nx
from misc import copy_graph, giant_component
import networkit as nk


class Gateway():
    def rank_gateways(self):
        self.sorted_nodes = {}
        # Abstract function

    def select_gateways(self, graph, n=1):
        self.rank_gateways(graph)
        sorted_gw = sorted(self.sorted_nodes.items(), key=lambda x: x[1], reverse=True)
        self.gateways = list(map(lambda x: x[0], sorted_gw[:n]))
        for gw in self.gateways:
            self.phig.nodes[gw]['type'] = 'gateway'


# class PageRankGW(Gateway):
#     def rank_gateways(self, graph):
#         self.sorted_nodes = nx.link_analysis.pagerank(graph)


# class BetweennessGW(Gateway):
#     def rank_gateways(self, graph):
#         self.sorted_nodes = nx.centrality.betweenness_centrality(graph)


class ClosenessGW(Gateway):
    def rank_gateways(self, graph):
        self.sorted_nodes = nx.centrality.closeness_centrality(graph)


class GroupClosenessCentrality(Gateway):
    def select_gateways(self, graph, n=1):
        nkg = nk.nxadapter.nx2nk(graph)
        cent = nk.centrality.GroupCloseness(nkg, k=self.n_clusters)
        cent.run()
        nkgws = cent.groupMaxCloseness()
        self.gateways = [list(graph.nodes())[nd] for nd in nkgws]
        for gw in self.gateways:
            self.phig.nodes[gw]['type'] = 'gateway'
