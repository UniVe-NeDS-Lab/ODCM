import networkx as nx
from Gateway import *
from misc import copy_graph, giant_component
import pdb


def get_topology_generator(gw_strategy_name: str, topology_strategy_name: str, n_clusters: int):
    # Get class using string as classname
    gw_strategy = globals()[gw_strategy_name]
    topology_strategy = globals()[topology_strategy_name]

    class TopologyGenerator(gw_strategy, topology_strategy):
        def __init__(self, vg, nodes):
            self.vg = vg
            self.phig = nx.Graph()
            self.nodes = nodes
            self.n_clusters = n_clusters

    return TopologyGenerator


class Backhaul():
    def extract_graph(self):
        raise NotImplementedError

    def save_graph(self, file):
        nx.write_graphml(self.phig, file)


class ClusteredBackhaul(Backhaul):
    def extract_cluster(self, graph):
        self.phig.add_nodes_from(graph.nodes(data=True))
        return

    def extract_graph(self):
        for i in range(self.n_clusters):
            mydf = self.nodes[self.nodes.cluster == i]
            cluster_vg = copy_graph(self.vg.subgraph(mydf.id))
            self.extract_cluster(cluster_vg)


class MSD(ClusteredBackhaul):
    def extract_cluster(self, graph):
        super().extract_cluster(graph)
        self.select_gateways(graph, 1)
        paths = nx.multi_source_dijkstra_path(graph, sources=self.gateways)
        for node, path in paths.items():
            for i in range(len(path)-1):
                self.phig.add_edge(path[i], path[i+1])


# class SP(ClusteredBackhaul):
#     def extract_cluster(self, graph):
#         super().extract_cluster(graph)
#         self.select_gateways(graph, 1)
#         p = nx.shortest_path(graph, self.gateways[0])
#         for path in p.values():
#             for i in range(len(path)-1):
#                 self.phig.add_edge(path[i], path[i+1])


# class K2EdgeAugmentedST(SP):
#     def extract_cluster(self, graph):
#         super().extract_cluster(graph)
#         for src, dst in nx.k_edge_augmentation(self.phig,
#                                                k=2,
#                                                avail=giant_component(graph).edges(),
#                                                partial=True):

#             self.phig.add_edge(src, dst)


# class MultiGWDisjointST(ClusteredBackhaul):
#     def extract_cluster(self, graph):
#         super().extract_cluster(graph)
#         self.select_gateways(graph, 3)
#         myvg = copy_graph(graph)
#         for gw in self.gateways[:3]:
#             T = nx.minimum_spanning_tree(myvg)
#             self.phig.add_edges_from(T.edges(data=True))
#             try:
#                 myvg.remove_edges_from(T.edges(data=True))
#             except nx.exception.NetworkXError:
#                 pass


# class MultiGWSP(ClusteredBackhaul):
#     def extract_cluster(self, graph):
#         super().extract_cluster(graph)
#         self.select_gateways(graph, 3)
#         for gw in self.gateways[:3]:
#             p = nx.shortest_path(graph, gw)
#             for path in p.values():
#                 for i in range(len(path)-1):
#                     self.phig.add_edge(path[i], path[i+1])


# class MultiGWDisjointSP(ClusteredBackhaul):
#     def extract_cluster(self, graph):
#         super().extract_cluster(graph)
#         self.select_gateways(graph, 3)
#         myvg = copy_graph(graph)
#         for gw in self.gateways[:3]:
#             p = nx.shortest_path(myvg, gw)
#             for path in p.values():
#                 for i in range(len(path)-1):
#                     self.phig.add_edge(path[i], path[i+1])
#                     try:
#                         myvg.remove_edge(path[i], path[i+1])
#                     except nx.exception.NetworkXError:
#                         pass


class MultiGwCluster(Backhaul):
    def extract_graph(self):
        thisvg = copy_graph(self.vg.subgraph(self.nodes.id))
        self.phig.add_nodes_from(thisvg.nodes(data=True))
        gateways = {}

        for i in range(self.n_clusters):
            mydf = self.nodes[self.nodes.cluster == i]
            cluster_vg = copy_graph(thisvg.subgraph(mydf.id))
            self.select_gateways(cluster_vg, 1)
            gateways[i] = self.gateways[0]

        # get best gw for each cluster
        all_paths = {}
        for i in range(self.n_clusters):
            paths = nx.shortest_path(thisvg, gateways[i])
            for p in paths:
                if p not in all_paths:
                    all_paths[p] = [paths[p]]
                else:
                    all_paths[p].append(paths[p])

        for node in all_paths.keys():
            choosen_paths = sorted(all_paths[node], key=len)[:3]
            for path in choosen_paths:
                for i in range(len(path)-1):
                    self.phig.add_edge(path[i], path[i+1])


class MSDijkstra(Backhaul):
    def extract_graph(self):
        thisvg = copy_graph(self.vg.subgraph(self.nodes.id))
        self.phig.add_nodes_from(thisvg.nodes(data=True))
        self.select_gateways(thisvg)
        paths = nx.multi_source_dijkstra_path(thisvg, sources=self.gateways)
        for node, path in paths.items():
            for i in range(len(path)-1):
                self.phig.add_edge(path[i], path[i+1])

# class SPUnion(Backhaul):
# Same as Multipath Dijkstra
#     def extract_graph(self):
#         thisvg = copy_graph(self.vg.subgraph(self.nodes.id))
#         self.phig.add_nodes_from(thisvg.nodes(data=True))
#         self.select_gateways(thisvg)
#         for n in thisvg.nodes():
#             #Find shortest path towards one of the gws\
#             paths = []
#             for gw in self.gateways:
#                 try:
#                     paths.append(nx.shortest_path(thisvg, n, gw))
#                 except nx.exception.NetworkXNoPath:
#                    pass
#             if paths:
#                 path = min(paths, key=len)
#                 for i in range(len(path)-1):
#                     self.phig.add_edge(path[i], path[i+1])  
            
            
                