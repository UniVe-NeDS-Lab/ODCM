import networkx as nx
from Gateway import *
from misc import copy_graph, giant_component
from heapq import heappop, heappush
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

def primdijkstra(G, root, alpha=1.0, weight='weight'):
    #Implement the MST-SPT tree with alpha, mentioned in the MENTOR paper
    #alpha = 1 yields the SPT, alpha=0 yields the MST
    if root not in G.nodes():
        raise nx.NetworkXError
    T = nx.Graph()
    T.add_node(root) #Initalize Tree by adding the root
    costs = {}
    costs[root] = 0  
    while(len(T)< len(G)):
        edges = []  #Frontier 
        for n in T.nodes():
            for e in G.edges(n, data=True):  
                if e[1] not in T: #For each edges incident to the current Tree T
                    dist = alpha*costs[e[0]] + e[2].get(weight, 1)  #Compute label and push to heap
                    heappush(edges, (dist, e))
        d, e = heappop(edges) #Get minimum
        T.add_edges_from([e])  #Add to the Tree
        costs[e[1]] = costs[e[0]] + e[2].get(weight, 1)  #Push the updated weight of the tree frontier
    return T


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
            filt_vg = self.vg.subgraph(mydf.id)
            conn_vg = filt_vg.subgraph(max(nx.connected_components(filt_vg), key=len))
            cluster_vg = copy_graph(conn_vg)
            self.extract_cluster(cluster_vg)
            

class SPT(ClusteredBackhaul):
    def extract_cluster(self, graph):
        super().extract_cluster(graph)
        self.select_gateways(graph, 1)
        paths = nx.multi_source_dijkstra_path(graph, sources=self.gateways, weight='dist')
        for node, path in paths.items():
            for i in range(len(path)-1):
                self.phig.add_edge(path[i], path[i+1])


class MST(ClusteredBackhaul):
    def extract_cluster(self, graph):
        super().extract_cluster(graph)
        self.select_gateways(graph, 1)
        T = nx.minimum_spanning_tree(graph)
        for e in T.edges(data=True):
            self.phig.add_edge(e[0], e[1])

class PDT0(ClusteredBackhaul):
    def extract_cluster(self, graph):
        super().extract_cluster(graph)
        self.select_gateways(graph, 1)
        T = primdijkstra(graph, root=self.gateways[0], alpha=0.5, weight='dist')
        for e in T.edges(data=True):
            self.phig.add_edge(e[0], e[1])

class PDT1(ClusteredBackhaul):
    def extract_cluster(self, graph):
        super().extract_cluster(graph)
        self.select_gateways(graph, 1)
        T = primdijkstra(graph, root=self.gateways[0], alpha=1, weight='dist')
        for e in T.edges(data=True):
            self.phig.add_edge(e[0], e[1])

class PDT(ClusteredBackhaul):
    def extract_cluster(self, graph):
        super().extract_cluster(graph)
        self.select_gateways(graph, 1)
        T = primdijkstra(graph, root=self.gateways[0], alpha=1.5, weight='dist')
        for e in T.edges(data=True):
            self.phig.add_edge(e[0], e[1])

class PDT2(ClusteredBackhaul):
    def extract_cluster(self, graph):
        super().extract_cluster(graph)
        self.select_gateways(graph, 1)
        T = primdijkstra(graph, root=self.gateways[0], alpha=2, weight='dist')
        for e in T.edges(data=True):
            self.phig.add_edge(e[0], e[1])
        

class CoreAugmentedMST(MST):
    def extract_cluster(self, graph):
        super().extract_cluster(graph)
        relays = [n for n in self.phig if  self.phig.degree[n] > 1]
        core = nx.subgraph(graph, relays)
        for src, dst in nx.k_edge_augmentation(self.phig,
                                               k=2,
                                               avail=giant_component(core).edges(),
                                               partial=True):
            self.phig.add_edge(src, dst)
        
class CoreAugmentedSPT(SPT):
    def extract_cluster(self, graph):
        super().extract_cluster(graph)
        relays = [n for n in self.phig if  self.phig.degree[n] > 1]
        core = nx.subgraph(graph, relays)
        
        for src, dst in nx.k_edge_augmentation(self.phig,
                                               k=2,
                                               avail=giant_component(core).edges(),
                                               partial=True):
            self.phig.add_edge(src, dst)

class AugmentedMSPT(SPT):
    def extract_cluster(self, graph):
        super().extract_cluster(graph)
        for src, dst in nx.k_edge_augmentation(self.phig,
                                               k=2,
                                               avail=giant_component(graph).edges(),
                                               partial=True):
            self.phig.add_edge(src, dst)

class AugmentedMST(MST):
    def extract_cluster(self, graph):
        super().extract_cluster(graph)
        for src, dst in nx.k_edge_augmentation(self.phig,
                                               k=2,
                                               avail=giant_component(graph).edges(),
                                               partial=True):
            self.phig.add_edge(src, dst)


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


# class MultiGwCluster(Backhaul):
#     def extract_graph(self):
#         thisvg = copy_graph(self.vg.subgraph(self.nodes.id))
#         self.phig.add_nodes_from(thisvg.nodes(data=True))
#         gateways = {}
#         all_paths = {}
#         #Connect each cluster to its gw
#         for i in range(self.n_clusters):
#             mydf = self.nodes[self.nodes.cluster == i]
#             cluster_vg = copy_graph(thisvg.subgraph(mydf.id))
#             self.select_gateways(cluster_vg, 1)
#             gateways[i] = self.gateways[0]
#             paths = nx.shortest_path(thisvg, gateways[i])
#             for p in paths:
#                 if p not in all_paths:
#                     all_paths[p] = [paths[p]]
#                 else:
#                     all_paths[p].append(paths[p])

#         for node in all_paths.keys():
#             choosen_paths = sorted(all_paths[node], key=len)[:2]
#             for path in choosen_paths:
#                 for i in range(len(path)-1):
#                     self.phig.add_edge(path[i], path[i+1])


# class MShTUnclustered(Backhaul):
#     def extract_graph(self):
#         thisvg = copy_graph(self.vg.subgraph(self.nodes.id))
#         self.phig.add_nodes_from(thisvg.nodes(data=True))
        
#         gateways = {}
#         for i in range(self.n_clusters):
#             mydf = self.nodes[self.nodes.cluster == i]
#             cluster_vg = thisvg.subgraph(mydf.id)
#             self.select_gateways(cluster_vg, 1)
#             gateways[i] = self.gateways[0]
            

#         paths = nx.multi_source_dijkstra_path(thisvg, sources=gateways.values())
#         for node, path in paths.items():
#             for i in range(len(path)-1):
#                 self.phig.add_edge(path[i], path[i+1])

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
            
            
                