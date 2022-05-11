import networkx as nx
from misc import copy_graph
import pdb
import osmnx as ox


class Topology():
    def __init__(self, vg, nodes, n_clusters):
        self.vg = vg
        self.phig = nx.Graph()
        self.nodes = nodes
        self.n_clusters = n_clusters
    
    def save_graph(self, file):
        nx.write_graphml(self.phig, f'{file}.wireless.graphml.gz')
        nx.write_graphml(self.T, f'{file}.fiber.graphml.gz')
        
    def select_gateways(self, graph, n=1):
        sorted_nodes = nx.centrality.closeness_centrality(graph)
        sorted_gw = sorted(sorted_nodes.items(), key=lambda x: x[1], reverse=True)
        self.gateways = list(map(lambda x: x[0], sorted_gw[:n]))
        for gw in self.gateways:
            self.phig.nodes[gw]['type'] = 'gateway'

    def extract_graph(self):
        for i in range(self.n_clusters):
            mydf = self.nodes[self.nodes.cluster == i]
            filt_vg = self.vg.subgraph(mydf.id)
            for n in filt_vg.nodes():
                self.phig.add_node(n, x=filt_vg.nodes[n]['x'], 
                                      id=filt_vg.nodes[n]['id'], 
                                      y=filt_vg.nodes[n]['y'], 
                                      subscriptions=filt_vg.nodes[n]['subscriptions'] )
            if len(mydf) == 0:
                continue
            conn_vg = filt_vg.subgraph(max(nx.connected_components(filt_vg), key=len))
            graph = copy_graph(conn_vg)
            self.select_gateways(graph, 1)
            paths = nx.multi_source_dijkstra_path(graph, sources=self.gateways, weight='dist')
            for node, path in paths.items():
                for i in range(len(path)-1):
                    self.phig.add_edge(path[i], path[i+1], dist=filt_vg[path[i]][path[i+1]]['dist'])
    
    def fiber_backhaul(self, road_graph, fiber_pop):
        gws = [n for n in self.phig if self.phig.nodes[n].get('type') == 'gateway']
        osm_gw = [ox.nearest_nodes(road_graph, self.vg.nodes()[gw]['x'], self.vg.nodes()[gw]['y']) for gw in gws]
        osm_gw.append(ox.nearest_nodes(road_graph, fiber_pop['x'], fiber_pop['y']))
        T = nx.algorithms.approximation.steiner_tree(road_graph, terminal_nodes=osm_gw, weight='length')
        self.T = nx.Graph()
        for n,d in T.nodes(data=True):
            self.T.add_node(n, x=d['x'], y=d['y'])
        for s,t,d in T.edges(data=True):
            self.T.add_edge(s,t, length=d['length'])
        

