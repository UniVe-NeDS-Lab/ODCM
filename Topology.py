import networkx as nx
import osmnx as ox

def copy_graph(g: nx.Graph) -> nx.Graph:
    copy = nx.Graph()
    copy.add_nodes_from(g.nodes(data=True))
    copy.add_edges_from(g.edges(data=True))
    return copy

class Topology():
    def __init__(self, vg, nodes, n_clusters):
        self.vg = vg
        self.phig = nx.Graph()
        self.nodes = nodes
        self.n_clusters = n_clusters
    
    def save_graph(self, file):
        nx.write_graphml(self.phig, f'{file}.wireless.graphml.gz')
        nx.write_graphml(self.T, f'{file}.fiber.graphml.gz')
        
    def select_gateways_old(self, graph, n=1):
        sorted_nodes = nx.centrality.closeness_centrality(graph)
        sorted_gw = sorted(sorted_nodes.items(), key=lambda x: x[1], reverse=True)
        self.gateways = list(map(lambda x: x[0], sorted_gw[:n]))
        for gw in self.gateways:
            self.phig.nodes[gw]['type'] = 'gateway'

    def group_closeness_weighted(self, graph, k): #n^(k+2)
        import itertools
        #TODO might be improved by using floyd_warshall algorithm to precompute all pair shortest path lengths
        subs = {n:graph.nodes[n]['subscriptions'] for n in graph.nodes()}
        pairs = itertools.combinations(graph.nodes(), k) #n^k
        group_c = {}
        for p in pairs:
            path_dists = nx.multi_source_dijkstra_path_length(graph, p, weight='dist') #n^2 ?
            try:
                #weighted version
                #group_c[p] = 1/sum([s*path_dists[n] for n,s in subs.items()])
                
                #unweighted version
                group_c[p] = 1/sum([path_dists[n] for n,s in subs.items()])
            except ZeroDivisionError:
                print("Found 0 sum paths")
                pass
        return(max(group_c.items(), key=lambda x:x[1]))


    def select_gateways(self, graph, n): 
        g, s = self.group_closeness_weighted(graph, n)       
        self.gateways=g
        for gw in self.gateways:
            self.phig.nodes[gw]['type'] = 'gateway'

    def extract_graph(self, n_gws=1):
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
            self.select_gateways(graph, n_gws)
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
        

