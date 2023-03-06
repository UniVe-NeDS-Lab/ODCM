import networkx as nx
import osmnx as ox
from matplotlib import pyplot as plt
import pandas as pd
import itertools
import numpy as np

def copy_graph(g: nx.Graph) -> nx.Graph:
    copy = nx.Graph()
    copy.add_nodes_from(g.nodes(data=True))
    copy.add_edges_from(g.edges(data=True))
    return copy

def giant_component(g: nx.Graph) -> nx.Graph:
    return nx.subgraph(g, max(nx.connected_components(g), key=len))

def core_graph(g: nx.Graph) -> tuple[list, nx.Graph]:
    core=[]
    for n in g.nodes():
        if g.degree(n) > 1:
            core.append(n)
    return core, g.subgraph(core)

def semicore_graph(g: nx.Graph) -> tuple[list, nx.Graph]:
    core=[]
    for n in g.nodes():
        if g.degree(n) > 0:
            core.append(n)
    return core, g.subgraph(core)

class Topology():
    def __init__(self, vg, nodes, n_clusters):
        self.vg = vg
        self.phig = nx.Graph()
        self.nodes = nodes
        self.n_clusters = n_clusters
    
    def save_graph(self, file):
        nx.write_graphml(self.phig, f'{file}.wireless.graphml.gz')
        nx.write_graphml(self.T, f'{file}.fiber.graphml.gz')
        
    # def select_gateways_old(self, graph, n=1):
    #     sorted_nodes = nx.centrality.closeness_centrality(graph)
    #     sorted_gw = sorted(sorted_nodes.items(), key=lambda x: x[1], reverse=True)
    #     self.gateways = list(map(lambda x: x[0], sorted_gw[:n]))
    #     for gw in self.gateways:
    #         self.phig.nodes[gw]['type'] = 'gateway'

    def group_closeness_weighted_fast(self, graph, k):
        #Fastest version of the group closeness that uses floyd_warshall algorithm 
        # n^k+1
        distances = nx.floyd_warshall(graph,  weight='dist')
        subs = {n:graph.nodes[n]['subscriptions'] for n in graph.nodes()}
        pairs = itertools.combinations(graph.nodes(), k) 
        group_c = {}
        for p in pairs: #n^k
            #for each possible pair
            sums = 0
            for n,s in subs.items(): #for each node times k
                sums += min([distances[src][n] for src in p])*s
            if sums!=0:
                group_c[p] = 1/sums
            else:
                print("Found 0 sum paths")
                continue
        return(max(group_c.items(), key=lambda x:x[1]))


    def group_closeness_weighted(self, graph, k): #n^(k+2)
        #consider only the largest component and search the gws there
        graph = nx.subgraph(graph, max(nx.connected_components(graph), key=len))
        #TODO might be improved by using floyd_warshall algorithm to precompute all pair shortest path lengths
        subs = {n:graph.nodes[n]['subscriptions'] for n in graph.nodes()}
        pairs = itertools.combinations(graph.nodes(), k) #n^k
        group_c = {}
        for p in pairs:
            path_dists = nx.multi_source_dijkstra_path_length(graph, p, weight='dist') #n^2 ?
            try:
                #weighted version
                group_c[p] = 1/sum([s*path_dists[n] for n,s in subs.items()])
                #unweighted version
                #group_c[p] = 1/sum([path_dists[n] for n,s in subs.items()])
            except ZeroDivisionError:
                print("Found 0 sum paths")
                continue
        return(max(group_c.items(), key=lambda x:x[1]))


    def select_gateways(self, graph, n): 
        g, s = self.group_closeness_weighted_fast(graph, n)       
        for gw in g:
            self.phig.nodes[gw]['type'] = 'gateway'
        return g
    
    def plot_graph(self, graph, visgraph):
        def get_col(type, degree):
            if type=='bs' and degree == 1:
                return 'yellow'
            if type=='bs' and degree > 1:
                return 'black'
            elif type=='gateway':
                return 'red'
        
        def get_edge_attr(graph, src, tgt):
            #return color, alpha, width
                if graph.has_edge(src, tgt):
                    if graph[src][tgt].get('addedd') == True:
                        return 'red', 1, 3
                    return 'black', 1, 3
                else:
                    return 'green', 0.3, 0.5
        
        pos = {n:(d['x'], d['y']) for n,d in visgraph.nodes(data=True)}
        col = [get_col(graph.nodes[n]['type'], nx.degree(graph, n)) for n in visgraph.nodes()]
        node_sizes = [100*visgraph.nodes[n]['subscriptions'] for n in visgraph]
        print(node_sizes)
        nx.draw_networkx_nodes(visgraph, pos=pos, node_color=col, node_size=node_sizes)
        edge_col = [get_edge_attr(graph, src, tgt)[0] for src, tgt in visgraph.edges()]
        edge_alpha = [get_edge_attr(graph, src, tgt)[1] for src, tgt in visgraph.edges()]
        edge_width = [get_edge_attr(graph, src, tgt)[2] for src, tgt in visgraph.edges()]
        nx.draw_networkx_edges(visgraph, pos=pos, edge_color=edge_col, alpha=edge_alpha, width=edge_width)
        plt.show()

    def extract_graph(self, n_gws=1):
        for i in range(self.n_clusters):
            mydf = self.nodes[self.nodes.cluster == i]
            filt_vg = self.vg.subgraph(mydf.id)
            for n in filt_vg.nodes():
                self.phig.add_node(n, x=filt_vg.nodes[n]['x'], 
                                      id=filt_vg.nodes[n]['id'], 
                                      y=filt_vg.nodes[n]['y'], 
                                      subscriptions=filt_vg.nodes[n]['subscriptions'],
                                      type='bs')
            if len(mydf) == 0:
                continue
            conn_vg = filt_vg.subgraph(max(nx.connected_components(filt_vg), key=len))
            # if(int(n_gws)==2):
            #     import pdb; pdb.set_trace()
            graph = copy_graph(conn_vg)
            gws = self.select_gateways(graph, n_gws)
            paths = nx.multi_source_dijkstra_path(graph, sources=gws, weight='dist')
           
            gw_nodes = []
            for node, path in paths.items():
                gw_nodes.append({'node': node, 
                                 'gateway': path[0], 
                                 'degree': nx.degree(graph, node),
                                 'attacched_gw': node != path[0] and path[1] == node})
                for i in range(len(path)-1):
                    self.phig.add_edge(path[i], path[i+1], dist=filt_vg[path[i]][path[i+1]]['dist'])
            gw_nodes = pd.DataFrame(gw_nodes)
            semicore, gsemicore = semicore_graph(self.phig)
            pre = len(self.phig.edges())
            core, gcore = core_graph(self.phig)
            for src, dst in self.two_edge_augmentation(self.phig, conn_vg, gw_nodes, gws):
                self.phig.add_edge(src, dst, dist=self.vg[src][dst]['dist'])
            if n_gws==2:
                #print(f"Addedd {len(self.phig.edges())-pre} edges out of {len(conn_vg.subgraph(core).edges())}. Vg has {len(conn_vg)} nodes, Core of Vg has {len(conn_vg.subgraph(core))}")
                #self.plot_graph(self.phig, conn_vg)
                edges = self.find_best_connecting_edges(self.phig, conn_vg, gws, gw_nodes)
                if not edges:
                    self.plot_graph(self.phig, conn_vg)
                for e in edges:
                    self.phig.add_edge(e[0], e[1], addedd=True, dist=self.vg[e[0]][e[1]]['dist'])
                
            nx.set_node_attributes(self.phig, mydf.cluster.to_dict(), "cluster")
    
    def two_edge_augmentation(self, topology, vg, gw_nodes, gws):
        #performs 2-edge augmentation on each gateway component separately
        edges = []
        for gw in gws:
            nodes =  gw_nodes[(gw_nodes.gateway==gw)].node.values
            topo = nx.subgraph(topology, nodes)
            vis = nx.subgraph(vg, nodes)
            edges += self.core_edge_augmentation(topo, vis)
        return edges

    def core_edge_augmentation(self, topology, vg):
        #perform 2-edge augmentation on the core of the network
        core, gcore = core_graph(topology)
        return list(nx.k_edge_augmentation(gcore,
                                           k=2,
                                           avail=nx.subgraph(vg, core).edges(),
                                           weight='dist',
                                           partial=True))
         

    def find_best_connecting_edge_old(self, topology, visgraph, gws, gw_nodes):
        net_a = gw_nodes[(gw_nodes.gateway==gws[0]) & (gw_nodes.node!=gws[0])] 
        net_b = gw_nodes[(gw_nodes.gateway==gws[1]) & (gw_nodes.node!=gws[1])]
        
        #relays
        relay_a = net_a[net_a.degree > 1].node.values
        relay_b = net_b[net_b.degree > 1].node.values
        edges = list(nx.edge_boundary(visgraph, relay_a, relay_b, data=True))
        if edges:
            return min(edges, key=lambda x: x[2]['dist'])
        
        #ok leaves if not attacched to a gw
        relay_a = net_a[(net_a.attacched_gw == False)].node.values
        relay_b = net_b[(net_b.attacched_gw == False)].node.values
        edges = list(nx.edge_boundary(visgraph, relay_a, relay_b, data=True))
        if edges:
            return min(edges, key=lambda x: x[2]['dist'])
        
        #everything
        relay_a = gw_nodes[(gw_nodes.gateway==gws[0])].node.values
        relay_b = gw_nodes[(gw_nodes.gateway==gws[1])].node.values
        edges = list(nx.edge_boundary(visgraph, relay_a, relay_b, data=True))
        if edges:
            return min(edges, key=lambda x: x[2]['dist'])
    
    def find_best_connecting_edges(self, topology, visgraph, gws, gw_nodes):
        net_a = gw_nodes[(gw_nodes.gateway==gws[0]) & (gw_nodes.node!=gws[0])] 
        net_b = gw_nodes[(gw_nodes.gateway==gws[1]) & (gw_nodes.node!=gws[1])]
        #relays
        relay_a = net_a.node.values
        relay_b = net_b.node.values
        edges = list(nx.edge_boundary(visgraph, relay_a, relay_b, data=True))
        if not edges:
            net_a = gw_nodes[(gw_nodes.gateway==gws[0])] 
            net_b = gw_nodes[(gw_nodes.gateway==gws[1])]
            #relays
            relay_a = net_a.node.values
            relay_b = net_b.node.values
            edges = list(nx.edge_boundary(visgraph, relay_a, relay_b, data=True))
            if not edges:
                import pdb; pdb.set_trace()
                return []
        if len(edges) <=2:
            return edges
        magic_g = nx.subgraph(self.phig, gw_nodes.node.values).copy()
        e_idx=0
        magic_nodes = {}
        for src,tgt, d in edges:
            magic_node_id = f'e_{e_idx}'
            magic_nodes[magic_node_id] = (src, tgt, d)
            magic_g.add_edge(src, magic_node_id, dist=d['dist']/2)
            magic_g.add_edge(magic_node_id, tgt, dist=d['dist']/2)
            e_idx+=1
        
        #compute group closeness centrality
        distances = nx.floyd_warshall(magic_g,  weight='dist')
        pairs = itertools.combinations(magic_nodes.keys(), 2) 
        group_c = {}
        for p in pairs:
            sums = 0
            for n in gw_nodes.node.values:
                sums += min([distances[src][n] for src in p])
            if sums!=0:
                group_c[p] = 1/sums
            else:
                print("Found 0 sum paths")
                continue
        try:
            best_pair = max(group_c.items(), key=lambda x:x[1])
        except:
            import pdb; pdb.set_trace()
        return [magic_nodes[e] for e in best_pair[0]]
        


    def fiber_backhaul(self, road_graph, fiber_pop):
        gws = [n for n in self.phig if self.phig.nodes[n].get('type') == 'gateway']
        osm_gw = [ox.nearest_nodes(road_graph, self.vg.nodes()[gw]['x'], self.vg.nodes()[gw]['y']) for gw in gws]
        osm_gw.append(ox.nearest_nodes(road_graph, fiber_pop['x'], fiber_pop['y']))
        T = nx.algorithms.approximation.steiner_tree(road_graph, terminal_nodes=osm_gw, weight='length', method='kou')
        self.T = nx.Graph()
        for n,d in T.nodes(data=True):
            self.T.add_node(n, x=d['x'], y=d['y'])
        for s,t,d in T.edges(data=True):
            self.T.add_edge(s,t, length=d['length'])
        

