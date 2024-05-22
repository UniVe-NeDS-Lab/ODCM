import numpy as np
import networkx as nx
import pandas as pd
import seaborn as sns
import re
import matplotlib.pyplot as plt
from utils import *
from tqdm.contrib.concurrent import process_map  # or thread_map

class CapacityAnalysis():
    def __init__(self, params):
        self.p = params

    def calc_ad_speed(self, dist, freq=70, ptp=False):
        pl = 20*np.log10(dist/1000) + 20*np.log10(freq) + 92.45
        if ptp:
            pr = self.p.pt + self.p.gr_d - pl
        else:
            pr = self.p.pt + self.p.gr_m - pl
        speed = 0
        for i in range(len(self.p.speed_table)):
            if pr<self.p.speed_table[i][0]:
                speed = self.p.speed_table[i][2]
        return speed

    def calc_maxbw(self, g, p):
        bottleneck = self.p.speed_table[0][2] #initalize to maximum value (400mbps)
        for i in range(len(p)-1):
            speed = self.calc_ad_speed(g[p[i]][p[i+1]]['dist'])
            if speed<bottleneck:
                bottleneck=speed
        return bottleneck

    def calc_minbw_gab(self, g, path):
        #print("nodes", path)
        uplinks = []
        downlinks = []
        #print("node ul_paths dl_paths ul_capacity dl_capacity # pred #succ")
        for p_i in range(len(path)-1):
            p0 = path[p_i]
            p1 = path[p_i+1]
            #print(p0, p1,  g[p0][p1]['paths'], g.nodes[p0]['paths'], g.nodes[p0]['uplink_capacity'], g.nodes[p0]['downlink_capacity'], len(list(g.predecessors(p0))), len(list(g.successors(p0))))
            #print(p1, g.nodes[p1]['paths'], g.nodes[p1]['uplink_capacity'], g.nodes[p1]['downlink_capacity'], len(list(g.predecessors(p1))), len(list(g.successors(p1))))
            
            downlinks.append(g.nodes[p1]['downlink_capacity']/g.nodes[p1]['paths'])
            uplinks.append(g.nodes[p0]['uplink_capacity']/g[p0][p1]['paths'])
        # print("paths", [g.nodes[p]['paths'] for p in path])
        # print("ants", [g.nodes[p]['n_ant'] for p in path])
        # print("ul", [g.nodes[p]['uplink_capacity'] for p in path])

        # print("dl", [g.nodes[p]['downlink_capacity'] for p in path])
        uplink = min(uplinks)
        downlink = min(downlinks)
        return uplink, downlink, min(uplink, downlink)


    def get_network_capacities(self, graphs):
        data = []
        #Need to chunk to avoid memory issues. It seems that the GC doesn't run untill the pool is closed
        for idx, c in enumerate(list(chunks(graphs, 500))):
            print(f"chunk {idx}")
            data = data + process_map(self.get_network_capacity, c, max_workers=8, chunksize=10)
        self.bwdf = pd.DataFrame(flatten(data))
        print("Finished processing capacity")

    def get_network_capacity(self, graph):
        data = []
        mac_efficiency = 0.84
        (p, mgb, w_g,f_g, n_subs)  = graph
        area, ratio, cluster_size, algo, n_gw, time, random_seed = p
        gws = [n for n in w_g if 'type' in w_g.nodes[n] and w_g.nodes[n]['type'] == 'gateway']
        paths = nx.multi_source_dijkstra_path(w_g, gws, weight='dist')
        T = nx.DiGraph()
        T.add_nodes_from((i, w_g.nodes[i]) for i in w_g.nodes)
        for src, p in paths.items():
            p.reverse()
            for i in range(len(p)-1):
                T.add_edge(p[i], p[i+1], **w_g[p[i]][p[i+1]])
        
        for n in T.nodes():
            if T not in gws:
                caps_uplink = []
                caps_downlink = []
                T.nodes[n]['bottleneck_uplink'] = np.nan
                T.nodes[n]['bottleneck_downlink'] = np.nan
                T.nodes[n]['downlink_capacity'] = np.nan
                T.nodes[n]['uplink_capacity'] = np.nan
                n_paths = 0
                for neigh in T.predecessors(n):
                    caps_downlink.append(self.calc_ad_speed(T[neigh][n]['dist']))
                    n_paths += T[neigh][n]['paths']
                for neigh in T.successors(n):
                    caps_uplink.append(self.calc_ad_speed(w_g[n][neigh]['dist']))
                assert(len(caps_uplink)<=1)
                if caps_uplink:
                    T.nodes[n]['uplink_capacity'] = caps_uplink[0]
                    #T.nodes[n]['bottleneck_uplink'] = T.nodes[n]['uplink_capacity'] / n_paths
                
                if caps_downlink:
                    T.nodes[n]['downlink_capacity'] = np.mean(caps_downlink)*(T.nodes[n]['n_ant']-1)
                    #T.nodes[n]['bottleneck_downlink'] = T.nodes[n]['downlink_capacity'] / n_paths
                
                
                T.nodes[n]['paths'] = n_paths

        for src, p in paths.items():
            if src in gws:
                continue
            min_bw = self.calc_minbw_gab(T, p)
            
            max_bw = self.calc_maxbw(T, p)
            for s in range(w_g.nodes(data=True)[src]['subscriptions']):
                run = {}
                run['area'] = area
                run['algo'] = algo
                run['cluster_size'] = cluster_size
                run['ratio'] =  ratio
                run['gateways'] = len([n for n,att in w_g.nodes(data=True) if att.get('type')=='gateway'])
                run['n_gw'] = n_gw
                run['node'] = src
                run['mgb'] = mgb
                #run['min_bw'] = mac_efficiency*min_bw
                data.append(run | {'bw_type': 'min_ul', 'bw': mac_efficiency*min_bw[0]})
                data.append(run | {'bw_type': 'min_dl', 'bw': mac_efficiency*min_bw[1]})
                data.append(run | {'bw_type': 'min', 'bw': mac_efficiency*min_bw[2]})
                data.append(run | {'bw_type': 'max', 'bw': mac_efficiency*max_bw})
                # run['max_bw'] = mac_efficiency*max_bw
                # run['min_bw_ul'] = mac_efficiency*min_bw[0]
                # run['min_bw_dl'] = mac_efficiency*min_bw[1]
                # run['min_bw'] = mac_efficiency*min_bw[2]
                #data.append(run)
        return data

    
    def plot_capacity(self, figfolder):
        for m in self.bwdf.mgb.unique():
            fg = sns.FacetGrid(data=self.bwdf[self.bwdf.mgb == m], col='bw_type', sharey=False)
            fg.map_dataframe(sns.barplot, x='cluster_size', y='bw', hue='ratio')
            fg.add_legend(title="Subscriber Ratio")
            plt.savefig(f'{figfolder}/capacity_{m}.pdf')
    
    def save_capacity_csv(self, csvfolder):
        for m in self.bwdf.mgb.unique():
            bg = self.bwdf[self.bwdf.mgb == m].groupby(['bw_type', 'cluster_size','ratio' ]).bw.mean()
            bg.to_csv(f'{csvfolder}/bw_{m}.csv')


