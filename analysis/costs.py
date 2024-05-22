import math as m
import networkx as nx
import numpy as np
import pandas as pd
from tqdm.contrib.concurrent import process_map  # or thread_map
from tqdm import tqdm


from utils import *

class CostsAnalysis:
    def __init__(self, params):
        self.p = params
        

    def calc_cost_fiber(self, g):
        return sum([d['length'] for s,t,d in g.edges(data=True)])*1e-3*self.p.capex_costs['fiber_deploy']

    def phi(self, g, n):
        #Find the size of the occupied beam and see how many antennas are needed to cover it.
        thisnode = g.nodes[n]
        beams = []
        for neigh in g[n]:
            thisneigh = g.nodes[neigh]
            beam = m.degrees(m.atan2(thisneigh['x'] - thisnode['x'], thisneigh['y']-thisnode['y'])) % 360
            beams.append(beam)
        beams.sort()
        try:
            beams.append(beams[0])
        except:
            print(g[n])
        beams = np.array(beams)
        
        diff = (beams[1:] - beams[:-1] -0.0001) % 360
        width = 360 - diff.max()    
        return width

    def calc_antennas(self, g, n, gw, mgb):
        phi_v = m.ceil(self.phi(g, n)/self.p.mpant_bw)
        n_paths = sum([n['paths'] for n in g[n].values()])
        if not gw:
            #Relay nodes needs double of the BW
            k=1 #TODO: check this
        else: 
            k=1
        d_v = m.ceil(n_paths*k*mgb/self.p.max_bx)
        return max(phi_v, d_v)

    def compute_paths(self, g, gws, mgb):
        for e in g.edges():
            g.edges[e]['paths'] = 0
        
        ## For each edge where a SP pass accumulate a value
        
        paths = nx.multi_source_dijkstra_path(g, gws, weight='dist')
        for k,path in paths.items():
            n_subs = g.nodes(data=True)[k]['subscriptions']
            for i in range(len(path)-1):
                g.edges[path[i], path[i+1]]['paths'] += n_subs
        for e in g.edges():
            if(g.edges[e]['paths'] == 0):
                ## For redundancy link assume a single path
                g.edges[e]['paths'] = 1
                g.edges[e]['redundant'] = 1
            else:
                g.edges[e]['redundant'] = 0
            g.edges[e]['bw'] = mgb * g.edges[e]['paths']
        

    def calc_cost_wireless(self, g, mgb) -> list[float]:
        gws = [n for n in g if 'type' in g.nodes[n] and g.nodes[n]['type'] == 'gateway']
        self.compute_paths(g, gws, mgb)
                    
        ## Multiplicate the value by the BW needed for each node
        ## Calculate the cost of each edge
        
        for n in g.nodes():
            #If it's unconnected it does not cost
            if nx.degree(g)[n] == 0:
                g.nodes[n]['router_cost'] = 0
                g.nodes[n]['deploy'] = 0
                g.nodes[n]['fiber_cost'] = 0
                g.nodes[n]['radio_cost'] = 0
                continue
            #If it's a GW
            if g.nodes[n].get('type') == 'gateway':
                g.nodes[n]['router_cost'] = self.p.capex_costs['gateway_router'] 
                #loc = (g.nodes[n]['x'], g.nodes[n]['y'])
                #g.nodes[n]['fiber_cost' ] = self.p.capex_costs['fiber_deploy']*road_distance(fiber_points[area], loc)
                g.nodes[n]['deploy'] = self.p.capex_costs['gateway_deploy']
                g.nodes[n]['n_ant'] =  self.calc_antennas(g, n, True, mgb)
                #print(g.nodes[n]['n_ant'])
                g.nodes[n]['radio_cost'] = self.p.capex_costs['mp_radio'] * g.nodes[n]['n_ant']
            else:
                g.nodes[n]['fiber_cost' ] = 0
                if g.degree()[n] == 1:
                    #leaf node, no router
                    g.nodes[n]['radio_cost'] = self.p.capex_costs['leaf_radio']
                    g.nodes[n]['deploy'] = self.p.capex_costs['leaf_deploy']
                    g.nodes[n]['router_cost'] = 0
                    g.nodes[n]['n_ant'] = 1
                else:
                    #relay node
                    g.nodes[n]['n_ant'] = self.calc_antennas(g, n, False, mgb) 
                    #Relays are leaves with additional antennas
                    g.nodes[n]['radio_cost'] = self.p.capex_costs['mp_radio'] * g.nodes[n]['n_ant'] + self.p.capex_costs['leaf_radio']
                    g.nodes[n]['deploy'] = self.p.capex_costs['relay_deploy']
                    g.nodes[n]['router_cost'] = self.p.capex_costs['relay_router']
            
            
        router_cost = sum([g.nodes[n]['router_cost'] for n in g.nodes()])
        deploy = sum([g.nodes[n]['deploy'] for n in g.nodes()])
        radio_cost = sum([g.nodes[n]['radio_cost'] for n in g.nodes()])

        return [router_cost, deploy, radio_cost]
    



    def calc_opex_fiber_network(self, g, mgb):
        #Fiber opex
        total_bw = sum([d['subscriptions'] for n,d in g.nodes(data=True)])*mgb/1000 #Gbps of mgb
        fiber_transit = m.ceil(total_bw)*self.p.opex_costs['bw']
        if total_bw < 10:
            fiber_trasport = self.p.opex_costs['transport_10']
        # if total_bw < 20:
        #     fiber_trasport = 2*self.p.opex_costs['transport_10']
        elif total_bw < 100:
            fiber_trasport = self.p.opex_costs['transport_100']
        elif total_bw < 200:
            fiber_trasport = 2*self.p.opex_costs['transport_100']
        elif total_bw < 300:
            fiber_trasport = 3*self.p.opex_costs['transport_100']
        elif total_bw < 400:
            fiber_trasport = 4*self.p.opex_costs['transport_100']
        else:
            raise ValueError(f"Can't relay more than 200G : {total_bw}")
        return fiber_transit, fiber_trasport

    def calc_opex_maintenance(self, g, kind):
        #Maintenance opex
        relays = [n for n in g if  g.degree[n] > 1]
        leafs = [n for n in g if  g.degree[n] == 1]
        gws = [n for n in g if 'type' in g.nodes[n] and g.nodes[n]['type'] == 'gateway']
        p_leaf = len(leafs)/len(g)
        p_relay = len(relays)/len(g)
        p_radio_failure = (364*24)/self.p.mttf['radio']
        p_router_failure = (364*24)/self.p.mttf['router']

        n_tot_ants = sum([g.nodes[n]['n_ant'] for n in relays])
            
        gw_maintenance = p_router_failure * len(gws) * (self.p.mttr['router'] * self.p.opex_costs[kind] + self.p.capex_costs['gateway_router'])
        router_maintenance = p_router_failure  * len(relays) * (self.p.mttr['router'] * self.p.opex_costs[kind] + self.p.capex_costs['relay_router'])
        leafs_maintenance = p_leaf * p_radio_failure * len(leafs) * (self.p.mttr['radio'] * self.p.opex_costs['planned_maintenance'] + self.p.capex_costs['leaf_radio'])
        relays_maintenance = p_relay * p_radio_failure * n_tot_ants * (self.p.mttr['radio'] * self.p.opex_costs[kind] + self.p.capex_costs['mp_radio'])
        return gw_maintenance + router_maintenance + leafs_maintenance + relays_maintenance
        

    def calc_power_consumption(self, g):
        #Maintenance opex
        relays = [n for n in g if  g.degree[n] > 1]
        leafs = [n for n in g if  g.degree[n] == 1]
        gws = [n for n in g if 'type' in g.nodes[n] and g.nodes[n]['type'] == 'gateway']

        n_tot_ants = sum([g.nodes[n]['n_ant'] for n in relays])
            
        gw_consumption = self.p.power_consumption['gateway_router'] * len(gws) * 24* 365 * self.p.cost_kw * self.p.power_factor
        router_consumption = self.p.power_consumption['relay_router']  * 24* 365 * self.p.cost_kw * self.p.power_factor
        leafs_consumption = self.p.power_consumption['leaf_radio'] * len(leafs) * 24* 365 * self.p.cost_kw * self.p.power_factor
        relays_consumption = self.p.power_consumption['mp_radio'] * n_tot_ants * 24* 365 * self.p.cost_kw * self.p.power_factor
        return gw_consumption + router_consumption + leafs_consumption + relays_consumption



    def get_network_capexes(self, graphs):
        #data_summed = process_map(self._get_network_capex, graphs, max_workers=16)    
        data_summed = [self._get_network_capex(g) for g in tqdm(graphs)]     
        # self.edf = pd.DataFrame(data)
        self.sedf = pd.DataFrame(flatten(data_summed))

    def _get_network_capex(self, graph):
        data = []
        data_summed = []
        (p, mgb, w_g,f_g, n_subs)  = graph
        area, ratio, cluster_size, algo, n_gw, time, random_seed = p
        
    
        cluster_size = int(cluster_size)
        ratio = int(ratio)
        costs = self.calc_cost_wireless(w_g, mgb)
        costs.append(self.calc_cost_fiber(f_g))
    
        type_costs = ['router_cost', 'deploy', 'radio_cost', 'fiber_cost']
        total_cost = sum(costs)
        data_summed.append({'capex': total_cost/n_subs/5/12,
                            'area': area, 
                            'cluster_size': cluster_size,
                            'ratio':ratio,
                            'algo': algo,
                            'n_gw': n_gw, 
                            'mgb': mgb})

        # for i in range(4):
        #     measures = {}
        #     measures['nodes'] = len(w_g)
        #     measures['n_gw'] = n_gw
        #     measures['cost'] = costs[i]
        #     measures['algo'] = algo
        #     measures['cost_customer'] = costs[i]/n_subs
        #     measures['5ymontlycostcustomer'] = costs[i]/n_subs/5/12
        #     measures['type_cost'] = type_costs[i]
        #     measures['area'] = area
        #     measures['ratio'] = ratio
        #     measures['cluster_size'] = cluster_size
        #     measures['time'] = time
        #     measures['seed'] = random_seed
        #     measures['n_gw'] = n_gw

        #     data.append(measures)
        return data_summed
         


    def get_network_opexes(self, graphs):
        data = process_map(self._get_network_opex, graphs, max_workers=16, chunksize = 100) 
        self.opdf = pd.DataFrame(flatten(data))

    def _get_network_opex(self, graph):
        data = []
        data_summed = []
        (p, mgb, w_g,f_g, n_subs)  = graph
        area, ratio, cluster_size, algo, n_gw, time, random_seed = p
            
        fiber_cost, transport_cost = self.calc_opex_fiber_network(w_g, mgb)
        cost_types = ['fiber', 'planed_maintenance', 'unplanned_maintenance']
        planned_maint = self.calc_opex_maintenance(w_g, 'planned_maintenance')
        unplanned_maint = self.calc_opex_maintenance(w_g, 'unplanned_maintenance')
        power_consumpt = self.calc_power_consumption(w_g)

        data.append({'fiber_cost': fiber_cost/n_subs/12,
                    'algo': algo,
                    'transport_cost': transport_cost/n_subs/12,
                    'unplanned_cost':  unplanned_maint/n_subs/12,
                    'power_consumption': power_consumpt/n_subs/12,
                    'area': area, 
                    'cluster_size': cluster_size,
                    'ratio':ratio,
                    'n_gw': n_gw,
                    'mgb': mgb})
        return data


    def save_csv_results(self, csvfolder):
        for mgb in self.opdf.mgb.unique():
            costs = self.opdf[(self.opdf.mgb==mgb)].groupby(['cluster_size', 'ratio'])[['fiber_cost', 'transport_cost', 'unplanned_cost', 'power_consumption']].agg(['mean', ci])
            capex = self.sedf[(self.opdf.mgb==mgb)].groupby(['cluster_size', 'ratio'])['capex'].agg(['mean', ci])
            costs['capex', 'mean'] = capex['mean']
            costs['capex', 'ci'] = capex['ci']
            costs.columns = ["_".join(a) for a in costs.columns.to_flat_index()]
            costs['recurring'] = costs['fiber_cost_mean'] + costs['transport_cost_mean'] + costs['unplanned_cost_mean']
            costs['sum'] = costs['recurring'] + costs['capex_mean']

            pd.set_option('display.max_rows',None)
            pd.set_option('display.max_columns',None)
            pd.set_option('display.width',None)
            print(costs[[i for i in costs.columns if "sum"==i]])

            #costs.index = costs.index/100
            to_csv_comment(costs, f'{csvfolder}/costs_1_dijkstra_{mgb}.csv')

    def plot(self):
        pass