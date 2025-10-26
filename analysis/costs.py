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
                g.nodes[n]['router_cost'] = np.array([0,0])
                g.nodes[n]['deploy'] = np.array([0,0])
                g.nodes[n]['fiber_cost'] = np.array([0,0])
                g.nodes[n]['radio_cost'] = np.array([0,0])
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
                g.nodes[n]['fiber_cost' ] = np.array([0,0])
                if g.degree()[n] == 1:
                    #leaf node, no router
                    g.nodes[n]['radio_cost'] = self.p.capex_costs['leaf_radio']
                    g.nodes[n]['deploy'] = self.p.capex_costs['leaf_deploy']
                    g.nodes[n]['router_cost'] = np.array([0,0])
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
        else:
            n_fibers = m.ceil(total_bw/100)
            fiber_trasport = n_fibers*self.p.opex_costs['transport_100']
        # elif total_bw < 100:
        #     fiber_trasport = self.p.opex_costs['transport_100']
        # elif total_bw < 200:
        #     fiber_trasport = 2*self.p.opex_costs['transport_100']
        # elif total_bw < 300:
        #     fiber_trasport = 3*self.p.opex_costs['transport_100']
        # elif total_bw < 400:
        #     fiber_trasport = 4*self.p.opex_costs['transport_100']
        # elif total_bw < 400:
        #     fiber_trasport = 4*self.p.opex_costs['transport_100']
        # else:
        #     raise ValueError(f"Can't relay more than 200G : {total_bw}")
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
        #Removing cost of devices as we already amortize it on 5y
        gw_maintenance = p_router_failure * len(gws) * (self.p.mttr['router'] * self.p.opex_costs[kind])
        router_maintenance = p_router_failure  * len(relays) * (self.p.mttr['router'] * self.p.opex_costs[kind])
        leafs_maintenance = p_leaf * p_radio_failure * len(leafs) * (self.p.mttr['radio'] * self.p.opex_costs['planned_maintenance'])
        relays_maintenance = p_relay * p_radio_failure * n_tot_ants * (self.p.mttr['radio'] * self.p.opex_costs[kind])
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
        #data_summed = process_map(self._get_network_capex, graphs, max_workers=8)   
        data = []
        data_summed = []
        for g in tqdm(graphs):
            d_s, d = self._get_network_capex(g) 
            data.append(d)
            data_summed.append(d_s)
        self.edf = pd.DataFrame(flatten(data))
        self.sedf = pd.DataFrame(flatten(data_summed))

    def _get_network_capex(self, graph):
        data = []
        data_summed = []
        (p, mgb, w_g,f_g, n_subs)  = graph
        area, ratio, cluster_size, algo, n_gw, time, random_seed = p
        
    
        cluster_size = int(cluster_size)
        ratio = int(ratio)
        costs = []
        wireless_cost = self.calc_cost_wireless(w_g, mgb)
        costs += wireless_cost
        #Take 1/3 of the cost of fiber to amortize its cost in 15y rather than 5
        costs.append(self.calc_cost_fiber(f_g)/3)
    
        type_costs = ['router_cost', 'deploy', 'radio_cost', 'fiber_cost']
        total_cost = sum(costs)
        sat_cost = sum(wireless_cost)
        data_summed.append({'capex': total_cost/n_subs/5/12,
                            'capex_sat': sat_cost/n_subs/5/12,
                            'area': area, 
                            'cluster_size': cluster_size,
                            'ratio':ratio,
                            'algo': algo,
                            'n_gw': n_gw, 
                            'mgb': mgb})

        for i in range(4):
            measures = {}
            measures['nodes'] = len(w_g)
            measures['n_gw'] = n_gw
            measures['cost'] = costs[i]
            measures['algo'] = algo
            measures['cost_customer'] = costs[i]/n_subs
            measures['5ymontlycostcustomer'] = costs[i]/n_subs/5/12
            measures['type_cost'] = type_costs[i]
            measures['area'] = area
            measures['ratio'] = ratio
            measures['cluster_size'] = cluster_size
            measures['time'] = time
            measures['seed'] = random_seed
            measures['n_gw'] = n_gw

            data.append(measures)
        return data_summed, data
         


    def get_network_opexes(self, graphs):
        data = process_map(self._get_network_opex, graphs, max_workers=8, chunksize = 100) 
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
        def get_min(x):
            return x.apply(lambda arr: arr[0]).mean()  # Get first element (min)
    
        def get_max(x):
            return x.apply(lambda arr: arr[1]).mean()  # Get second element (max)
    
        for mgb in self.opdf.mgb.unique():
            costs = self.opdf[(self.opdf.mgb==mgb)].groupby(['cluster_size', 'ratio'])[['fiber_cost', 'transport_cost', 'unplanned_cost', 'power_consumption']].agg([get_min, get_max])
            capex = self.sedf[(self.opdf.mgb==mgb)].groupby(['cluster_size', 'ratio'])[['capex', 'capex_sat']].agg([get_min, get_max])
            capex.columns = ["_".join(a) for a in capex.columns.to_flat_index()]

            costs['capex', 'min'] = capex['capex_get_min']
            costs['capex', 'max'] = capex['capex_get_max']
            costs['capex_sat', 'min'] = capex['capex_sat_get_min']
            costs['capex_sat', 'max'] = capex['capex_sat_get_max']
            
            costs.columns = ["_".join(a) for a in costs.columns.to_flat_index()]
            costs['recurring_min'] = costs['fiber_cost_get_min'] + costs['transport_cost_get_min'] + costs['unplanned_cost_get_min'] + costs['power_consumption_get_min']
            costs['recurring_max'] = costs['fiber_cost_get_max'] + costs['transport_cost_get_max'] + costs['unplanned_cost_get_max'] + costs['power_consumption_get_max']
            costs['sum_min'] = costs['recurring_min'] + costs['capex_min']
            costs['sum_max'] = costs['recurring_max'] + costs['capex_max']

            pd.set_option('display.max_rows',None)
            pd.set_option('display.max_columns',None)
            pd.set_option('display.width',None)
            print(costs[[i for i in costs.columns if "sum"==i]])

            #costs.index = costs.index/100
            to_csv_comment(costs, f'{csvfolder}/costs_1_dijkstra_{mgb}.csv')
            edf_processed = self.edf.copy()
        
            # Process each column with array values
            array_columns = ['cost', 'cost_customer', '5ymontlycostcustomer']
            
            for col in array_columns:
                # Extract min and max into separate columns
                edf_processed[f'{col}_min'] = edf_processed[col].apply(lambda x: x[0])
                edf_processed[f'{col}_max'] = edf_processed[col].apply(lambda x: x[1])
                #edf_processed[f'{col}_mean'] = edf_processed[col].apply(lambda x: (x[0] + x[1])/2)
                
                # Drop the original array column to avoid confusion
                edf_processed = edf_processed.drop(columns=[col])
            
            # Save the processed edf to CSV
            to_csv_comment(edf_processed, f'{csvfolder}/capex_1_dijkstra_{mgb}.csv')

    def plot(self):
        pass