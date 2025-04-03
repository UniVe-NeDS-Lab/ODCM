import networkx as nx
import glob
import copy
import re
from utils import *
from base_metrics import *
from capacity import CapacityAnalysis
from costs import CostsAnalysis
from params import Params
from tqdm.contrib.concurrent import process_map  # or thread_map

class odcm_analyze:
    def __init__(self, basedir, figdir, csvdir, radio_type):
        self.basedir = basedir
        self.figdir = figdir+'/'+radio_type
        self.csvdir = csvdir+'/'+radio_type
        os.makedirs(self.figdir, exist_ok=True)
        os.makedirs(self.csvdir, exist_ok=True)
        #CapEx costs
        self.p = Params()
        self.radio_type = radio_type
        self.p.set_radio(self.radio_type)
        self.p.min_guaranteed_bws = [70, 90]

    def read_graph(self, f):  
        area, ratio, cluster_size, algo, n_gw, time, random_seed = re.split('[._/]', f)[4:11]
        cluster_size = int(cluster_size)
        ratio = int(ratio)
        g_params = [area, ratio, cluster_size, algo, n_gw, time, random_seed]
        g = nx.read_graphml(f)
        f_g = nx.read_graphml(f.replace('wireless', 'fiber'))
        n_cust = sum([g.nodes[n]['subscriptions'] for n in g.nodes() if g.degree()[n] > 0])
        if n_cust==0:
            print(f"No customers in {f}")
            return []

        graphs = []
        for mgb in self.p.min_guaranteed_bws:
            graphs.append((g_params, mgb, copy.deepcopy(g),  copy.deepcopy(f_g), n_cust))
        return  graphs

    #@cached("allgraphs.pickle")
    def read_graphs_parallel(self):
        files = glob.glob(f'{self.basedir}/*/*.wireless.graphml.gz')
        print(f"Found {len(files)} graphs in {self.basedir}")

        graphs = process_map(self.read_graph, files, max_workers=8, chunksize=10)
        return flatten(graphs)

    
    def analyze(self, n=0):
        self.graphs = self.read_graphs_parallel()
        if n>0:
            print(f"Loaded {n} out of {len(self.graphs)} Topologies")
            self.graphs = self.graphs[:n]
        else:
            print(f"Loaded {len(self.graphs)} out of {len(self.graphs)} Topologies")
        print("Computing base metrics")
        bmdf = compute_simple_metricses(self.graphs)
        print(bmdf)
        print(bmdf.subscriptions.mean())
        
        print("Computing costs")
        ca = CostsAnalysis(self.p)
        ca.get_network_capexes(self.graphs)
        ca.get_network_opexes(self.graphs)
        ca.save_csv_results(self.csvdir)

        print("compute number of antennas")
        ant_df = compute_n_ants(self.graphs)
        for mgb in ant_df.mgb.unique():
            print(mgb)
            a = ant_df[ant_df.mgb==mgb]
            ag = a.groupby(['type', 'cluster_size','ratio' ]).antennae.mean()
            ag.to_csv(f'{self.csvdir}/n_ant_{mgb}.csv')

        print("Computing capacities")
        bc = CapacityAnalysis(self.p)
        bc.get_link_capacities(self.graphs)
        bc.get_network_capacities(self.graphs)
        bc.plot_capacity(self.figdir)
        bc.save_capacity_csv(self.csvdir)
        bc.save_link_capacities_csv(self.csvdir)
        




if __name__ == "__main__":
    basedir = '../results'
    figdir = 'figures'
    csvdir = 'processed'
    oa = odcm_analyze(basedir, figdir, csvdir, radio_type='802.11ad_pro')
    oa.analyze()