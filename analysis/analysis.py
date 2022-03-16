import seaborn as sns
import os
import networkx as nx
from pprint import pprint
import pandas as pd
import matplotlib.pyplot as plt

basedir = 'results'
data = []
for d in os.listdir(basedir):
    for f in os.listdir(f'{basedir}/{d}'):
        g = nx.read_graphml(f'{basedir}/{d}/{f}')
        measures = {}
        area, lamb, radius, x, y = d.split('_')
        time, random_seed, topo_strategy, gw_strategy, runid = f.split('.')[0].split('_')
        measures['run'] = runid
        measures['nodes'] = len(g)
        measures['edges'] = len(g.edges())
        measures['diameter'] = nx.diameter(g)
        measures['area'] = area
        measures['lamb'] = lamb
        measures['topo_strategy'] = topo_strategy
        measures['gw_strategy'] = gw_strategy
        measures['time'] = time
        measures['seed'] = random_seed
        data.append(measures)


df = pd.DataFrame(data)
print(df)
sns.relplot(data=df, x='topo_strategy', y='diameter', kind='line', hue='area')
plt.show()
