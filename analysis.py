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
        for rdx, r in enumerate(os.listdir(f'{basedir}/{d}/{f}')):
            g = nx.read_graphml(f'{basedir}/{d}/{f}/{r}')
            key = f"{f}_{rdx}"
            measures = {}
            area, lamb, radius, x, y = d.split('_')
            measures['key'] = key
            measures['run'] = f
            measures['nodes'] = len(g)
            measures['edges'] = len(g.edges())
            measures['diamter'] = nx.diameter(g)
            measures['area'] = area
            measures['lamb'] = lamb
            data.append(measures)


df = pd.DataFrame(data)
print(df)
sns.relplot(data=df, x='lamb', y='diamter', kind='line')
plt.show()
