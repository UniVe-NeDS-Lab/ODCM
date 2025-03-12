import matplotlib.pyplot as plt
import networkx as nx
from grave import plot_network

fig, ax = plt.subplots()
plot_network(network, node_style=color_dominators)
plt.show()
