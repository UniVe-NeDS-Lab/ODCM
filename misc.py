import networkx as nx


def copy_graph(g: nx.Graph) -> nx.Graph:
    copy = nx.Graph()
    copy.add_nodes_from(g.nodes(data=True))
    copy.add_edges_from(g.edges(data=True))
    return copy
