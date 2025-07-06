from graph_creation import basic_graph_visualization, create_networkx_graph, load_data, network_statistics


if __name__ == "__main__":
    nodes, edges = load_data()
    if nodes and edges:
        G = create_networkx_graph(nodes, edges)
        network_statistics(G)
        basic_graph_visualization(G, title="Professional Social Network")
