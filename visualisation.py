import json
import networkx as nx
import matplotlib.pyplot as plt


with open("edges.json", "r", encoding="utf-8") as f:
    edges = json.load(f)
G = nx.Graph()
G.add_edges_from([(edge['source'], edge['target']) for edge in edges])
top_nodes = list(G.nodes)[:200]
subgraph = G.subgraph(top_nodes)
pos = nx.kamada_kawai_layout(subgraph)
plt.figure(figsize=(12, 10))
nx.draw_networkx_nodes(subgraph, pos, node_size=40, node_color='skyblue', alpha=0.8)
nx.draw_networkx_edges(subgraph, pos, alpha=0.4)
plt.title("Visualisation d’un Sous-graphe (200 Nœuds)")
plt.axis("off")
plt.show()











# import json
# import matplotlib.pyplot as plt
# import networkx as nx

# # Load data
# with open("nodes.json", "r", encoding='utf-8') as f:
#     nodes = json.load(f)

# with open("edges.json", "r", encoding='utf-8') as f:
#     edges = json.load(f)

# # Build full graph
# G = nx.Graph()
# G.add_nodes_from([node["id"] for node in nodes])
# G.add_edges_from([(edge["source"], edge["target"]) for edge in edges])

# # ⚠️ Use a subgraph of the top 200 nodes (to keep it fast)
# sub_nodes = list(G.nodes)[:200]
# subgraph = G.subgraph(sub_nodes)

# # Use a faster layout: kamada_kawai_layout
# pos = nx.kamada_kawai_layout(subgraph)

# # Draw the subgraph
# plt.figure(figsize=(12, 10))
# nx.draw(
#     subgraph,
#     pos,
#     node_size=40,
#     with_labels=False,
#     node_color="skyblue",
#     edge_color="gray",
#     alpha=0.7
# )
# plt.title("Network Visualization (Top 200 Nodes - Kamada-Kawai Layout)")
# plt.show()
