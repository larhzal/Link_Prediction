import json

# === Load nodes ===
with open('reduced_nodes_connected.json', 'r', encoding='utf-8') as f_nodes:
    nodes_data = json.load(f_nodes)
    node_count = len(nodes_data)

# === Load edges ===
with open('reduced_edges_connected.json', 'r', encoding='utf-8') as f_edges:
    edges_data = json.load(f_edges)
    edge_count = len(edges_data)

# === Output ===
print(f"Number of nodes: {node_count}")
print(f"Number of edges: {edge_count}")
