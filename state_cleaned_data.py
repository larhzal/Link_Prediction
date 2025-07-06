import json
import networkx as nx

# === Load cleaned data ===
with open("clean_nodes.json", "r", encoding="utf-8") as f:
    nodes = json.load(f)

with open("clean_edges.json", "r", encoding="utf-8") as f:
    edges = json.load(f)

#🔽 Build a set of valid node IDs from nodes list
valid_node_ids = set(node["id"] for node in nodes)

G = nx.Graph()

# 🔽 Add only nodes from nodes.json
for node in nodes:
    G.add_node(node["id"])

# 🔽 Add edges only if both source and target exist in nodes
for edge in edges:
    if edge["source"] in valid_node_ids and edge["target"] in valid_node_ids:
        G.add_edge(edge["source"], edge["target"], weight=edge.get("weight", 1))
        
# === Statistics ===
num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()
avg_degree = sum(dict(G.degree()).values()) / num_nodes

print("📊 Cleaned Graph Statistics")
print(f"🔹 Total Nodes       : {num_nodes}")
print(f"🔹 Total Edges       : {num_edges}")
print(f"🔹 Average Degree    : {avg_degree:.2f}")

# Optional: Degree distribution summary
degree_sequence = [d for n, d in G.degree()]
max_degree = max(degree_sequence)
min_degree = min(degree_sequence)

print(f"🔹 Max Degree        : {max_degree}")
print(f"🔹 Min Degree        : {min_degree}")
