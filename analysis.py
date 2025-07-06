import json
import networkx as nx

# Load data
with open("nodes.json", "r", encoding='utf-8') as f:
    nodes = json.load(f)

with open("edges.json", "r", encoding='utf-8') as f:
    edges = json.load(f)

# Create graph
G = nx.Graph()
for edge in edges:
    G.add_edge(edge["source"], edge["target"], weight=edge["weight"])

# Save basic stats
with open("step1_basic_stats.txt", "w", encoding='utf-8') as f:
    f.write(f"Total nodes: {len(G.nodes())}\n")
    f.write(f"Total edges: {len(G.edges())}\n")

print("✅ Step 1: Graph loaded and basic stats saved.")
