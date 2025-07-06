import json
import matplotlib.pyplot as plt
from collections import defaultdict

# Load the data
with open("nodes.json", "r", encoding='utf-8') as f:
    nodes = json.load(f)

with open("edges.json", "r", encoding='utf-8') as f:
    edges = json.load(f)

# Step 1: Calculate total number of nodes and edges
all_node_ids = set(node["id"] for node in nodes)
all_node_ids.update(edge["source"] for edge in edges)
all_node_ids.update(edge["target"] for edge in edges)
total_nodes = len(all_node_ids)
total_nodes = len(nodes)
total_edges = len(edges)

# Step 2: Calculate degree and distribution of connections
degree_count = defaultdict(int)

# Count the degree of each node by processing the edges
for edge in edges:
    degree_count[edge["source"]] += 1
    degree_count[edge["target"]] += 1

# Calculate the average degree
degrees = list(degree_count.values())
avg_degree = sum(degrees) / len(degrees)

# Step 3: Degree distribution data
degree_distribution = [degree_count[node] for node in degree_count]

# Prepare the results to write to a text file
results = []
results.append(f"Total nodes: {total_nodes}")
results.append(f"Total edges: {total_edges}")
results.append(f"Average degree: {avg_degree:.2f}")
results.append("\nDegree distribution (node -> degree):")

# Add degree distribution data to results
for node, degree in degree_count.items():
    results.append(f"Node {node}: Degree {degree}")

# Write the results to a text file
with open("statistics_results.txt", "w", encoding='utf-8') as f:
    f.write("\n".join(results))

# Output message
print("✅ Statistics saved to 'statistics_results.txt'.")

# Plot the degree distribution (limited to degrees from 0 to 150)
plt.figure(figsize=(10, 6))
plt.hist(degrees, bins=range(0, 151), alpha=0.75, color="blue", edgecolor='black')
plt.title("Degree Distribution")
plt.xlabel("Degree")
plt.ylabel("Frequency")
plt.xticks(range(0, 151, 10))  # Tick every 10 degrees
plt.xlim(0, 150)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
