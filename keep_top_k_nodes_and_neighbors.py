import json


def keep_top_k_nodes_and_neighbors(G, k=20000):
    # Step 1: Get top k nodes by degree
    top_nodes = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:k]
    top_node_ids = set(n for n, _ in top_nodes)

    # Step 2: Add all neighbors of these top nodes
    neighbors = set()
    for node in top_node_ids:
        neighbors.update(G.neighbors(node))

    # Step 3: Combine and trim to unique node set
    final_node_ids = top_node_ids.union(neighbors)

    # Optional: if the final number of nodes is too large, re-trim
    if len(final_node_ids) > 25000:
        final_node_ids = set(list(final_node_ids)[:25000])

    # Step 4: Create subgraph
    G_sub = G.subgraph(final_node_ids).copy()
    return G_sub

def save_nodes_to_json(G_sub, filename="top_nodes.json"):
    nodes_data = []
    for node, data in G_sub.nodes(data=True):
        node_entry = {"id": node}
        node_entry.update(data)  # include any attributes like 'name'
        nodes_data.append(node_entry)
    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(nodes_data, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved {len(nodes_data)} nodes to {filename}")

def save_edges_to_json(G_sub, filename="top_edges.json"):
    edges_data = []
    for source, target in G_sub.edges():
        edges_data.append({"source": source, "target": target})
    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(edges_data, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved {len(edges_data)} edges to {filename}")