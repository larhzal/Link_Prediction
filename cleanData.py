import json

# === Load raw data ===

with open("json_files/nodes.json", "r", encoding="utf-8") as f:
    raw_nodes = json.load(f)

with open("json_files\edges.json", "r", encoding="utf-8") as f:
    raw_edges = json.load(f)

# === Step 1: Remove duplicate nodes ===

seen_ids = set()
clean_nodes = []
for node in raw_nodes:
    if node['id'] not in seen_ids:
        seen_ids.add(node['id'])
        clean_nodes.append(node)

# === Step 2: Remove duplicate and reverse edges ===

seen_edges = set()
clean_edges = []
for edge in raw_edges:
    a, b = edge['source'], edge['target']
    key = tuple(sorted([a, b]))  # undirected
    if key not in seen_edges:
        seen_edges.add(key)
        clean_edges.append(edge)

# === Step 3: Validate Author IDs ===

valid_ids = set(node['id'] for node in clean_nodes)
clean_edges = [
    edge for edge in clean_edges
    if edge['source'] in valid_ids and edge['target'] in valid_ids
]

# === Step 4: Remove self-loops ===

clean_edges = [edge for edge in clean_edges if edge['source'] != edge['target']]

# === Step 5: Remove isolated nodes ===

connected_ids = set()
for edge in clean_edges:
    connected_ids.add(edge['source'])
    connected_ids.add(edge['target'])

clean_nodes = [node for node in clean_nodes if node['id'] in connected_ids]

# === Step 6: Clean names and check missing values ===

final_nodes = []
for node in clean_nodes:
    if node.get('id') is not None and node.get('name') and node.get('affiliations'):
        if isinstance(node['affiliations'], list) and len(node['affiliations']) > 0:
            node['name'] = node['name'].strip()
            final_nodes.append(node)

# === Save cleaned data ===

with open("clean_nodes.json", "w", encoding="utf-8") as f:
    json.dump(final_nodes, f, indent=2)

with open("clean_edges.json", "w", encoding="utf-8") as f:
    json.dump(clean_edges, f, indent=2)

print("✅ Cleaning complete! Cleaned files: clean_nodes.json and clean_edges.json")
