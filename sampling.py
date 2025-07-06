import json
import random
from collections import deque

# Paramètres
target_size = 5000
nodes_file = 'clean_nodes.json'
edges_file = 'clean_edges.json'

# Charger les données
with open(nodes_file, 'r', encoding='utf-8') as f:
    all_nodes = json.load(f)
with open(edges_file, 'r', encoding='utf-8') as f:
    all_edges = json.load(f)

# Construire un index pour les nœuds
node_dict = {node['id']: node for node in all_nodes}

# Construire le graphe en mémoire (dictionnaire d'adjacence)
adjacency = {}
edge_weights = {}
for edge in all_edges:
    src, tgt, weight = edge['source'], edge['target'], edge['weight']
    adjacency.setdefault(src, set()).add(tgt)
    adjacency.setdefault(tgt, set()).add(src)
    edge_weights[(src, tgt)] = weight
    edge_weights[(tgt, src)] = weight

# Choisir un nœud de départ avec au moins un voisin
valid_start_nodes = [nid for nid in adjacency if len(adjacency[nid]) > 0]
start_node = random.choice(valid_start_nodes)

# Parcours BFS pour collecter un sous-graphe connexe
visited = set()
queue = deque([start_node])
collected_edges = []
while queue and len(visited) < target_size:
    current = queue.popleft()
    if current in visited:
        continue
    visited.add(current)
    for neighbor in adjacency.get(current, []):
        if neighbor not in visited:
            queue.append(neighbor)
        # Ajouter l'arête si les deux nœuds sont dans le graphe final
        if neighbor in node_dict and current in node_dict:
            weight = edge_weights.get((current, neighbor), None)
            collected_edges.append({'source': current, 'target': neighbor, 'weight': weight})


# Limiter les arêtes aux seuls nœuds visités
final_nodes = [node_dict[nid] for nid in visited if nid in node_dict]
final_node_ids = {node['id'] for node in final_nodes}
final_edges = [
    edge for edge in collected_edges
    if edge['source'] in final_node_ids and edge['target'] in final_node_ids
]

# Sauvegarde
with open('reduced_nodes_connected.json', 'w', encoding='utf-8') as f:
    json.dump(final_nodes, f, indent=2)
with open('reduced_edges_connected.json', 'w', encoding='utf-8') as f:
    json.dump(final_edges, f, indent=2)

print(f"✅ {len(final_nodes)} nœuds connectés sélectionnés.")
print(f"🔗 {len(final_edges)} arêtes connectées conservées.")
