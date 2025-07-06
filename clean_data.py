import json
import os
import warnings
import networkx as nx  # N'oublie pas d'importer NetworkX si tu l'utilises
warnings.filterwarnings('ignore')

class AMinerGraphAnalyzer:
    def __init__(self, nodes_file='clean_nodes.json', edges_file='clean_edges.json'):
        self.nodes_file = nodes_file
        self.edges_file = edges_file
        self.graph = nx.Graph()
        self.nodes_data = None
        self.edges_data = None
        self.load_graph()  # Charge les fichiers directement dans le graphe

    def load_graph(self):
        """Charge les fichiers JSON dans le graphe"""
        print("load_graph")
        with open(self.nodes_file, 'r', encoding='utf-8') as f:
            self.nodes_data = json.load(f)
        with open(self.edges_file, 'r', encoding='utf-8') as f:
            self.edges_data = json.load(f)

        for node in self.nodes_data:
            self.graph.add_node(node['id'], **{k: v for k, v in node.items() if k != 'id'})
        for edge in self.edges_data:
            self.graph.add_edge(edge['source'], edge['target'])

    def clean_and_save_nodes(self, original_nodes_path="clean_nodes.json", removed_nodes_path="removed_nodes.json"):
        """Supprime les nœuds invalides, sauvegarde ceux supprimés et réécrit les nœuds valides"""
        print("start clean_and_save_nodes")
        removed_nodes = []
        valid_nodes = []
        nodes_to_remove = []

        for node_id, attrs in self.graph.nodes(data=True):
            if (
                not attrs.get('name') or
                not attrs.get('affiliations') or
                attrs.get('paper_count', 0) <= 0 or
                attrs.get('citation_count', 0) <= 0 or
                attrs.get('h_index', 0) <= 0 or
                attrs.get('p_index_eq', 0.0) <= 0 or
                attrs.get('p_index_uneq', 0.0) <= 0 or
                not attrs.get('research_interests')
            ):
                node_info = {'id': node_id}
                node_info.update(attrs)
                removed_nodes.append(node_info)
                nodes_to_remove.append(node_id)
            else:
                node_info = {'id': node_id}
                node_info.update(attrs)
                valid_nodes.append(node_info)

        # Supprimer les nœuds invalides du graphe
        self.graph.remove_nodes_from(nodes_to_remove)

        # Écriture des nœuds valides
        with open(original_nodes_path, 'w', encoding='utf-8') as f:
            json.dump(valid_nodes, f, ensure_ascii=False, indent=2)

        # Écriture des nœuds supprimés
        with open(removed_nodes_path, 'w', encoding='utf-8') as f:
            json.dump(removed_nodes, f, ensure_ascii=False, indent=2)

        print(f"{len(nodes_to_remove)} nœuds supprimés et enregistrés dans {removed_nodes_path}")
        print(f"{len(valid_nodes)} nœuds valides enregistrés dans {original_nodes_path}")

    def clean_and_save_edges(self, original_edges_path="clean_edges.json", txt_path="clean_edges.txt"):
        """Supprime les arêtes invalides et les sauvegarde en JSON et TXT (ligne par ligne au format JSON)"""
        print("start clean_and_save_edges")
        cleaned_edges = []
        current_nodes = set(self.graph.nodes)

        for edge in self.edges_data:
            if edge['source'] in current_nodes and edge['target'] in current_nodes:
                cleaned_edges.append({
                    "source": edge["source"],
                    "target": edge["target"],
                    "weight": edge.get("weight", 1)  # default weight = 1 if not specified
                })

        # ✅ Save to clean_edges.json
        with open(original_edges_path, 'w', encoding='utf-8') as f:
            json.dump(cleaned_edges, f, ensure_ascii=False, indent=2)

        print(f"{len(cleaned_edges)} arêtes valides ont été nettoyées et enregistrées.")
        print(f"✔ Les arêtes nettoyées ont été sauvegardées dans '{original_edges_path}' (JSON) et '{txt_path}' (ligne par ligne).")


def main():
    print("start main")
    analyzer = AMinerGraphAnalyzer()
    analyzer.clean_and_save_nodes(
        original_nodes_path="clean_nodes.json",
        removed_nodes_path="removed_nodes.json"
    )
    analyzer.clean_and_save_edges(
    original_edges_path="clean_edges.json",
    txt_path="clean_edges.txt"
)

if __name__ == "__main__":
    main()
