import json
import pickle
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

class AMinerGraphAnalyzer:
    def __init__(self, nodes_file='json_files/reduced_nodes_connected.json', edges_file='json_files/reduced_edges_connected.json'):
        self.nodes_file = nodes_file
        self.edges_file = edges_file
        self.graph = None
        self.nodes_data = None
        self.edges_data = None

    def load_data(self):
        """Charger les données des fichiers JSON."""
        print("📂 Chargement des données...")
        with open(self.nodes_file, 'r', encoding='utf-8') as f:
            self.nodes_data = json.load(f)
        with open(self.edges_file, 'r', encoding='utf-8') as f:
            self.edges_data = json.load(f)
        print(f"✅ {len(self.nodes_data)} nœuds chargés, {len(self.edges_data)} arêtes chargées.")

    def build_graph_and_save_matrix(self):
        """Construit le graphe et sauvegarde la matrice d’adjacence."""
        if self.nodes_data is None or self.edges_data is None:
            self.load_data()

        self.graph = nx.Graph()

        # Ajouter les nœuds
        for node in self.nodes_data:
            node_id = str(int(float(node['id'])))
            self.graph.add_node(node_id, **{k: v for k, v in node.items() if k != 'id'})

        # Ajouter les arêtes
        for edge in self.edges_data:
            src = str(int(float(edge['source'])))
            tgt = str(int(float(edge['target'])))
            if self.graph.has_node(src) and self.graph.has_node(tgt):
                self.graph.add_edge(src, tgt)
            else:
                print(f"❌ Arête ignorée : ({src}, {tgt})")

        """Sauvegarde la liste d'adjacence dans un fichier texte."""
        print("📄 Sauvegarde de la liste d’adjacence...")

        with open("adjacency_list.txt", "w", encoding="utf-8") as f:
            for node in self.graph.nodes():
                neighbors = list(self.graph.neighbors(node))
                line = f"{node}: {', '.join(neighbors)}\n"
                f.write(line)

        print("✅ Liste d’adjacence enregistrée dans 'adjacency_list.txt'.")

        # Matrice d’adjacence
        print("💾 Sauvegarde de la matrice d’adjacence...")
        nodes = list(self.graph.nodes())
        node_index = {str(node): i for i, node in enumerate(nodes)}
        adj_matrix = np.zeros((len(nodes), len(nodes)), dtype=int)

        for u, v in self.graph.edges():
            i, j = node_index[str(u)], node_index[str(v)]
            adj_matrix[i][j] = 1
            adj_matrix[j][i] = 1  # Graphe non orienté

        # ... après avoir rempli la matrice
        nonzero = np.count_nonzero(adj_matrix)
        print(f"✅ Nombre de connexions dans la matrice d’adjacence : {nonzero}")

        np.savetxt("adjacency_matrix.txt", adj_matrix, fmt="%d")
        print("✅ Matrice d’adjacence enregistrée dans 'adjacency_matrix.txt'.")

    def save_graph_visualization(self):
        """Enregistre une image du graphe."""
        print("🖼️ Sauvegarde de la visualisation du graphe...")
        plt.figure(figsize=(16, 12))
        pos = nx.spring_layout(self.graph, seed=42)

        nx.draw(
            self.graph,
            pos,
            with_labels=False,
            node_size=30,
            node_color='skyblue',
            edge_color='gray',
            alpha=0.7
        )

        plt.title("Visualisation complète du graphe")
        plt.savefig("graph_visualization.png", dpi=300)
        plt.close()
        print("✅ Graphe enregistré dans 'graph_visualization.png'.")

def main():
    analyzer = AMinerGraphAnalyzer()
    analyzer.build_graph_and_save_matrix()
    with open("reduced_graph.gpickle", "wb") as f:
        pickle.dump(analyzer.graph, f)
    analyzer.save_graph_visualization()

if __name__ == "__main__":
    main()
