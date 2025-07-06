import random
import json
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def generate_link_datasets(graph: nx.Graph, test_size=0.15, val_size=0.15, balance=True, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    # Liens positifs
    positive_edges = list(graph.edges())
    positive_labels = [1] * len(positive_edges)

    # Liens négatifs (pairs non connectées)
    all_nodes = list(graph.nodes())
    num_negatives = len(positive_edges)
    negative_edges = set()

    while len(negative_edges) < num_negatives:
        u, v = random.sample(all_nodes, 2)
        if not graph.has_edge(u, v) and u != v:
            negative_edges.add(tuple(sorted((u, v))))

    negative_edges = list(negative_edges)
    negative_labels = [0] * len(negative_edges)

    # Fusionner
    edges = positive_edges + negative_edges
    labels = positive_labels + negative_labels

    df = pd.DataFrame(edges, columns=["source", "target"])
    df["label"] = labels

    # Mélanger
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Split
    train_val_df, test_df = train_test_split(df, test_size=test_size, stratify=df["label"], random_state=seed)
    val_ratio = val_size / (1 - test_size)
    train_df, val_df = train_test_split(train_val_df, test_size=val_ratio, stratify=train_val_df["label"], random_state=seed)

    # Équilibrage optionnel
    if balance:
        def balance_df(df):
            min_class = df['label'].value_counts().min()
            return pd.concat([
                df[df['label'] == 0].sample(min_class, random_state=seed),
                df[df['label'] == 1].sample(min_class, random_state=seed)
            ]).sample(frac=1, random_state=seed).reset_index(drop=True)

        train_df = balance_df(train_df)
        val_df = balance_df(val_df)
        test_df = balance_df(test_df)

    return train_df, val_df, test_df

def load_graph_from_json(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        edges = json.load(f)

    G = nx.Graph()
    for edge in edges:
        G.add_edge(str(edge['source']), str(edge['target']))  # utiliser str() pour une compatibilité large
    return G

def main():
    print("🔄 Chargement du graphe JSON...")
    G = load_graph_from_json("json_files/reduced_edges_connected.json")
    print("✅ Graphe chargé. Nombre de nœuds :", G.number_of_nodes(), "| arêtes :", G.number_of_edges())

    train_df, val_df, test_df = generate_link_datasets(G)

    train_df.to_csv("train_links.csv", index=False)
    val_df.to_csv("val_links.csv", index=False)
    test_df.to_csv("test_links.csv", index=False)

    print("✅ Données sauvegardées : train_links.csv, val_links.csv, test_links.csv")

if __name__ == "__main__":
    main()
