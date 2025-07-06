import json
import numpy as np
from collections import Counter

def build_vocab(nodes, top_k=50):
    # Gather all words in research interests and affiliations
    all_words = []
    for node in nodes:
        all_words.extend([w.lower() for w in node.get('research_interests', [])])
        for aff in node.get('affiliations', []):
            all_words.extend(aff.lower().split())
    # Count frequency and take top_k
    counter = Counter(all_words)
    vocab = [word for word, freq in counter.most_common(top_k)]
    return vocab

def text_to_binary_vector(text_list, vocab):
    text_words = set(w.lower() for w in text_list)
    return np.array([1 if word in text_words else 0 for word in vocab], dtype=np.float32)

def affiliations_to_binary_vector(affiliations, vocab):
    words = []
    for aff in affiliations:
        words.extend(aff.lower().split())
    return text_to_binary_vector(words, vocab)

def get_node_features(node, vocab):
    # Numeric features
    paper_count = node.get('paper_count', 0)
    citation_count = node.get('citation_count', 0)
    h_index = node.get('h_index', 0)
    p_index_eq = node.get('p_index_eq', 0.0)
    p_index_uneq = node.get('p_index_uneq', 0.0)
    numeric_features = np.array([paper_count, citation_count, h_index, p_index_eq, p_index_uneq], dtype=np.float32)

    # Binary vector for research interests
    research_vec = text_to_binary_vector(node.get('research_interests', []), vocab)

    # Binary vector for affiliations
    affiliations_vec = affiliations_to_binary_vector(node.get('affiliations', []), vocab)

    # Combine all features
    features = np.concatenate([numeric_features, research_vec, affiliations_vec])
    return features

def main():
    with open('json_files/reduced_nodes_connected.json', 'r', encoding='utf-8') as f:
        nodes = json.load(f)

    vocab = build_vocab(nodes, top_k=50)
    print("Vocabulary:", vocab)

    all_features = {}
    for node in nodes:
        features = get_node_features(node, vocab)
        all_features[node['id']] = features

    import pickle
    with open('simple_node_features.pkl', 'wb') as f:
        pickle.dump(all_features, f)

    print("Features extracted and saved to simple_node_features.pkl")

if __name__ == "__main__":
    main()
