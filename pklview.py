import pickle

with open('simple_node_features.pkl', 'rb') as f:
    node_features = pickle.load(f)

print("Keys in node_features:", list(node_features.keys())[:10])  # print first 10 keys to see their type and values
