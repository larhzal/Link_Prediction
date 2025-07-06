import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from models import GCN, GCNLinkPredictor, GAT, GATLinkPredictor, GraphSAGE, GraphSAGELinkPredictor
from evaluate_models import evaluate_and_plot
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

from models import LinkPredictor

# Choose your model here: "GCN", "GAT", or "GraphSAGE"
MODEL_NAME = "GCN"

# Load data (adjust paths if needed)
train_df = pd.read_csv("data/train_links.csv")
val_df = pd.read_csv("data/val_links.csv")
test_df = pd.read_csv("data/test_links.csv")

# Create full graph edges and node mapping
all_edges = pd.concat([train_df[['source', 'target']],
                       val_df[['source', 'target']],
                       test_df[['source', 'target']]]).drop_duplicates()
all_nodes = pd.unique(all_edges[['source', 'target']].values.ravel())
id_map = {id_: i for i, id_ in enumerate(all_nodes)}
num_nodes = len(all_nodes)

def encode_edges(df):
    return torch.tensor([[id_map[s], id_map[t]] for s, t in zip(df['source'], df['target'])], dtype=torch.long).T

# Encode edges and labels for train/val/test
train_edge_index = encode_edges(train_df)
val_edge_index = encode_edges(val_df)
test_edge_index = encode_edges(test_df)

train_labels = torch.tensor(train_df['label'].values, dtype=torch.float32)
val_labels = torch.tensor(val_df['label'].values, dtype=torch.float32)
test_labels = torch.tensor(test_df['label'].values, dtype=torch.float32)

# --------- IMPORTANT: Load your actual preprocessed node features here ---------
# Example: load from a saved file or preprocess again exactly like training
# For example, if saved as tensor: x = torch.load('node_features.pt')
# Replace this line with your real features:
x = torch.eye(num_nodes)  # <--- REPLACE this with your real features tensor!
# ------------------------------------------------------------------------------

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move data to device
x = x.to(device)
train_edge_index, val_edge_index, test_edge_index = train_edge_index.to(device), val_edge_index.to(device), test_edge_index.to(device)
train_labels, val_labels, test_labels = train_labels.to(device), val_labels.to(device), test_labels.to(device)

# Select model class and instantiate model
model_class = {"GCN": GCN, "GAT": GAT, "GraphSAGE": GraphSAGE}[MODEL_NAME]
gnn_model = model_class(x.size(1), 64)
model = LinkPredictor(gnn_model).to(device)

# Load trained weights - replace filename as needed
checkpoint_path = f'best_{MODEL_NAME.lower()}_model.pt'
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# Define loss with pos_weight for imbalance
pos_weight = (len(train_labels) - train_labels.sum()) / train_labels.sum()
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

# Evaluate on test set
with torch.no_grad():
    test_out = model(x, train_edge_index, test_edge_index)
    test_probs = torch.sigmoid(test_out).cpu().numpy()
    test_pred = (test_probs > 0.5).astype(int)
    y_true = test_labels.cpu().numpy()

    test_acc = accuracy_score(y_true, test_pred)
    test_auc = roc_auc_score(y_true, test_probs)
    test_f1 = f1_score(y_true, test_pred)

print(f"Final Test Accuracy: {test_acc:.4f}")
print(f"Final Test AUC-ROC: {test_auc:.4f}")
print(f"Final Test F1 Score: {test_f1:.4f}")

# Plot ROC and PR curves using your evaluate_and_plot function
evaluate_and_plot(y_true, test_probs)
