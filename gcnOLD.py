import json
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, auc
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Paths to datasets
train_path = "data/train_links.csv"
val_path = "data/val_links.csv"
test_path = "data/test_links.csv"
nodes_path = "json_files/reduced_nodes_connected.json"

# Load and process data
with open(nodes_path) as f:
    nodes = json.load(f)

# Feature selection with fallbacks
feature_sets = {
    'full': ['paper_count', 'citation_count', 'h_index', 'p_index_eq', 'p_index_uneq', 
             'coauthor_count', 'venue_count', 'recent_paper_count'],
    'advanced': ['paper_count', 'citation_count', 'h_index', 'p_index_eq', 'p_index_uneq', 'coauthor_count'],
    'basic': ['paper_count', 'citation_count', 'h_index'],
}

# Select the most comprehensive available feature set
available_features = set(nodes[0].keys())
for set_name, features in feature_sets.items():
    if all(f in available_features for f in features):
        features_to_use = features
        break
else:
    features_to_use = feature_sets['basic']

print(f"Using features: {features_to_use}")

# Load and process edges
train_df = pd.read_csv(train_path)
val_df = pd.read_csv(val_path)
test_df = pd.read_csv(test_path)

# Create full graph
all_edges = pd.concat([train_df[['source', 'target']], 
                      val_df[['source', 'target']], 
                      test_df[['source', 'target']]]).drop_duplicates()

# Node mapping
all_nodes = pd.unique(all_edges[['source', 'target']].values.ravel())
id_map = {id_: i for i, id_ in enumerate(all_nodes)}
num_nodes = len(all_nodes)

def encode_edges(df):
    return torch.tensor([[id_map[s], id_map[t]] for s, t in zip(df['source'], df['target'])], dtype=torch.long).T

edge_index = encode_edges(all_edges)

# Node features processing
nodes_df = pd.DataFrame(nodes)
nodes_df = nodes_df[nodes_df['id'].isin(id_map)]
nodes_df['node_idx'] = nodes_df['id'].map(id_map)
features_df = nodes_df[['node_idx'] + features_to_use].fillna(0).sort_values('node_idx')

# Feature scaling with robust scaler
scaler = StandardScaler()
x_features = scaler.fit_transform(features_df[features_to_use].values)
x = torch.zeros((num_nodes, len(features_to_use)), dtype=torch.float32)
x[features_df['node_idx'].values] = torch.tensor(x_features, dtype=torch.float32)

# Edge data preparation
def create_edge_label_data(df):
    src = [id_map[s] for s in df['source']]
    dst = [id_map[t] for t in df['target']]
    y = torch.tensor(df['label'].values, dtype=torch.float32)
    edge_idx = torch.tensor([src, dst], dtype=torch.long)
    return edge_idx, y

train_edge_index, train_labels = create_edge_label_data(train_df)
val_edge_index, val_labels = create_edge_label_data(val_df)
test_edge_index, test_labels = create_edge_label_data(test_df)

# Enhanced GCN Model with skip connections
class GCNWithSkip(nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        
        # GCN layers
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.bn3 = nn.BatchNorm1d(hidden_channels)
        
        # Skip connection projections
        self.skip1 = nn.Linear(in_channels, hidden_channels)
        self.skip2 = nn.Linear(hidden_channels, hidden_channels)
        
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        # First GCN layer with skip connection
        x_initial = self.skip1(x)
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x) + x_initial  # Skip connection with ELU activation
        x = self.dropout_layer(x)
        
        # Second GCN layer with skip connection
        x_skip = self.skip2(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.elu(x) + x_skip  # Skip connection
        x = self.dropout_layer(x)
        
        # Third GCN layer
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.elu(x)
        
        return x

# Advanced Link Predictor with GCN
class GCNLinkPredictor(nn.Module):
    def __init__(self, gcn_model, hidden_channels):
        super().__init__()
        self.gcn = gcn_model
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels * 2 + 3, hidden_channels),  # +3 for additional features
            nn.BatchNorm1d(hidden_channels),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.BatchNorm1d(hidden_channels // 2),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_channels // 2, 1)
        )
    
    def forward(self, x, edge_index, edge_pairs):
        node_emb = self.gcn(x, edge_index)
        src = node_emb[edge_pairs[0]]
        dst = node_emb[edge_pairs[1]]
        
        # Multiple similarity features for better link prediction
        dot_product = (src * dst).sum(dim=1, keepdim=True)
        l2_dist = torch.norm(src - dst, p=2, dim=1, keepdim=True)
        cosine_sim = F.cosine_similarity(src, dst, dim=1).unsqueeze(1)
        
        combined = torch.cat([src, dst, dot_product, l2_dist, cosine_sim], dim=1)
        
        return self.mlp(combined).squeeze()

# Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Model initialization with adjusted parameters for GCN
gcn_model = GCNWithSkip(x.size(1), 128, dropout=0.3).to(device)  # Increased hidden size for GCN
model = GCNLinkPredictor(gcn_model, 128).to(device)

x = x.to(device)
edge_index = edge_index.to(device)
train_edge_index, train_labels = train_edge_index.to(device), train_labels.to(device)
val_edge_index, val_labels = val_edge_index.to(device), val_labels.to(device)
test_edge_index, test_labels = test_edge_index.to(device), test_labels.to(device)

# Class weighting for imbalanced data
pos_weight = (len(train_labels) - train_labels.sum()) / train_labels.sum()
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# Optimizer with different learning rates for different components
optimizer = torch.optim.AdamW([
    {'params': model.gcn.parameters(), 'lr': 0.01},   # Higher LR for GCN layers
    {'params': model.mlp.parameters(), 'lr': 0.002}   # Lower LR for MLP
], weight_decay=1e-4)

scheduler = ReduceLROnPlateau(optimizer, 'max', patience=7, factor=0.7, min_lr=1e-6)

# Training loop with early stopping
best_auc = 0
patience = 25
counter = 0

print("Starting training...")
for epoch in range(1, 301):
    model.train()
    optimizer.zero_grad()
    out = model(x, edge_index, train_edge_index)
    loss = criterion(out, train_labels)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    # Evaluation
    if epoch % 2 == 0:  # Evaluate every 2 epochs to speed up training
        model.eval()
        with torch.no_grad():
            # Training metrics
            train_probs = torch.sigmoid(out).cpu().numpy()
            train_auc = roc_auc_score(train_labels.cpu().numpy(), train_probs)
            
            # Validation metrics
            val_out = model(x, edge_index, val_edge_index)
            val_probs = torch.sigmoid(val_out).cpu().numpy()
            val_auc = roc_auc_score(val_labels.cpu().numpy(), val_probs)
            
            # Update scheduler
            scheduler.step(val_auc)
            
            # Check for best model
            if val_auc > best_auc:
                best_auc = val_auc
                counter = 0
                torch.save(model.state_dict(), 'best_gcn_model.pt')
                print(f"Epoch {epoch:03d} | New best model (Val AUC: {val_auc:.4f})")
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        # Print metrics every 10 epochs
        if epoch % 10 == 0:
            train_pred = (train_probs > 0.5).astype(int)
            train_acc = (train_pred == train_labels.cpu().numpy()).mean()
            val_pred = (val_probs > 0.5).astype(int)
            val_acc = (val_pred == val_labels.cpu().numpy()).mean()
            
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | "
                  f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | "
                  f"Train AUC: {train_auc:.4f} | Val AUC: {val_auc:.4f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")

# Load best model and evaluate
print("Loading best model for final evaluation...")
model.load_state_dict(torch.load('best_gcn_model.pt', weights_only=True))
model.eval()
with torch.no_grad():
    test_out = model(x, edge_index, test_edge_index)
    test_probs = torch.sigmoid(test_out).cpu().numpy()
    test_pred = (test_probs > 0.5).astype(int)
    
    test_acc = (test_pred == test_labels.cpu().numpy()).mean()
    test_auc = roc_auc_score(test_labels.cpu().numpy(), test_probs)
    test_f1 = f1_score(test_labels.cpu().numpy(), test_pred)
    
    # Precision-Recall curve metrics
    precision, recall, _ = precision_recall_curve(test_labels.cpu().numpy(), test_probs)
    test_auprc = auc(recall, precision)

print("\n=== Final Test Results (GCN Model) ===")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test AUC-ROC: {test_auc:.4f}")
print(f"Test F1 Score: {test_f1:.4f}")
print(f"Test AUPRC: {test_auprc:.4f}")
print(f"Best Validation AUC: {best_auc:.4f}")

# Model parameter count
total_params = sum(p.numel() for p in model.parameters())
print(f"Total model parameters: {total_params:,}")