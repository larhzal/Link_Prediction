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
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from roc_pr import plot_all_roc_curves
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter


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

# Enhanced GCN Model with residual connections and advanced techniques
class GCNWithResidual(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=4, dropout=0.3, use_residual=True):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_residual = use_residual
        
        # GCN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.residual_projections = nn.ModuleList()
        
        # Input layer
        self.convs.append(GCNConv(in_channels, hidden_channels, improved=True, cached=False))
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        if self.use_residual:
            self.residual_projections.append(nn.Linear(in_channels, hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, improved=True, cached=False))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
            if self.use_residual:
                self.residual_projections.append(nn.Linear(hidden_channels, hidden_channels))
        
        # Output layer
        self.convs.append(GCNConv(hidden_channels, hidden_channels, improved=True, cached=False))
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        if self.use_residual:
            self.residual_projections.append(nn.Linear(hidden_channels, hidden_channels))
        
        self.dropout_layer = nn.Dropout(dropout)
        
        # Layer normalization for better stability
        self.layer_norm = nn.LayerNorm(hidden_channels)

    def forward(self, x, edge_index):
        # Store initial features for global residual connection
        x_initial = x
        
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            # Store input for residual connection
            x_residual = x
            
            # Apply GCN convolution
            x = conv(x, edge_index)
            x = bn(x)
            
            # Apply residual connection if enabled
            if self.use_residual and i < len(self.residual_projections):
                x_proj = self.residual_projections[i](x_residual)
                x = x + x_proj
            
            # Apply activation and normalization
            x = F.relu(x)
            x = self.layer_norm(x)
            
            # Apply dropout (except for the last layer)
            if i < self.num_layers - 1:
                x = self.dropout_layer(x)
        
        return x

# Advanced Link Predictor with GCN
class GCNLinkPredictor(nn.Module):
    def __init__(self, gcn_model, hidden_channels):
        super().__init__()
        self.gcn = gcn_model
        
        # Multi-scale MLP for link prediction
        self.mlp = nn.Sequential(
            # First block - capture complex interactions
            nn.Linear(hidden_channels * 2 + 5, hidden_channels * 2),  # +5 for similarity features
            nn.BatchNorm1d(hidden_channels * 2),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            # Second block - intermediate representation
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Third block - refined features
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.BatchNorm1d(hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Fourth block - final compression
            nn.Linear(hidden_channels // 2, hidden_channels // 4),
            nn.BatchNorm1d(hidden_channels // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # Output layer
            nn.Linear(hidden_channels // 4, 1)
        )
        
        # Attention mechanism for node pair importance
        self.attention = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, edge_index, edge_pairs):
        # Get node embeddings from GCN
        node_emb = self.gcn(x, edge_index)
        
        # Extract source and target node embeddings
        src = node_emb[edge_pairs[0]]
        dst = node_emb[edge_pairs[1]]
        
        # Comprehensive similarity features
        dot_product = (src * dst).sum(dim=1, keepdim=True)
        l2_dist = torch.norm(src - dst, p=2, dim=1, keepdim=True)
        l1_dist = torch.norm(src - dst, p=1, dim=1, keepdim=True)
        cosine_sim = F.cosine_similarity(src, dst, dim=1).unsqueeze(1)
        
        # Euclidean similarity (inverse of distance)
        euclidean_sim = 1 / (1 + l2_dist)
        
        # Attention-weighted combination
        pair_concat = torch.cat([src, dst], dim=1)
        attention_weights = self.attention(pair_concat)
        
        # Apply attention to source and destination embeddings
        src_weighted = src * attention_weights
        dst_weighted = dst * attention_weights
        
        # Combine all features
        combined = torch.cat([
            src_weighted, dst_weighted, 
            dot_product, l2_dist, l1_dist, cosine_sim, euclidean_sim
        ], dim=1)
        
        return self.mlp(combined).squeeze()

# Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Model initialization with GCN
gcn_model = GCNWithResidual(
    in_channels=x.size(1), 
    hidden_channels=128,  # Optimal hidden dimension for GCN
    num_layers=4,         # Deeper network for better representation
    dropout=0.3,
    use_residual=True     # Enable residual connections
).to(device)

model = GCNLinkPredictor(gcn_model, 128).to(device)

# Move data to device
x = x.to(device)
edge_index = edge_index.to(device)
train_edge_index, train_labels = train_edge_index.to(device), train_labels.to(device)
val_edge_index, val_labels = val_edge_index.to(device), val_labels.to(device)
test_edge_index, test_labels = test_edge_index.to(device), test_labels.to(device)

# Class weighting for imbalanced data
pos_weight = (len(train_labels) - train_labels.sum()) / train_labels.sum()
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# Optimizer with layer-specific learning rates
optimizer = torch.optim.AdamW([
    {'params': model.gcn.parameters(), 'lr': 0.01, 'weight_decay': 1e-4},
    {'params': model.mlp.parameters(), 'lr': 0.005, 'weight_decay': 1e-3},
    {'params': model.attention.parameters(), 'lr': 0.005, 'weight_decay': 1e-3}
])

# Cosine annealing scheduler for better convergence
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=20, T_mult=2, eta_min=1e-6
)

# Training loop with advanced early stopping
best_auc = 0
patience = 35
counter = 0
best_epoch = 0

print("Starting GCN training...")
for epoch in range(1, 501):  # More epochs for GCN convergence
    model.train()
    optimizer.zero_grad()
    out = model(x, edge_index, train_edge_index)
    loss = criterion(out, train_labels)
    loss.backward()
    
    # Gradient clipping for stability
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()

    # Evaluation every 2 epochs
    if epoch % 2 == 0:
        model.eval()
        with torch.no_grad():
            # Training metrics
            train_probs = torch.sigmoid(out).cpu().numpy()
            train_auc = roc_auc_score(train_labels.cpu().numpy(), train_probs)
            
            # Validation metrics
            val_out = model(x, edge_index, val_edge_index)
            val_probs = torch.sigmoid(val_out).cpu().numpy()
            val_auc = roc_auc_score(val_labels.cpu().numpy(), val_probs)
            
            # Check for best model
            if val_auc > best_auc:
                best_auc = val_auc
                best_epoch = epoch
                counter = 0
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'gcn_config': {
                        'in_channels': x.size(1),
                        'hidden_channels': 128,
                        'num_layers': gcn_model.num_layers,
                        'dropout': gcn_model.dropout,
                        'use_residual': gcn_model.use_residual,
                    },
                    'best_epoch': epoch,
                    'val_auc': val_auc
                }, 'best_gcn_model.pth')
                print(f"Epoch {epoch:03d} | New best model (Val AUC: {val_auc:.4f})")
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping at epoch {epoch} (Best epoch: {best_epoch})")
                    break

        # Print detailed metrics every 10 epochs
        if epoch % 10 == 0:
            train_pred = (train_probs > 0.5).astype(int)
            train_acc = (train_pred == train_labels.cpu().numpy()).mean()
            train_f1 = f1_score(train_labels.cpu().numpy(), train_pred)
            
            val_pred = (val_probs > 0.5).astype(int)
            val_acc = (val_pred == val_labels.cpu().numpy()).mean()
            val_f1 = f1_score(val_labels.cpu().numpy(), val_pred)
            
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | "
                  f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | "
                  f"Train AUC: {train_auc:.4f} | Val AUC: {val_auc:.4f} | "
                  f"Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")

# Load best model and comprehensive evaluation
print("Loading best model for final evaluation...")
model.load_state_dict(torch.load('best_gcn_model.pt', weights_only=True))
model.eval()

with torch.no_grad():
    # Test predictions
    test_out = model(x, edge_index, test_edge_index)
    test_probs = torch.sigmoid(test_out).cpu().numpy()
    test_pred = (test_probs > 0.5).astype(int)
    
    # Comprehensive metrics
    test_acc = (test_pred == test_labels.cpu().numpy()).mean()
    test_auc = roc_auc_score(test_labels.cpu().numpy(), test_probs)
    test_f1 = f1_score(test_labels.cpu().numpy(), test_pred)
    
    # Precision-Recall curve metrics
    precision, recall, _ = precision_recall_curve(test_labels.cpu().numpy(), test_probs)
    test_auprc = auc(recall, precision)
    
    # Additional metrics
    from sklearn.metrics import precision_score, recall_score
    test_precision = precision_score(test_labels.cpu().numpy(), test_pred)
    test_recall = recall_score(test_labels.cpu().numpy(), test_pred)

print("\n=== Final Test Results (GCN Model) ===")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test AUC-ROC: {test_auc:.4f}")
print(f"Test F1 Score: {test_f1:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")
print(f"Test AUPRC: {test_auprc:.4f}")
print(f"Best Validation AUC: {best_auc:.4f}")
print(f"Best Epoch: {best_epoch}")

# Model parameter count and architecture info
total_params = sum(p.numel() for p in model.parameters())
gcn_params = sum(p.numel() for p in model.gcn.parameters())
mlp_params = sum(p.numel() for p in model.mlp.parameters())
attention_params = sum(p.numel() for p in model.attention.parameters())

print(f"\nModel Architecture Details:")
print(f"Total parameters: {total_params:,}")
print(f"GCN parameters: {gcn_params:,}")
print(f"MLP parameters: {mlp_params:,}")
print(f"Attention parameters: {attention_params:,}")

print(f"\nGCN Model Configuration:")
print(f"- Number of GCN layers: {gcn_model.num_layers}")
print(f"- Hidden channels: 128")
print(f"- Residual connections: {gcn_model.use_residual}")
print(f"- Dropout rate: {gcn_model.dropout}")
print(f"- Improved GCN: True (Kipf & Welling enhancement)")
y_true = test_labels.cpu().numpy()
y_pred_proba = test_probs

# Compute ROC curve
fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
roc_auc = auc(fpr, tpr)

# fpr_gcn, tpr_gcn, _ = roc_curve(test_labels.cpu().numpy(), test_probs)
# roc_auc_gcn = auc(fpr_gcn, tpr_gcn)

# plot_all_roc_curves(
#     y_true=test_labels.cpu().numpy(),
#     y_pred_gcn=y_pred_proba_gcn,
# )

# gcn_test_pred = (test_probs > 0.5).astype(int)
# true_labels = test_labels.cpu().numpy()
# pred_labels = gcn_test_pred

# cm = confusion_matrix(true_labels, pred_labels)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm)
# disp.plot(cmap=plt.cm.Greens)
# plt.title("Confusion Matrix - GCN")
# plt.show()

# # # Plot ROC curve
# plt.figure()
# plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
# plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve')
# plt.legend()
# plt.grid(True)
# plt.show()





# def plot_high_confidence_predictions(probs, edge_index, threshold=0.8):
#     """
#     Plot only high-confidence predictions to see clear patterns
#     """
#     probs = np.asarray(probs)
#     edges = edge_index.cpu().numpy().T
    
#     # Filter high-confidence predictions
#     high_conf_mask = probs >= threshold
#     high_conf_edges = edges[high_conf_mask]
#     high_conf_scores = probs[high_conf_mask]
    
#     if len(high_conf_edges) == 0:
#         print(f"⚠️ No predictions above threshold {threshold}")
#         return
    
#     print(f"🎯 Found {len(high_conf_edges)} high-confidence predictions (≥{threshold})")
    
#     # Build network
#     G = nx.Graph()
#     for (u, v), score in zip(high_conf_edges, high_conf_scores):
#         G.add_edge(u, v, weight=score)
    
#     # Layout
#     pos = nx.spring_layout(G, seed=42, k=1.5)
    
#     plt.figure(figsize=(12, 10))
    
#     # Node colors based on degree
#     node_colors = [G.degree(n) for n in G.nodes()]
    
#     # Draw network
#     nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=400, 
#                           cmap=plt.cm.viridis, alpha=0.8)
    
#     # Edge colors based on prediction confidence
#     edge_colors = [G[u][v]['weight'] for u, v in G.edges()]
#     nx.draw_networkx_edges(G, pos, edge_color=edge_colors, edge_cmap=plt.cm.Reds, 
#                           width=2, alpha=0.8)
    
#     # Labels for important nodes (high degree)
#     important_nodes = [n for n in G.nodes() if G.degree(n) >= 3]
#     labels = {n: str(n) for n in important_nodes}
#     nx.draw_networkx_labels(G, pos, labels, font_size=8)
    
#     plt.title(f"High-Confidence Predictions (≥{threshold})\n{len(G.nodes())} nodes, {len(G.edges())} edges")
#     plt.axis('off')
#     plt.tight_layout()
#     plt.show()

def analyze_predictions(probs, labels=None):
    """
    Analyze the distribution of prediction probabilities
    """
    probs = np.asarray(probs)
    
    print("🔍 Prediction Analysis:")
    print(f"Total predictions: {len(probs)}")
    print(f"Min probability: {probs.min():.4f}")
    print(f"Max probability: {probs.max():.4f}")
    print(f"Mean probability: {probs.mean():.4f}")
    print(f"Median probability: {np.median(probs):.4f}")
    
    # Distribution at different thresholds
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    print("\n📊 Predictions above threshold:")
    for thresh in thresholds:
        count = np.sum(probs >= thresh)
        percentage = count / len(probs) * 100
        print(f"≥{thresh}: {count:4d} ({percentage:5.1f}%)")
    
    # Plot histogram
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.hist(probs, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Prediction Probability')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Probabilities')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.hist(probs, bins=50, alpha=0.7, color='skyblue', edgecolor='black', cumulative=True, density=True)
    plt.xlabel('Prediction Probability')
    plt.ylabel('Cumulative Density')
    plt.title('Cumulative Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return probs

def create_subnetwork_from_predictions(probs, edge_index, num_nodes=200, method='top_predictions'):
    """
    Create a subnetwork of specified size from predictions
    
    Parameters:
    - probs: prediction probabilities
    - edge_index: edge indices [2, num_edges]
    - num_nodes: target number of nodes in subnetwork
    - method: 'top_predictions', 'high_degree', or 'mixed'
    """
    probs = np.asarray(probs)
    edges = edge_index.cpu().numpy().T
    
    if method == 'top_predictions':
        # Select top predictions regardless of threshold
        top_k = min(1000, len(probs))  # Start with top 1000 edges
        top_indices = np.argsort(probs)[-top_k:]
        selected_edges = edges[top_indices]
        selected_probs = probs[top_indices]
        
    elif method == 'high_degree':
        # Select edges involving high-degree nodes
        node_degrees = Counter()
        for u, v in edges:
            node_degrees[u] += 1
            node_degrees[v] += 1
        
        # Get top degree nodes
        top_nodes = set([node for node, _ in node_degrees.most_common(num_nodes)])
        
        # Select edges involving these nodes
        mask = np.array([u in top_nodes or v in top_nodes for u, v in edges])
        selected_edges = edges[mask]
        selected_probs = probs[mask]
        
    elif method == 'mixed':
        # Combine high confidence and high degree
        # First, get edges with reasonable confidence
        conf_mask = probs >= np.percentile(probs, 70)  # Top 30% of predictions
        conf_edges = edges[conf_mask]
        conf_probs = probs[conf_mask]
        
        # Build degree count from confident edges
        node_degrees = Counter()
        for u, v in conf_edges:
            node_degrees[u] += 1
            node_degrees[v] += 1
        
        # Select top degree nodes from confident edges
        top_nodes = set([node for node, _ in node_degrees.most_common(num_nodes)])
        
        # Select edges involving these nodes
        mask = np.array([u in top_nodes or v in top_nodes for u, v in conf_edges])
        selected_edges = conf_edges[mask]
        selected_probs = conf_probs[mask]
    
    # Build the network
    G = nx.Graph()
    for (u, v), score in zip(selected_edges, selected_probs):
        if not G.has_edge(u, v):  # Avoid duplicate edges
            G.add_edge(u, v, weight=score)
    
    # If we have too many nodes, keep only the most connected ones
    if len(G.nodes()) > num_nodes:
        # Sort nodes by degree and keep top ones
        node_degrees = dict(G.degree())
        top_nodes = sorted(node_degrees.keys(), key=lambda x: node_degrees[x], reverse=True)[:num_nodes]
        G = G.subgraph(top_nodes).copy()
    
    print(f"📊 Created subnetwork with {len(G.nodes())} nodes and {len(G.edges())} edges")
    print(f"Average degree: {2 * len(G.edges()) / len(G.nodes()):.2f}")
    
    return G

def plot_subnetwork(G, title="Predicted Link Subnetwork", layout='spring', figsize=(15, 12)):
    """
    Plot the subnetwork with enhanced visualization
    """
    if len(G.nodes()) == 0:
        print("❌ Empty network - nothing to plot!")
        return
    
    plt.figure(figsize=figsize)
    
    # Choose layout
    if layout == 'spring':
        pos = nx.spring_layout(G, seed=42, k=2, iterations=50)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42)
    
    # Node properties
    node_degrees = dict(G.degree())
    node_sizes = [min(1000, 100 + node_degrees[node] * 50) for node in G.nodes()]
    node_colors = [node_degrees[node] for node in G.nodes()]
    
    # Edge properties
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    edge_colors = edge_weights
    edge_widths = [1 + 3 * weight for weight in edge_weights]
    
    # Draw network
    nodes = nx.draw_networkx_nodes(G, pos, 
                                  node_size=node_sizes,
                                  node_color=node_colors,
                                  cmap=plt.cm.viridis,
                                  alpha=0.8)
    
    edges = nx.draw_networkx_edges(G, pos,
                                  edge_color=edge_colors,
                                  edge_cmap=plt.cm.Reds,
                                  width=edge_widths,
                                  alpha=0.6)
    
    # Add labels for highly connected nodes
    high_degree_nodes = [n for n, d in node_degrees.items() if d >= np.percentile(list(node_degrees.values()), 90)]
    labels = {n: str(n) for n in high_degree_nodes}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold')
    
    # Add colorbars
    plt.colorbar(nodes, label='Node Degree', shrink=0.8)
    plt.colorbar(edges, label='Prediction Score', shrink=0.8)
    
    plt.title(f"{title}\n{len(G.nodes())} nodes, {len(G.edges())} edges")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    # Print network statistics
    print(f"\n📈 Network Statistics:")
    print(f"Nodes: {len(G.nodes())}")
    print(f"Edges: {len(G.edges())}")
    print(f"Density: {nx.density(G):.4f}")
    print(f"Average clustering: {nx.average_clustering(G):.4f}")
    
    if nx.is_connected(G):
        print(f"Average shortest path: {nx.average_shortest_path_length(G):.2f}")
    else:
        print(f"Connected components: {nx.number_connected_components(G)}")

def plot_prediction_network_comprehensive(probs, edge_index, num_nodes=200):
    """
    Comprehensive function to analyze and visualize prediction network
    """
    print("🔍 Starting comprehensive network analysis...")
    
    # Step 1: Analyze prediction distribution
    analyze_predictions(probs)
    
    # Step 2: Try different methods to create subnetwork
    methods = ['top_predictions', 'mixed', 'high_degree']
    
    for method in methods:
        print(f"\n🎯 Creating subnetwork using method: {method}")
        try:
            G = create_subnetwork_from_predictions(probs, edge_index, num_nodes, method)
            if len(G.nodes()) > 0:
                plot_subnetwork(G, f"Subnetwork - {method.title()} Method", 'spring')
                break
        except Exception as e:
            print(f"❌ Method {method} failed: {e}")
            continue
    
    # Step 3: If all methods fail, create a simple version
    if len(G.nodes()) == 0:
        print("🔧 Creating simple network from all predictions...")
        edges = edge_index.cpu().numpy().T
        all_nodes = np.unique(edges.flatten())
        
        # Sample nodes if too many
        if len(all_nodes) > num_nodes:
            selected_nodes = np.random.choice(all_nodes, num_nodes, replace=False)
        else:
            selected_nodes = all_nodes
        
        # Create edges between selected nodes
        G = nx.Graph()
        for (u, v), score in zip(edges, probs):
            if u in selected_nodes and v in selected_nodes:
                G.add_edge(u, v, weight=score)
        
        if len(G.nodes()) > 0:
            plot_subnetwork(G, "Simple Subnetwork - All Predictions", 'spring')

# Replace your current plotting code with this:

# First, let's analyze what we have
print("🔍 Analyzing prediction results...")
# === Clean Visualization Pipeline ===
print("\n" + "="*70)
print("🔍 Professional Graph Visualization of Predicted Links")
print("="*70)

# Step 1: Analyze Prediction Distribution
analyze_predictions(test_probs)

# Step 2: Create and plot only ONE subnetwork using best method
G_main = None
for method in ['top_predictions', 'mixed', 'high_degree']:
    print(f"\n🧠 Trying method: {method}...")
    try:
        G_main = create_subnetwork_from_predictions(test_probs, test_edge_index, num_nodes=200, method=method)
        if len(G_main.nodes()) > 0:
            print(f"✅ Subnetwork built with method '{method}'")
            plot_subnetwork(G_main, f"Predicted Link Subnetwork ({method.title()} Method)", layout='spring')
            break
    except Exception as e:
        print(f"❌ Method {method} failed: {e}")

if G_main is None or len(G_main.nodes()) == 0:
    print("⚠️ All methods failed or empty network generated.")
else:
    print("\n📊 Final Network Summary:")
    print(f"- Nodes: {len(G_main.nodes())}")
    print(f"- Edges: {len(G_main.edges())}")
    print(f"- Density: {nx.density(G_main):.4f}")
    print(f"- Avg. Clustering Coefficient: {nx.average_clustering(G_main):.4f}")
    if nx.is_connected(G_main):
        print(f"- Avg. Shortest Path Length: {nx.average_shortest_path_length(G_main):.2f}")
    else:
        print(f"- Connected Components: {nx.number_connected_components(G_main)}")
