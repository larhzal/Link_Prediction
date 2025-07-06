import json
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
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

# Enhanced GraphSAGE Model with skip connections
class GraphSAGEWithSkip(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=3, dropout=0.3, aggr='mean'):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.aggr = aggr
        
        # GraphSAGE layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.skip_connections = nn.ModuleList()
        
        # First layer
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggr))
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        self.skip_connections.append(nn.Linear(in_channels, hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggr))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
            self.skip_connections.append(nn.Linear(hidden_channels, hidden_channels))
        
        # Final layer
        self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggr))
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        self.skip_connections.append(nn.Linear(hidden_channels, hidden_channels))
        
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        for i, (conv, bn, skip) in enumerate(zip(self.convs, self.batch_norms, self.skip_connections)):
            x_skip = skip(x)
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x) + x_skip  # Skip connection with ReLU activation
            
            if i < self.num_layers - 1:  # Don't apply dropout to the last layer
                x = self.dropout_layer(x)
        
        return x

# Advanced Link Predictor with GraphSAGE
class GraphSAGELinkPredictor(nn.Module):
    def __init__(self, sage_model, hidden_channels):
        super().__init__()
        self.sage = sage_model
        
        # Enhanced MLP with more sophisticated architecture
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels * 2 + 4, hidden_channels * 2),  # +4 for additional features
            nn.BatchNorm1d(hidden_channels * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.BatchNorm1d(hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_channels // 2, 1)
        )
    
    def forward(self, x, edge_index, edge_pairs):
        # Get node embeddings from GraphSAGE
        node_emb = self.sage(x, edge_index)
        
        # Extract source and target node embeddings
        src = node_emb[edge_pairs[0]]
        dst = node_emb[edge_pairs[1]]
        
        # Multiple similarity features for comprehensive link prediction
        dot_product = (src * dst).sum(dim=1, keepdim=True)
        l2_dist = torch.norm(src - dst, p=2, dim=1, keepdim=True)
        l1_dist = torch.norm(src - dst, p=1, dim=1, keepdim=True)
        cosine_sim = F.cosine_similarity(src, dst, dim=1).unsqueeze(1)
        
        # Combine all features
        combined = torch.cat([src, dst, dot_product, l2_dist, l1_dist, cosine_sim], dim=1)
        
        return self.mlp(combined).squeeze()

# Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Model initialization with GraphSAGE
sage_model = GraphSAGEWithSkip(
    in_channels=x.size(1), 
    hidden_channels=128,  # Increased hidden dimension for better representation
    num_layers=3, 
    dropout=0.3,
    aggr='mean'  # 'mean', 'max', or 'add' aggregation
).to(device)

model = GraphSAGELinkPredictor(sage_model, 128).to(device)

# Move data to device
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
    {'params': model.sage.parameters(), 'lr': 0.01},   # Higher LR for SAGE layers
    {'params': model.mlp.parameters(), 'lr': 0.005}    # Moderate LR for MLP
], weight_decay=1e-4)

scheduler = ReduceLROnPlateau(optimizer, 'max', patience=10, factor=0.5, min_lr=1e-6)

# Training loop with early stopping
best_auc = 0
patience = 30
counter = 0

print("Starting GraphSAGE training...")
for epoch in range(1, 401):  # Increased epochs for GraphSAGE
    model.train()
    optimizer.zero_grad()
    out = model(x, edge_index, train_edge_index)
    loss = criterion(out, train_labels)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    # Evaluation
    if epoch % 2 == 0:  # Evaluate every 2 epochs
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
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'sage_config': {
                        'in_channels': x.size(1),
                        'hidden_channels': 128,
                        'num_layers': model.sage.num_layers,
                        'dropout': model.sage.dropout,
                        'aggr': model.sage.aggr,
                    },
                    'best_epoch': epoch,
                    'val_auc': val_auc
                }, 'best_graphsage_model.pth')
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
model.load_state_dict(torch.load('best_graphsage_model.pt', weights_only=True))
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

print("\n=== Final Test Results (GraphSAGE Model) ===")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test AUC-ROC: {test_auc:.4f}")
print(f"Test F1 Score: {test_f1:.4f}")
print(f"Test AUPRC: {test_auprc:.4f}")
print(f"Best Validation AUC: {best_auc:.4f}")

# Model parameter count
total_params = sum(p.numel() for p in model.parameters())
print(f"Total model parameters: {total_params:,}")

# Additional GraphSAGE-specific information
print(f"\nGraphSAGE Model Configuration:")
print(f"- Number of layers: {sage_model.num_layers}")
print(f"- Hidden channels: 128")
print(f"- Aggregation function: {sage_model.aggr}")
print(f"- Dropout rate: {sage_model.dropout}")


y_true_sage = test_labels.cpu().numpy()
y_pred_proba_sage = test_probs

# Compute ROC curve
fpr_sage, tpr_sage, _ = roc_curve(y_true_sage, y_pred_proba_sage)
roc_auc_sage = auc(fpr_sage, tpr_sage)

# fpr_sage, tpr_sage, _ = roc_curve(test_labels.cpu().numpy(), test_probs)
# roc_auc_sage = auc(fpr_sage, tpr_sage)

# plot_all_roc_curves(
#     y_true=test_labels.cpu().numpy(),
#     y_pred_sage=y_pred_proba_sage
# )
sage_test_pred = (test_probs > 0.5).astype(int)
true_labels = test_labels.cpu().numpy()
pred_labels = sage_test_pred

cm = confusion_matrix(true_labels, pred_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# Plot ROC curve
plt.figure()
plt.plot(fpr_sage, tpr_sage, color='blue', lw=2, label=f'GraphSAGE ROC (AUC = {roc_auc_sage:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - GraphSAGE')
plt.legend()
plt.grid(True)
plt.show()

def plot_top_predicted_links_sage(probs, edge_index, top_k=50):
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np

    edges = edge_index.cpu().numpy().T
    scores = probs

    top_indices = np.argsort(scores)[-top_k:]
    top_edges = edges[top_indices]
    top_scores = scores[top_indices]

    G = nx.Graph()
    for (u, v), score in zip(top_edges, top_scores):
        G.add_edge(u, v, weight=score)

    pos = nx.spring_layout(G, seed=42)
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]

    # === USE OBJECT-ORIENTED INTERFACE ===
    fig, ax = plt.subplots(figsize=(10, 8))

    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=300, node_color='lightgreen')
    edges = nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color=edge_weights,
        edge_cmap=plt.cm.viridis,
        edge_vmin=min(edge_weights),
        edge_vmax=max(edge_weights),
        width=2
    )
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)

    # Create colorbar linked to edge color mapping
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
    sm.set_array(edge_weights)
    fig.colorbar(sm, ax=ax, label="Predicted Score")

    ax.set_title("Top 50 Predicted Links - GraphSAGE")
    ax.axis('off')
    plt.tight_layout()
    plt.show()


# Call it with your GraphSAGE predictions
plot_top_predicted_links_sage(test_probs, test_edge_index)

def analyze_predictions(probs, threshold=0.5):
    """
    Analyze the distribution of prediction probabilities
    """
    print("📊 Prediction Analysis:")
    print(f"Total predictions: {len(probs)}")
    print(f"Mean probability: {np.mean(probs):.4f}")
    print(f"Median probability: {np.median(probs):.4f}")
    print(f"Std probability: {np.std(probs):.4f}")
    print(f"Min probability: {np.min(probs):.4f}")
    print(f"Max probability: {np.max(probs):.4f}")
    print(f"Predictions above {threshold}: {np.sum(probs > threshold)} ({100*np.sum(probs > threshold)/len(probs):.1f}%)")
    
    # Plot distribution
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.hist(probs, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold ({threshold})')
    plt.xlabel('Prediction Probability')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Probabilities')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    percentiles = [50, 70, 80, 90, 95, 99]
    values = [np.percentile(probs, p) for p in percentiles]
    plt.bar(range(len(percentiles)), values, color='lightcoral', alpha=0.7)
    plt.xlabel('Percentile')
    plt.ylabel('Probability Value')
    plt.title('Prediction Probability Percentiles')
    plt.xticks(range(len(percentiles)), [f'{p}%' for p in percentiles])
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

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

def plot_degree_distribution(G, title="Degree Distribution"):
    """
    Plot the degree distribution of the network
    """
    degrees = [d for n, d in G.degree()]
    
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(degrees, bins=20, alpha=0.7, color='lightblue', edgecolor='black')
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.title(f'{title} - Histogram')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    degree_counts = Counter(degrees)
    degrees_sorted = sorted(degree_counts.keys())
    counts = [degree_counts[d] for d in degrees_sorted]
    
    plt.loglog(degrees_sorted, counts, 'bo-', alpha=0.7)
    plt.xlabel('Degree')
    plt.ylabel('Count')
    plt.title(f'{title} - Log-Log Scale')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_prediction_network_comprehensive(probs, edge_index, num_nodes=200):
    """
    Comprehensive function to analyze and visualize prediction network
    """
    print("🔍 Starting comprehensive network analysis...")
    print("="*70)
    
    # Step 1: Analyze prediction distribution
    analyze_predictions(probs)
    
    # Step 2: Try different methods to create subnetwork
    methods = ['top_predictions', 'mixed', 'high_degree']
    G_main = None
    
    for method in methods:
        print(f"\n🎯 Creating subnetwork using method: {method}")
        try:
            G_main = create_subnetwork_from_predictions(probs, edge_index, num_nodes, method)
            if len(G_main.nodes()) > 0:
                print(f"✅ Successfully created subnetwork with method '{method}'")
                plot_subnetwork(G_main, f"Predicted Link Subnetwork ({method.title()} Method)", 'spring')
                plot_degree_distribution(G_main, f"Degree Distribution ({method.title()} Method)")
                break
        except Exception as e:
            print(f"❌ Method {method} failed: {e}")
            continue
    
    # Step 3: If all methods fail, create a simple version
    if G_main is None or len(G_main.nodes()) == 0:
        print("🔧 Creating simple network from all predictions...")
        edges = edge_index.cpu().numpy().T
        all_nodes = np.unique(edges.flatten())
        
        # Sample nodes if too many
        if len(all_nodes) > num_nodes:
            selected_nodes = np.random.choice(all_nodes, num_nodes, replace=False)
        else:
            selected_nodes = all_nodes
        
        # Create edges between selected nodes
        G_main = nx.Graph()
        for (u, v), score in zip(edges, probs):
            if u in selected_nodes and v in selected_nodes:
                G_main.add_edge(u, v, weight=score)
        
        if len(G_main.nodes()) > 0:
            plot_subnetwork(G_main, "Simple Subnetwork - All Predictions", 'spring')
            plot_degree_distribution(G_main, "Degree Distribution - Simple Method")
    
    # Final summary
    if G_main is not None and len(G_main.nodes()) > 0:
        print("\n📊 Final Network Summary:")
        print(f"- Nodes: {len(G_main.nodes())}")
        print(f"- Edges: {len(G_main.edges())}")
        print(f"- Density: {nx.density(G_main):.4f}")
        print(f"- Avg. Clustering Coefficient: {nx.average_clustering(G_main):.4f}")
        if nx.is_connected(G_main):
            print(f"- Avg. Shortest Path Length: {nx.average_shortest_path_length(G_main):.2f}")
        else:
            print(f"- Connected Components: {nx.number_connected_components(G_main)}")
        
        # Additional network metrics
        print(f"- Max Degree: {max(dict(G_main.degree()).values())}")
        print(f"- Min Degree: {min(dict(G_main.degree()).values())}")
        print(f"- Average Degree: {np.mean(list(dict(G_main.degree()).values())):.2f}")
    else:
        print("⚠️ No valid network could be created from the predictions.")
    
    return G_main
print("\n" + "="*70)
print("🔍 Professional Graph Visualization of Predicted Links")
print("="*70)

# First, analyze what we have
print("🔍 Analyzing prediction results...")

# Use the comprehensive plotting function
G_result = plot_prediction_network_comprehensive(test_probs, test_edge_index, num_nodes=200)

# Keep your existing confusion matrix and ROC curve plots
print("\n" + "="*50)
print("📊 Additional Performance Visualizations")
print("="*50)

# Confusion Matrix
sage_test_pred = (test_probs > 0.5).astype(int)
true_labels = test_labels.cpu().numpy()
pred_labels = sage_test_pred

cm = confusion_matrix(true_labels, pred_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - GraphSAGE")
plt.show()

# ROC Curve
fpr_sage, tpr_sage, _ = roc_curve(true_labels, test_probs)
roc_auc_sage = auc(fpr_sage, tpr_sage)

plt.figure(figsize=(8, 6))
plt.plot(fpr_sage, tpr_sage, color='blue', lw=2, label=f'GraphSAGE ROC (AUC = {roc_auc_sage:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - GraphSAGE Link Prediction')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Optional: Additional network analysis if a graph was successfully created
if G_result is not None and len(G_result.nodes()) > 0:
    print("\n🔬 Advanced Network Analysis:")
    
    # Centrality measures for top nodes
    try:
        betweenness = nx.betweenness_centrality(G_result)
        closeness = nx.closeness_centrality(G_result)
        eigenvector = nx.eigenvector_centrality(G_result, max_iter=1000)
        
        # Top 5 nodes by different centrality measures
        print("\n🏆 Top 5 Most Central Nodes:")
        print("By Betweenness Centrality:")
        for node, score in sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  Node {node}: {score:.4f}")
        
        print("By Closeness Centrality:")
        for node, score in sorted(closeness.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  Node {node}: {score:.4f}")
        
        print("By Eigenvector Centrality:")
        for node, score in sorted(eigenvector.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  Node {node}: {score:.4f}")
            
    except Exception as e:
        print(f"⚠️ Could not compute centrality measures: {e}")

print("\n✅ Visualization pipeline completed successfully!")
print("="*70)