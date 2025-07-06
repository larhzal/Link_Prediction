import torch
import torch.nn.functional as F
import json
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, auc
from models import GATWithSkip, GATLinkPredictor, GraphSAGEWithSkip, GraphSAGELinkPredictor, GCNNet, GCNLinkPredictor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------- Load features and edges ----------
def load_data():
    with open('json_files/reduced_nodes_connected.json') as f:
        nodes = json.load(f)
    test_df = pd.read_csv("data/test_links.csv")
    all_edges = pd.read_csv("data/all_edges.csv")  # optional: or reconstruct from train/val/test

    feature_keys = ['paper_count', 'citation_count', 'h_index', 'p_index_eq', 'p_index_uneq', 
                    'coauthor_count', 'venue_count', 'recent_paper_count']
    id_map = {id_: i for i, id_ in enumerate(pd.unique(all_edges[['source', 'target']].values.ravel()))}
    num_nodes = len(id_map)

    # Process features
    df = pd.DataFrame(nodes)
    df = df[df['id'].isin(id_map)]
    df['node_idx'] = df['id'].map(id_map)
    df = df.sort_values('node_idx')
    x = np.stack([df.get(key, pd.Series(0, index=df.index)).fillna(0).values for key in feature_keys], axis=1)
    x_tensor = torch.tensor(x, dtype=torch.float32)
    full_x = torch.zeros((num_nodes, x.shape[1]))
    full_x[df['node_idx'].values] = x_tensor

    # Process edge index
    edge_index = torch.tensor([
        [id_map[s] for s in all_edges['source']],
        [id_map[t] for t in all_edges['target']]
    ], dtype=torch.long)

    # Test edge index
    test_edges = torch.tensor([
        [id_map[s] for s in test_df['source']],
        [id_map[t] for t in test_df['target']]
    ], dtype=torch.long)
    test_labels = torch.tensor(test_df['label'].values, dtype=torch.float32)

    return full_x.to(device), edge_index.to(device), test_edges.to(device), test_labels.to(device)

# ---------- Evaluate function ----------
def evaluate(model, x, edge_index, edge_pairs, labels):
    model.eval()
    with torch.no_grad():
        out = model(x, edge_index, edge_pairs)
        probs = torch.sigmoid(out).cpu().numpy()
        preds = (probs > 0.5).astype(int)
        labels_np = labels.cpu().numpy()

        acc = (preds == labels_np).mean()
        auc_roc = roc_auc_score(labels_np, probs)
        f1 = f1_score(labels_np, preds)
        precision, recall, _ = precision_recall_curve(labels_np, probs)
        auprc = auc(recall, precision)
    return acc, auc_roc, f1, auprc

# ---------- Load and evaluate all models ----------
def run_evaluation():
    x, edge_index, test_edge_index, test_labels = load_data()

    # ==== GCN ====
    print("\n🔷 Evaluating GCN...")
    gcn_checkpoint = torch.load('best_gcn_model.pth')
    gcn_config = gcn_checkpoint['gcn_config']
    gcn_backbone = GCNNet(**gcn_config).to(device)
    gcn_model = GCNLinkPredictor(gcn_backbone, gcn_config['hidden_channels']).to(device)
    gcn_model.load_state_dict(gcn_checkpoint['model_state_dict'])
    print("✅ GCN loaded.")
    acc, auc_score, f1, auprc = evaluate(gcn_model, x, edge_index, test_edge_index, test_labels)
    print(f"GCN: Acc={acc:.4f}, AUC={auc_score:.4f}, F1={f1:.4f}, AUPRC={auprc:.4f}")

    # ==== GAT ====
    print("\n🟣 Evaluating GAT...")
    gat_backbone = GATWithSkip(x.size(1), 64, heads=8, dropout=0.3).to(device)
    gat_model = GATLinkPredictor(gat_backbone, 64).to(device)
    gat_model.load_state_dict(torch.load('best_gat_model.pt'))
    print("✅ GAT loaded.")
    acc, auc_score, f1, auprc = evaluate(gat_model, x, edge_index, test_edge_index, test_labels)
    print(f"GAT: Acc={acc:.4f}, AUC={auc_score:.4f}, F1={f1:.4f}, AUPRC={auprc:.4f}")

    # ==== GraphSAGE ====
    print("\n🟢 Evaluating GraphSAGE...")
    sage_backbone = GraphSAGEWithSkip(in_channels=x.size(1), hidden_channels=128, num_layers=3).to(device)
    sage_model = GraphSAGELinkPredictor(sage_backbone, 128).to(device)
    sage_model.load_state_dict(torch.load('best_graphsage_model.pt'))
    print("✅ GraphSAGE loaded.")
    acc, auc_score, f1, auprc = evaluate(sage_model, x, edge_index, test_edge_index, test_labels)
    print(f"GraphSAGE: Acc={acc:.4f}, AUC={auc_score:.4f}, F1={f1:.4f}, AUPRC={auprc:.4f}")

if __name__ == "__main__":
    run_evaluation()
