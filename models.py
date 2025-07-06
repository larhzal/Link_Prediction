import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv

# === GCN ===
class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=2, dropout=0.5, use_residual=False):
        super().__init__()
        self.convs = nn.ModuleList()
        self.dropout = dropout
        self.num_layers = num_layers
        self.use_residual = use_residual

        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x_res = x
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.use_residual and i > 0:
                x = x + x_res
        return x

class GCNLinkPredictor(nn.Module):
    def __init__(self, gcn_model, hidden_channels):
        super().__init__()
        self.gcn = gcn_model
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels * 2 + 3, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels, 1)
        )

    def forward(self, x, edge_index, edge_pairs):
        node_emb = self.gcn(x, edge_index)
        src = node_emb[edge_pairs[0]]
        dst = node_emb[edge_pairs[1]]
        dot = (src * dst).sum(dim=1, keepdim=True)
        l2 = torch.norm(src - dst, p=2, dim=1, keepdim=True)
        cos = F.cosine_similarity(src, dst, dim=1).unsqueeze(1)
        combined = torch.cat([src, dst, dot, l2, cos], dim=1)
        return self.mlp(combined).squeeze()

# === GAT ===
class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, heads=8, dropout=0.3):
        super().__init__()
        self.num_layers = 3
        self.hidden_channels = hidden_channels
        self.aggregation = "attention"
        self.dropout = dropout

        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout, concat=True)
        self.bn1 = nn.BatchNorm1d(hidden_channels * heads)

        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout, concat=True)
        self.bn2 = nn.BatchNorm1d(hidden_channels * heads)

        self.conv3 = GATConv(hidden_channels * heads, hidden_channels, heads=1, dropout=dropout, concat=False)

        self.skip1 = nn.Linear(in_channels, hidden_channels * heads)
        self.skip2 = nn.Linear(hidden_channels * heads, hidden_channels)

        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x_initial = self.skip1(x)
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x) + x_initial
        x = self.dropout_layer(x)

        x_skip = self.skip2(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.dropout_layer(x)

        x = self.conv3(x, edge_index) + x_skip
        return x

class GATLinkPredictor(nn.Module):
    def __init__(self, gat_model, hidden_channels):
        super().__init__()
        self.gat = gat_model
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels * 2 + 3, hidden_channels),
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
        node_emb = self.gat(x, edge_index)
        src = node_emb[edge_pairs[0]]
        dst = node_emb[edge_pairs[1]]
        dot_product = (src * dst).sum(dim=1, keepdim=True)
        l2_dist = torch.norm(src - dst, p=2, dim=1, keepdim=True)
        cosine_sim = F.cosine_similarity(src, dst, dim=1).unsqueeze(1)
        combined = torch.cat([src, dst, dot_product, l2_dist, cosine_sim], dim=1)
        return self.mlp(combined).squeeze()

# === GraphSAGE ===
class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=3, dropout=0.3, aggr='mean'):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.aggr = aggr

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.skip_connections = nn.ModuleList()

        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggr))
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        self.skip_connections.append(nn.Linear(in_channels, hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggr))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
            self.skip_connections.append(nn.Linear(hidden_channels, hidden_channels))

        self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggr))
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        self.skip_connections.append(nn.Linear(hidden_channels, hidden_channels))

        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        for i, (conv, bn, skip) in enumerate(zip(self.convs, self.batch_norms, self.skip_connections)):
            x_skip = skip(x)
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x) + x_skip
            if i < self.num_layers - 1:
                x = self.dropout_layer(x)
        return x

class GraphSAGELinkPredictor(nn.Module):
    def __init__(self, sage_model, hidden_channels):
        super().__init__()
        self.sage = sage_model
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels * 2 + 4, hidden_channels * 2),
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
        node_emb = self.sage(x, edge_index)
        src = node_emb[edge_pairs[0]]
        dst = node_emb[edge_pairs[1]]
        dot = (src * dst).sum(dim=1, keepdim=True)
        l2 = torch.norm(src - dst, p=2, dim=1, keepdim=True)
        l1 = torch.norm(src - dst, p=1, dim=1, keepdim=True)
        cos = F.cosine_similarity(src, dst, dim=1).unsqueeze(1)
        combined = torch.cat([src, dst, dot, l2, l1, cos], dim=1)
        return self.mlp(combined).squeeze()
