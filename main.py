"""
Cresci-2017 Bot Detection with Graph Neural Networks
=====================================================
- Dataset: Simulated Cresci-2017 (~5000 users, 5 features)
- Features: followers_count, friends_count, tweet_frequency, url_ratio, sentiment_score
- Graph: Homogeneous user graph with co-mention edges
- Models: 2-layer GIN (hidden=64), GCN, GraphSAGE
- Evaluation: F1 score on 10% test set
- Visualization: GIN first-layer neighbor aggregation patterns
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GINConv, GCNConv, SAGEConv
from torch_geometric.utils import to_networkx
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import random
import os

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_DIR = 'outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# 1. Data Simulation (Cresci-2017 Style)
# =============================================================================

def simulate_cresci2017(n_users=5000, bot_ratio=0.4):
    """
    Simulate Cresci-2017 dataset characteristics.
    Genuine users: ~3000, Bots: ~2000
    Features: followers_count, friends_count, tweet_frequency, url_ratio, sentiment_score
    """
    n_bots = int(n_users * bot_ratio)
    n_genuine = n_users - n_bots

    # Genuine user features
    genuine_followers = np.random.lognormal(mean=5.5, sigma=1.5, size=n_genuine).clip(0, 1e6)
    genuine_friends = np.random.lognormal(mean=4.5, sigma=1.2, size=n_genuine).clip(0, 5e5)
    genuine_tweet_freq = np.random.exponential(scale=3.0, size=n_genuine).clip(0, 50)
    genuine_url_ratio = np.random.beta(a=2, b=8, size=n_genuine)
    genuine_sentiment = np.random.normal(loc=0.1, scale=0.3, size=n_genuine).clip(-1, 1)

    # Bot features (different distributions)
    bot_followers = np.random.lognormal(mean=3.0, sigma=2.0, size=n_bots).clip(0, 1e5)
    bot_friends = np.random.lognormal(mean=6.0, sigma=1.0, size=n_bots).clip(0, 5e5)
    bot_tweet_freq = np.random.exponential(scale=15.0, size=n_bots).clip(0, 200)
    bot_url_ratio = np.random.beta(a=5, b=3, size=n_bots)
    bot_sentiment = np.random.normal(loc=-0.05, scale=0.15, size=n_bots).clip(-1, 1)

    # Combine
    followers = np.concatenate([genuine_followers, bot_followers])
    friends = np.concatenate([genuine_friends, bot_friends])
    tweet_freq = np.concatenate([genuine_tweet_freq, bot_tweet_freq])
    url_ratio = np.concatenate([genuine_url_ratio, bot_url_ratio])
    sentiment = np.concatenate([genuine_sentiment, bot_sentiment])
    labels = np.array([0] * n_genuine + [1] * n_bots)

    # Shuffle
    indices = np.random.permutation(n_users)
    features = np.column_stack([followers, friends, tweet_freq, url_ratio, sentiment])[indices]
    labels = labels[indices]

    # Normalize features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    return features, labels, indices


def build_comention_graph(n_users=5000, n_edges=15000, labels=None):
    """
    Build co-mention graph: edge (u, v) if users u and v mentioned same entity.
    Bots tend to co-mention with other bots (homophily).
    """
    edges = set()
    bot_indices = np.where(labels == 1)[0]
    genuine_indices = np.where(labels == 0)[0]

    # Intra-bot edges (bots co-mention each other more)
    n_bot_edges = int(n_edges * 0.4)
    while len(edges) < n_bot_edges:
        u, v = np.random.choice(bot_indices, 2, replace=False)
        if u != v:
            edge = (min(u, v), max(u, v))
            if edge not in edges:
                edges.add(edge)

    # Intra-genuine edges
    n_genuine_edges = int(n_edges * 0.35)
    while len(edges) < n_bot_edges + n_genuine_edges:
        u, v = np.random.choice(genuine_indices, 2, replace=False)
        if u != v:
            edge = (min(u, v), max(u, v))
            if edge not in edges:
                edges.add(edge)

    # Cross edges
    while len(edges) < n_edges:
        u = np.random.choice(n_users)
        v = np.random.choice(n_users)
        if u != v:
            edge = (min(u, v), max(u, v))
            if edge not in edges:
                edges.add(edge)

    edge_list = list(edges)
    src = [e[0] for e in edge_list]
    dst = [e[1] for e in edge_list]
    # Undirected: add both directions
    edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)
    return edge_index


# =============================================================================
# 2. Model Definitions
# =============================================================================

class GINNet(nn.Module):
    """2-layer Graph Isomorphism Network, hidden_dim=64"""

    def __init__(self, in_channels, hidden_channels=64, num_classes=2):
        super().__init__()
        mlp1 = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        mlp2 = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.conv1 = GINConv(mlp1, train_eps=True)
        self.conv2 = GINConv(mlp2, train_eps=True)
        self.classifier = nn.Linear(hidden_channels, num_classes)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)

    def forward(self, x, edge_index):
        h1 = self.conv1(x, edge_index)
        h1 = self.bn1(h1)
        h1 = F.relu(h1)
        h2 = self.conv2(h1, edge_index)
        h2 = self.bn2(h2)
        h2 = F.relu(h2)
        out = self.classifier(h2)
        return out

    def get_first_layer_output(self, x, edge_index):
        """Get first layer aggregation output for visualization"""
        h1 = self.conv1(x, edge_index)
        return h1


class GCNNet(nn.Module):
    """2-layer GCN, hidden_dim=64"""

    def __init__(self, in_channels, hidden_channels=64, num_classes=2):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.conv2(h, edge_index)
        h = F.relu(h)
        out = self.classifier(h)
        return out


class GraphSAGENet(nn.Module):
    """2-layer GraphSAGE, hidden_dim=64"""

    def __init__(self, in_channels, hidden_channels=64, num_classes=2):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.conv2(h, edge_index)
        h = F.relu(h)
        out = self.classifier(h)
        return out


# =============================================================================
# 3. Training and Evaluation
# =============================================================================

def train_model(model, data, train_mask, val_mask, epochs=200, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    best_val_f1 = 0
    best_state = None
    patience = 30
    no_improve = 0
    best_epoch = 0
    best_train_loss = 0.0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_out = model(data.x, data.edge_index)
            val_pred = val_out[val_mask].argmax(dim=1).cpu().numpy()
            val_true = data.y[val_mask].cpu().numpy()
            val_f1 = f1_score(val_true, val_pred, average='macro')

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_train_loss = loss.item()
            best_epoch = epoch + 1
            best_state = model.state_dict().copy()
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}: Loss={loss.item():.4f}, Val F1={val_f1:.4f}")

    model.load_state_dict(best_state)
    print(f"  Best @ Epoch {best_epoch}: Train Loss={best_train_loss:.4f}, Val F1={best_val_f1:.4f}")
    return model


def evaluate_model(model, data, test_mask):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out[test_mask].argmax(dim=1).cpu().numpy()
        true = data.y[test_mask].cpu().numpy()
    f1_macro = f1_score(true, pred, average='macro')
    f1_per_class = f1_score(true, pred, average=None)
    report = classification_report(true, pred, target_names=['Genuine', 'Bot'])
    return f1_macro, f1_per_class, report


# =============================================================================
# 4. Visualization: GIN First-Layer Neighbor Aggregation
# =============================================================================

def visualize_gin_aggregation(model, data, n_samples=20):
    """
    Visualize how GIN first layer aggregates neighbor information.
    Shows the embedding patterns of sampled nodes and their neighborhoods.
    """
    model.eval()
    with torch.no_grad():
        h1 = model.get_first_layer_output(data.x, data.edge_index).cpu().numpy()

    labels = data.y.cpu().numpy()
    n_nodes = len(labels)
    edge_index_np = data.edge_index.cpu().numpy()

    # Precompute adjacency list from edge_index (bidirectional)
    adj_list = [[] for _ in range(n_nodes)]
    for i in range(edge_index_np.shape[1]):
        src, dst = edge_index_np[0, i], edge_index_np[1, i]
        adj_list[src].append(dst)
    # Deduplicate neighbors
    adj_list = [np.unique(np.array(neighbors)) if neighbors else np.array([], dtype=int)
                for neighbors in adj_list]
    degrees = np.array([len(adj_list[i]) for i in range(n_nodes)])

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # (a) t-SNE-like 2D projection of first layer embeddings
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    h1_2d = pca.fit_transform(h1)

    ax = axes[0, 0]
    colors = ['#2ecc71' if l == 0 else '#e74c3c' for l in labels]
    ax.scatter(h1_2d[:, 0], h1_2d[:, 1], c=colors, alpha=0.3, s=8)
    ax.set_title('GIN Layer-1 Embeddings (PCA)', fontsize=12)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#2ecc71', label='Genuine'),
                       Patch(facecolor='#e74c3c', label='Bot')]
    ax.legend(handles=legend_elements, loc='upper right')

    # (b) Neighbor label distribution for sample nodes
    ax = axes[0, 1]
    sample_bots = np.random.choice(np.where(labels == 1)[0], n_samples // 2, replace=False)
    sample_genuine = np.random.choice(np.where(labels == 0)[0], n_samples // 2, replace=False)
    sample_nodes = np.concatenate([sample_genuine, sample_bots])

    neighbor_stats = []
    for node in sample_nodes:
        neighbors = adj_list[node]
        if len(neighbors) == 0:
            neighbor_stats.append((0, 0))
            continue
        n_bot_neighbors = (labels[neighbors] == 1).sum()
        n_genuine_neighbors = (labels[neighbors] == 0).sum()
        neighbor_stats.append((n_genuine_neighbors, n_bot_neighbors))

    genuine_counts = [s[0] for s in neighbor_stats]
    bot_counts = [s[1] for s in neighbor_stats]
    x_pos = np.arange(len(sample_nodes))
    bar_colors = ['#2ecc71' if labels[n] == 0 else '#e74c3c' for n in sample_nodes]

    ax.bar(x_pos, genuine_counts, label='Genuine neighbors', color='#2ecc71', alpha=0.7)
    ax.bar(x_pos, bot_counts, bottom=genuine_counts, label='Bot neighbors', color='#e74c3c', alpha=0.7)
    ax.set_xlabel('Sampled Nodes')
    ax.set_ylabel('Neighbor Count')
    ax.set_title('Neighbor Label Distribution (GIN Aggregation)', fontsize=12)
    ax.legend()
    for i, color in enumerate(bar_colors):
        ax.axvline(x=i, color=color, alpha=0.1, linewidth=3)

    # (c) Feature heatmap of aggregated embeddings
    ax = axes[1, 0]
    h1_sample = h1[sample_nodes]
    sns.heatmap(h1_sample, ax=ax, cmap='RdBu_r', center=0,
                yticklabels=[f"{'G' if labels[n]==0 else 'B'}{i}" for i, n in enumerate(sample_nodes)],
                xticklabels=False)
    ax.set_title('GIN Layer-1 Aggregated Features (Sample)', fontsize=12)
    ax.set_xlabel('Hidden Dimensions')
    ax.set_ylabel('Nodes (G=Genuine, B=Bot)')

    # (d) Ego-network visualization for a bot and a genuine user
    ax = axes[1, 1]
    bot_candidates = np.where((labels == 1) & (degrees >= 5) & (degrees <= 20))[0]
    genuine_candidates = np.where((labels == 0) & (degrees >= 5) & (degrees <= 20))[0]

    if len(bot_candidates) > 0 and len(genuine_candidates) > 0:
        center_bot = np.random.choice(bot_candidates)
        center_genuine = np.random.choice(genuine_candidates)

        G = nx.Graph()
        bot_neighbors = adj_list[center_bot][:10]
        G.add_node(f"B_{center_bot}", node_type='bot_center')
        for nb in bot_neighbors:
            ntype = 'bot' if labels[nb] == 1 else 'genuine'
            G.add_node(f"N_{nb}", node_type=ntype)
            G.add_edge(f"B_{center_bot}", f"N_{nb}")

        genuine_neighbors = adj_list[center_genuine][:10]
        G.add_node(f"G_{center_genuine}", node_type='genuine_center')
        for nb in genuine_neighbors:
            ntype = 'bot' if labels[nb] == 1 else 'genuine'
            G.add_node(f"N2_{nb}", node_type=ntype)
            G.add_edge(f"G_{center_genuine}", f"N2_{nb}")

        color_map = []
        for node in G.nodes():
            nt = G.nodes[node]['node_type']
            if nt == 'bot_center':
                color_map.append('#c0392b')
            elif nt == 'genuine_center':
                color_map.append('#27ae60')
            elif nt == 'bot':
                color_map.append('#e74c3c')
            else:
                color_map.append('#2ecc71')

        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, ax=ax, node_color=color_map, node_size=200,
                edge_color='#bdc3c7', width=1.5, with_labels=False)
        ax.set_title('Ego Networks: GIN Aggregation Pattern', fontsize=12)
        legend_elements = [
            Patch(facecolor='#c0392b', label='Bot (center)'),
            Patch(facecolor='#27ae60', label='Genuine (center)'),
            Patch(facecolor='#e74c3c', label='Bot (neighbor)'),
            Patch(facecolor='#2ecc71', label='Genuine (neighbor)'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'gin_aggregation_visualization.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Visualization saved to {OUTPUT_DIR}/gin_aggregation_visualization.png")


# =============================================================================
# 5. Main Pipeline
# =============================================================================

def main():
    print("=" * 60)
    print("Cresci-2017 Bot Detection with GNN")
    print("=" * 60)

    # --- Data Preparation ---
    print("\n[1/4] Simulating Cresci-2017 dataset...")
    features, labels, indices = simulate_cresci2017(n_users=5000, bot_ratio=0.4)
    print(f"  Users: {len(labels)}, Genuine: {(labels==0).sum()}, Bots: {(labels==1).sum()}")
    print(f"  Features: followers, friends, tweet_freq, url_ratio, sentiment")

    print("\n[2/4] Building co-mention graph...")
    edge_index = build_comention_graph(n_users=5000, n_edges=15000, labels=labels)
    print(f"  Edges (undirected): {edge_index.shape[1] // 2}")

    # Build PyG data object
    x = torch.tensor(features, dtype=torch.float)
    y = torch.tensor(labels, dtype=torch.long)
    data = Data(x=x, y=y, edge_index=edge_index).to(DEVICE)

    # Train/Val/Test split: 70/20/10
    all_indices = np.arange(len(labels))
    train_idx, temp_idx = train_test_split(all_indices, test_size=0.3, random_state=SEED, stratify=labels)
    val_idx, test_idx = train_test_split(temp_idx, test_size=1/3, random_state=SEED, stratify=labels[temp_idx])

    train_mask = torch.zeros(len(labels), dtype=torch.bool)
    val_mask = torch.zeros(len(labels), dtype=torch.bool)
    test_mask = torch.zeros(len(labels), dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    train_mask = train_mask.to(DEVICE)
    val_mask = val_mask.to(DEVICE)
    test_mask = test_mask.to(DEVICE)

    print(f"  Split: Train={train_mask.sum().item()}, Val={val_mask.sum().item()}, Test={test_mask.sum().item()}")

    # --- Model Training and Evaluation ---
    print("\n[3/4] Training models...")
    results = {}

    models = {
        'GIN': GINNet(in_channels=5, hidden_channels=64, num_classes=2),
        'GCN': GCNNet(in_channels=5, hidden_channels=64, num_classes=2),
        'GraphSAGE': GraphSAGENet(in_channels=5, hidden_channels=64, num_classes=2),
    }

    for name, model in models.items():
        print(f"\n  --- {name} ---")
        model = model.to(DEVICE)
        model = train_model(model, data, train_mask, val_mask, epochs=200, lr=0.01)
        f1_macro, f1_per_class, report = evaluate_model(model, data, test_mask)
        results[name] = {'f1_macro': f1_macro, 'f1_per_class': f1_per_class, 'report': report}
        models[name] = model
        print(f"  Test F1 (macro): {f1_macro:.4f}")
        print(f"  F1 per class - Genuine: {f1_per_class[0]:.4f}, Bot: {f1_per_class[1]:.4f}")

    # --- Results Comparison ---
    print("\n" + "=" * 60)
    print("Model Comparison (10% Test Set)")
    print("=" * 60)
    print(f"{'Model':<12} {'F1 (Macro)':<12} {'F1 (Genuine)':<14} {'F1 (Bot)':<10}")
    print("-" * 48)
    for name, res in results.items():
        print(f"{name:<12} {res['f1_macro']:<12.4f} {res['f1_per_class'][0]:<14.4f} {res['f1_per_class'][1]:<10.4f}")

    print("\n\nDetailed Classification Report (GIN):")
    print(results['GIN']['report'])

    # --- F1 Comparison Bar Chart ---
    fig, ax = plt.subplots(figsize=(8, 5))
    model_names = list(results.keys())
    f1_scores = [results[n]['f1_macro'] for n in model_names]
    bars = ax.bar(model_names, f1_scores, color=['#3498db', '#e67e22', '#9b59b6'], edgecolor='black', width=0.5)
    for bar, score in zip(bars, f1_scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{score:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1 Score (Macro)', fontsize=12)
    ax.set_title('Bot Detection: GIN vs GCN vs GraphSAGE (10% Test Set)', fontsize=13)
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'f1_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  F1 comparison chart saved to {OUTPUT_DIR}/f1_comparison.png")

    # --- GIN Aggregation Visualization ---
    print("\n[4/4] Visualizing GIN first-layer neighbor aggregation...")
    gin_model = models['GIN']
    visualize_gin_aggregation(gin_model, data, n_samples=20)

    print("\n" + "=" * 60)
    print("Done! All outputs saved to ./outputs/")
    print("=" * 60)


if __name__ == '__main__':
    main()
