"""
Temporal Bot Detection: GIN+GRU with Self-Supervised Edge Prediction
====================================================================
- Dataset: Simulated Cresci-2017 (~5000 users, 5 features)
- Temporal Graph: 3 daily time slices with evolving edges + time-varying node features
- Models: GIN+GRU (temporal) vs GIN-Only (static baseline)
- Self-supervised: 10% edge masking per slice, loss summed across all slices
- Evaluation: Low-activity bot recall with only 200 labeled samples
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, precision_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_DIR = 'outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# 1. Temporal Data Simulation
# =============================================================================

def simulate_cresci2017_temporal(n_users=5000, bot_ratio=0.4, n_days=3):
    n_bots = int(n_users * bot_ratio)
    n_genuine = n_users - n_bots

    # Genuine user features
    genuine_followers = np.random.lognormal(mean=5.5, sigma=1.5, size=n_genuine).clip(0, 1e6)
    genuine_friends = np.random.lognormal(mean=4.5, sigma=1.2, size=n_genuine).clip(0, 5e5)
    genuine_tweet_freq = np.random.exponential(scale=3.0, size=n_genuine).clip(0, 50)
    genuine_url_ratio = np.random.beta(a=2, b=8, size=n_genuine)
    genuine_sentiment = np.random.normal(loc=0.1, scale=0.3, size=n_genuine).clip(-1, 1)

    # Regular bots
    n_low_activity = int(n_bots * 0.25)
    n_regular_bots = n_bots - n_low_activity

    reg_bot_followers = np.random.lognormal(mean=3.0, sigma=2.0, size=n_regular_bots).clip(0, 1e5)
    reg_bot_friends = np.random.lognormal(mean=6.0, sigma=1.0, size=n_regular_bots).clip(0, 5e5)
    reg_bot_tweet_freq = np.random.exponential(scale=15.0, size=n_regular_bots).clip(0, 200)
    reg_bot_url_ratio = np.random.beta(a=5, b=3, size=n_regular_bots)
    reg_bot_sentiment = np.random.normal(loc=-0.05, scale=0.15, size=n_regular_bots).clip(-1, 1)

    # Low-activity bots (stealthy)
    low_bot_followers = np.random.lognormal(mean=4.0, sigma=1.5, size=n_low_activity).clip(0, 5e4)
    low_bot_friends = np.random.lognormal(mean=4.8, sigma=1.0, size=n_low_activity).clip(0, 2e5)
    low_bot_tweet_freq = np.random.exponential(scale=1.0, size=n_low_activity).clip(0, 5)
    low_bot_url_ratio = np.random.beta(a=3, b=5, size=n_low_activity)
    low_bot_sentiment = np.random.normal(loc=0.0, scale=0.2, size=n_low_activity).clip(-1, 1)

    # Combine
    followers = np.concatenate([genuine_followers, reg_bot_followers, low_bot_followers])
    friends = np.concatenate([genuine_friends, reg_bot_friends, low_bot_friends])
    tweet_freq = np.concatenate([genuine_tweet_freq, reg_bot_tweet_freq, low_bot_tweet_freq])
    url_ratio = np.concatenate([genuine_url_ratio, reg_bot_url_ratio, low_bot_url_ratio])
    sentiment = np.concatenate([genuine_sentiment, reg_bot_sentiment, low_bot_sentiment])
    labels = np.array([0] * n_genuine + [1] * n_regular_bots + [1] * n_low_activity)

    low_activity_mask_pre = np.zeros(n_users, dtype=bool)
    low_activity_mask_pre[n_genuine + n_regular_bots:] = True

    # Shuffle
    indices = np.random.permutation(n_users)
    features_raw = np.column_stack([followers, friends, tweet_freq, url_ratio, sentiment])[indices]
    labels = labels[indices]
    low_activity_mask = low_activity_mask_pre[indices]

    # Generate time-varying node features for each day
    # Base features are Day 1; subsequent days add drift + noise
    # Genuine users: small random noise (natural fluctuation)
    # Regular bots: increasing tweet_freq drift (ramp-up campaign)
    # Low-activity bots: sudden sentiment shift + url_ratio spike on Day 2-3
    scaler = StandardScaler()
    base_features = scaler.fit_transform(features_raw)

    features_per_day = []
    for day in range(n_days):
        day_features = base_features.copy()

        # Global noise (all users have natural daily fluctuation)
        noise = np.random.normal(0, 0.03, size=day_features.shape)
        day_features += noise

        if day >= 1:
            # Regular bots: tweet frequency drifts up over time
            reg_bot_mask = (labels == 1) & (~low_activity_mask)
            drift_scale = 0.1 * day
            day_features[reg_bot_mask, 2] += np.random.normal(drift_scale, 0.03,
                                                               size=reg_bot_mask.sum())

            # Low-activity bots: subtle oscillating drift (not monotone)
            # The pattern is: Day1=baseline, Day2=+shift, Day3=back toward baseline
            # Only by observing the full sequence can the model detect the anomalous
            # "pulse" pattern vs. genuine users who stay stable
            low_bot_mask = low_activity_mask
            if day == 1:
                # Spike on Day 2
                day_features[low_bot_mask, 3] += np.random.normal(0.25, 0.05,
                                                                   size=low_bot_mask.sum())
                day_features[low_bot_mask, 4] += np.random.normal(-0.2, 0.04,
                                                                   size=low_bot_mask.sum())
            elif day == 2:
                # Partial revert on Day 3 (not identical to Day 1)
                day_features[low_bot_mask, 3] += np.random.normal(0.08, 0.05,
                                                                   size=low_bot_mask.sum())
                day_features[low_bot_mask, 4] += np.random.normal(-0.06, 0.04,
                                                                   size=low_bot_mask.sum())

        features_per_day.append(day_features)

    return features_per_day, labels, low_activity_mask


def build_temporal_graphs(n_users=5000, labels=None, low_activity_mask=None, n_days=3):
    bot_indices = np.where(labels == 1)[0]
    genuine_indices = np.where(labels == 0)[0]
    regular_bot_indices = np.where((labels == 1) & (~low_activity_mask))[0]
    low_bot_indices = np.where((labels == 1) & low_activity_mask)[0]

    edge_targets = [10000, 12000, 14000]
    retention_rates = [1.0, 0.70, 0.60]

    def generate_edges(n_target, bot_idx, genuine_idx, low_bot_idx, existing_edges=None, retain_rate=1.0):
        edges = set()
        if existing_edges is not None and retain_rate < 1.0:
            retained = random.sample(list(existing_edges), int(len(existing_edges) * retain_rate))
            edges.update(retained)

        n_bot_edges = int(n_target * 0.4)
        attempts = 0
        while len(edges) < n_bot_edges and attempts < n_bot_edges * 10:
            u, v = np.random.choice(regular_bot_indices, 2, replace=False)
            edges.add((min(u, v), max(u, v)))
            attempts += 1

        n_genuine_target = int(n_target * 0.35)
        target_so_far = n_bot_edges + n_genuine_target
        attempts = 0
        while len(edges) < target_so_far and attempts < n_genuine_target * 10:
            u, v = np.random.choice(genuine_idx, 2, replace=False)
            edges.add((min(u, v), max(u, v)))
            attempts += 1

        for bot_node in low_bot_idx:
            n_edges_for_bot = np.random.randint(1, 3)
            targets = np.random.choice(genuine_idx, n_edges_for_bot, replace=False)
            for t in targets:
                if t != bot_node:
                    edges.add((min(bot_node, t), max(bot_node, t)))

        attempts = 0
        while len(edges) < n_target and attempts < n_target * 5:
            u = np.random.randint(0, n_users)
            v = np.random.randint(0, n_users)
            if u != v:
                edges.add((min(u, v), max(u, v)))
            attempts += 1

        return edges

    edge_index_list = []
    prev_edges = None

    n_groups = len(low_bot_indices) // 10
    low_bot_groups = [low_bot_indices[i*10:(i+1)*10] for i in range(n_groups)]

    for day in range(n_days):
        if day == 0:
            day_edges = generate_edges(edge_targets[day], bot_indices, genuine_indices,
                                       low_bot_indices, None, 1.0)
        else:
            day_edges = generate_edges(edge_targets[day], bot_indices, genuine_indices,
                                       low_bot_indices, prev_edges, retention_rates[day])

        coord_prob = [0.05, 0.25, 0.50][day]
        for group in low_bot_groups:
            for i in range(len(group)):
                for j in range(i+1, len(group)):
                    if np.random.random() < coord_prob:
                        day_edges.add((min(group[i], group[j]), max(group[i], group[j])))

        prev_edges = day_edges

        edge_list = list(day_edges)
        src = [e[0] for e in edge_list]
        dst = [e[1] for e in edge_list]
        edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)
        edge_index_list.append(edge_index)

    return edge_index_list


# =============================================================================
# 2. Model Definitions
# =============================================================================

class GINEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels=64):
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
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = self.bn1(h)
        h = F.relu(h)
        h = self.conv2(h, edge_index)
        h = self.bn2(h)
        h = F.relu(h)
        return h


class TemporalGINGRU(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, num_classes=2):
        super().__init__()
        self.gin_encoder = GINEncoder(in_channels, hidden_channels)
        self.gru = nn.GRU(input_size=hidden_channels, hidden_size=hidden_channels,
                          num_layers=1, batch_first=True)
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def encode_temporal(self, x_list, edge_index_list):
        """Encode all time slices through GIN + GRU, return last hidden state."""
        embeddings = []
        for x_t, edge_index in zip(x_list, edge_index_list):
            h = self.gin_encoder(x_t, edge_index)
            embeddings.append(h)
        seq = torch.stack(embeddings, dim=1)  # [N, T, hidden]
        output, _ = self.gru(seq)
        return output[:, -1, :]  # [N, hidden]

    def encode_all_steps(self, x_list, edge_index_list):
        """Encode and return GRU output at every time step (for per-slice SSL)."""
        embeddings = []
        for x_t, edge_index in zip(x_list, edge_index_list):
            h = self.gin_encoder(x_t, edge_index)
            embeddings.append(h)
        seq = torch.stack(embeddings, dim=1)  # [N, T, hidden]
        output, _ = self.gru(seq)  # [N, T, hidden]
        return output  # return all T steps

    def forward(self, x_list, edge_index_list):
        h = self.encode_temporal(x_list, edge_index_list)
        return self.classifier(h)


class GINOnly(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, num_classes=2):
        super().__init__()
        self.gin_encoder = GINEncoder(in_channels, hidden_channels)
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(self, x_list, edge_index_list):
        h = self.gin_encoder(x_list[-1], edge_index_list[-1])
        return self.classifier(h)

    def encode(self, x_list, edge_index_list):
        return self.gin_encoder(x_list[-1], edge_index_list[-1])


class EdgePredictor(nn.Module):
    def __init__(self, hidden_channels=64):
        super().__init__()
        self.proj = nn.Linear(hidden_channels, hidden_channels)

    def forward(self, z, pos_edges, neg_edges):
        z_proj = self.proj(z)
        pos_scores = (z_proj[pos_edges[0]] * z_proj[pos_edges[1]]).sum(dim=1)
        neg_scores = (z_proj[neg_edges[0]] * z_proj[neg_edges[1]]).sum(dim=1)
        return pos_scores, neg_scores


# =============================================================================
# 3. Self-Supervised Pretraining (Per-Slice Edge Prediction)
# =============================================================================

def mask_edges(edge_index, mask_ratio=0.1):
    num_edges = edge_index.shape[1]
    half = num_edges // 2
    n_mask = int(half * mask_ratio)
    perm = torch.randperm(half)
    mask_idx = perm[:n_mask]

    masked_pos = edge_index[:, mask_idx]

    keep_mask = torch.ones(half, dtype=torch.bool)
    keep_mask[mask_idx] = False
    keep_idx = torch.cat([torch.where(keep_mask)[0], torch.where(keep_mask)[0] + half])
    remaining_edge_index = edge_index[:, keep_idx]

    return remaining_edge_index, masked_pos


def negative_sampling(edge_index, n_nodes, n_samples):
    edge_set = set()
    ei = edge_index.cpu().numpy()
    for i in range(ei.shape[1]):
        edge_set.add((ei[0, i], ei[1, i]))

    neg_src, neg_dst = [], []
    while len(neg_src) < n_samples:
        u = np.random.randint(0, n_nodes)
        v = np.random.randint(0, n_nodes)
        if u != v and (u, v) not in edge_set and (v, u) not in edge_set:
            neg_src.append(u)
            neg_dst.append(v)

    return torch.tensor([neg_src, neg_dst], dtype=torch.long)


def pretrain_ssl(model, edge_predictor, x_list, edge_index_list, epochs=100, lr=0.001):
    """
    Self-supervised pretraining: mask 10% edges per slice, compute link prediction
    loss at each time step independently and sum, so GRU receives gradients from
    every slice.
    """
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(edge_predictor.parameters()),
        lr=lr, weight_decay=1e-5
    )
    n_nodes = x_list[0].shape[0]
    is_temporal = hasattr(model, 'gru')

    print("  Self-supervised pretraining (per-slice edge prediction)...")
    for epoch in range(epochs):
        model.train()
        edge_predictor.train()
        optimizer.zero_grad()

        # Mask 10% edges in each time slice
        masked_edge_lists = []
        all_pos_edges = []
        for ei in edge_index_list:
            remaining_ei, masked_pos = mask_edges(ei, mask_ratio=0.1)
            masked_edge_lists.append(remaining_ei.to(DEVICE))
            all_pos_edges.append(masked_pos)

        if is_temporal:
            # Get GRU output at every time step: [N, T, hidden]
            all_step_outputs = model.encode_all_steps(x_list, masked_edge_lists)

            # Compute loss for EACH time slice and sum
            total_loss = torch.tensor(0.0, device=DEVICE)
            for t in range(len(edge_index_list)):
                z_t = all_step_outputs[:, t, :]  # [N, hidden] at step t
                pos_edges_t = all_pos_edges[t].to(DEVICE)
                n_pos = pos_edges_t.shape[1]
                if n_pos == 0:
                    continue
                neg_edges_t = negative_sampling(edge_index_list[t], n_nodes, n_pos).to(DEVICE)
                pos_scores, neg_scores = edge_predictor(z_t, pos_edges_t, neg_edges_t)
                pos_loss = F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores))
                neg_loss = F.binary_cross_entropy_with_logits(neg_scores, torch.zeros_like(neg_scores))
                total_loss = total_loss + pos_loss + neg_loss
        else:
            # GIN-Only: encode last slice, predict last slice edges
            z = model.gin_encoder(x_list[-1], masked_edge_lists[-1])
            pos_edges = all_pos_edges[-1].to(DEVICE)
            n_pos = pos_edges.shape[1]
            if n_pos == 0:
                continue
            neg_edges = negative_sampling(edge_index_list[-1], n_nodes, n_pos).to(DEVICE)
            pos_scores, neg_scores = edge_predictor(z, pos_edges, neg_edges)
            pos_loss = F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores))
            neg_loss = F.binary_cross_entropy_with_logits(neg_scores, torch.zeros_like(neg_scores))
            total_loss = pos_loss + neg_loss

        total_loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"    Epoch {epoch+1}: SSL Loss={total_loss.item():.4f}")

    return model, edge_predictor


# =============================================================================
# 4. Supervised Training and Evaluation
# =============================================================================

def train_supervised(model, x_list, edge_index_list, labels, train_mask, val_mask,
                     epochs=150, lr=0.005):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    best_val_f1 = 0
    best_state = None
    patience = 30
    no_improve = 0
    val_f1_history = []

    y = labels.to(DEVICE)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(x_list, edge_index_list)
        loss = F.cross_entropy(out[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_out = model(x_list, edge_index_list)
            val_pred = val_out[val_mask].argmax(dim=1).cpu().numpy()
            val_true = y[val_mask].cpu().numpy()
            val_f1 = f1_score(val_true, val_pred, average='macro')

        val_f1_history.append(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

        if (epoch + 1) % 50 == 0:
            print(f"    Epoch {epoch+1}: Loss={loss.item():.4f}, Val F1={val_f1:.4f}")

    model.load_state_dict(best_state)
    print(f"    Best Val F1: {best_val_f1:.4f}")
    return model, val_f1_history


def evaluate_model(model, x_list, edge_index_list, labels, test_mask, low_activity_mask):
    model.eval()
    with torch.no_grad():
        out = model(x_list, edge_index_list)
        pred = out[test_mask].argmax(dim=1).cpu().numpy()
        true = labels[test_mask].cpu().numpy()

    overall_recall_bot = recall_score(true, pred, pos_label=1)
    overall_precision_bot = precision_score(true, pred, pos_label=1)
    overall_f1 = f1_score(true, pred, average='macro')

    test_indices = torch.where(test_mask)[0].cpu().numpy()
    low_act_in_test = low_activity_mask[test_indices]
    bot_in_test = true == 1
    low_act_bot_mask = low_act_in_test & bot_in_test

    if low_act_bot_mask.sum() > 0:
        low_act_pred = pred[low_act_bot_mask]
        low_activity_recall = (low_act_pred == 1).sum() / low_act_bot_mask.sum()
    else:
        low_activity_recall = 0.0

    report = classification_report(true, pred, target_names=['Genuine', 'Bot'])

    return {
        'overall_recall_bot': overall_recall_bot,
        'overall_precision_bot': overall_precision_bot,
        'overall_f1': overall_f1,
        'low_activity_recall': float(low_activity_recall),
        'report': report,
    }


# =============================================================================
# 5. Visualization
# =============================================================================

def visualize_comparison(results_gru, results_no_gru, val_history_gru, val_history_no_gru,
                         model_gru, x_list, edge_index_list, labels, low_activity_mask, test_mask):
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # (a) Recall comparison bar chart
    ax = axes[0, 0]
    metrics = ['Overall Bot\nRecall', 'Low-Activity\nBot Recall', 'Macro F1']
    gru_vals = [results_gru['overall_recall_bot'], results_gru['low_activity_recall'], results_gru['overall_f1']]
    no_gru_vals = [results_no_gru['overall_recall_bot'], results_no_gru['low_activity_recall'], results_no_gru['overall_f1']]

    x_pos = np.arange(len(metrics))
    width = 0.35
    bars1 = ax.bar(x_pos - width/2, gru_vals, width, label='GIN+GRU', color='#3498db', edgecolor='black')
    bars2 = ax.bar(x_pos + width/2, no_gru_vals, width, label='GIN-Only', color='#e67e22', edgecolor='black')

    for bar, val in zip(bars1, gru_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar, val in zip(bars2, no_gru_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(metrics)
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1.1)
    ax.set_title('(a) GIN+GRU vs GIN-Only: Key Metrics', fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    # (b) Training curves
    ax = axes[0, 1]
    ax.plot(val_history_gru, label='GIN+GRU', color='#3498db', linewidth=2)
    ax.plot(val_history_no_gru, label='GIN-Only', color='#e67e22', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation F1 (Macro)')
    ax.set_title('(b) Fine-tuning Validation Curves (200 labels)', fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)

    # (c) PCA of GIN+GRU embeddings
    ax = axes[1, 0]
    model_gru.eval()
    with torch.no_grad():
        embeddings = model_gru.encode_temporal(x_list, edge_index_list).cpu().numpy()

    pca = PCA(n_components=2)
    emb_2d = pca.fit_transform(embeddings)

    labels_np = labels.cpu().numpy()
    genuine_mask = labels_np == 0
    bot_mask = (labels_np == 1) & (~low_activity_mask)
    low_act_mask = low_activity_mask

    ax.scatter(emb_2d[genuine_mask, 0], emb_2d[genuine_mask, 1],
               c='#2ecc71', alpha=0.2, s=8, label='Genuine')
    ax.scatter(emb_2d[bot_mask, 0], emb_2d[bot_mask, 1],
               c='#e74c3c', alpha=0.3, s=12, label='Regular Bot')
    ax.scatter(emb_2d[low_act_mask, 0], emb_2d[low_act_mask, 1],
               c='#9b59b6', alpha=0.6, s=25, marker='^', label='Low-Activity Bot')
    ax.set_title('(c) GIN+GRU Embeddings (PCA)', fontsize=12)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.legend(fontsize=9)

    # (d) Degree evolution across time slices
    ax = axes[1, 1]
    n_nodes = len(labels_np)
    degrees_per_day = []
    for ei in edge_index_list:
        ei_np = ei.cpu().numpy()
        deg = np.zeros(n_nodes)
        for i in range(ei_np.shape[1]):
            deg[ei_np[0, i]] += 1
        degrees_per_day.append(deg)

    low_bot_idx = np.where(low_activity_mask)[0][:5]
    reg_bot_idx = np.where((labels_np == 1) & (~low_activity_mask))[0][:5]
    genuine_idx = np.where(labels_np == 0)[0][:5]

    days = [1, 2, 3]
    for idx in low_bot_idx:
        degs = [degrees_per_day[d][idx] for d in range(3)]
        ax.plot(days, degs, 'v-', color='#9b59b6', alpha=0.6, markersize=5)
    for idx in reg_bot_idx:
        degs = [degrees_per_day[d][idx] for d in range(3)]
        ax.plot(days, degs, 'o-', color='#e74c3c', alpha=0.4, markersize=4)
    for idx in genuine_idx:
        degs = [degrees_per_day[d][idx] for d in range(3)]
        ax.plot(days, degs, 's-', color='#2ecc71', alpha=0.4, markersize=4)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#9b59b6', marker='v', label='Low-Activity Bot'),
        Line2D([0], [0], color='#e74c3c', marker='o', label='Regular Bot'),
        Line2D([0], [0], color='#2ecc71', marker='s', label='Genuine'),
    ]
    ax.legend(handles=legend_elements, fontsize=9)
    ax.set_xlabel('Day')
    ax.set_ylabel('Node Degree')
    ax.set_title('(d) Degree Evolution Across Time Slices', fontsize=12)
    ax.set_xticks(days)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'temporal_gin_gru_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Visualization saved to {OUTPUT_DIR}/temporal_gin_gru_comparison.png")


# =============================================================================
# 6. Main Pipeline
# =============================================================================

def main():
    print("=" * 60)
    print("Temporal Bot Detection: GIN+GRU vs GIN-Only")
    print("Self-Supervised Edge Prediction | 200 Labeled Samples")
    print("=" * 60)

    # --- Data Preparation ---
    print("\n[1/5] Simulating temporal Cresci-2017 dataset...")
    features_per_day, labels, low_activity_mask = simulate_cresci2017_temporal(
        n_users=5000, bot_ratio=0.4)
    print(f"  Users: {len(labels)}, Genuine: {(labels==0).sum()}, Bots: {(labels==1).sum()}")
    print(f"  Low-activity bots: {low_activity_mask.sum()}")
    print(f"  Time-varying features: {len(features_per_day)} days x {features_per_day[0].shape}")

    print("\n[2/5] Building temporal co-mention graphs (3 days)...")
    edge_index_list = build_temporal_graphs(n_users=5000, labels=labels,
                                            low_activity_mask=low_activity_mask)
    for i, ei in enumerate(edge_index_list):
        print(f"  Day {i+1}: {ei.shape[1]//2} undirected edges")

    # Move to device
    x_list = [torch.tensor(f, dtype=torch.float).to(DEVICE) for f in features_per_day]
    y = torch.tensor(labels, dtype=torch.long).to(DEVICE)
    edge_index_list_dev = [ei.to(DEVICE) for ei in edge_index_list]

    # --- Label split: only 200 for training ---
    all_indices = np.arange(len(labels))
    train_idx, rest_idx = train_test_split(all_indices, train_size=200,
                                            random_state=SEED, stratify=labels)
    val_idx, test_idx = train_test_split(rest_idx, train_size=200,
                                          random_state=SEED, stratify=labels[rest_idx])

    train_mask = torch.zeros(len(labels), dtype=torch.bool, device=DEVICE)
    val_mask = torch.zeros(len(labels), dtype=torch.bool, device=DEVICE)
    test_mask = torch.zeros(len(labels), dtype=torch.bool, device=DEVICE)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    print(f"  Labels: Train=200, Val=200, Test={test_mask.sum().item()}")

    # --- Self-Supervised Pretraining ---
    print("\n[3/5] Self-supervised pretraining (10% edge masking, per-slice loss)...")

    print("\n  >> GIN+GRU Model:")
    model_gru = TemporalGINGRU(in_channels=5, hidden_channels=64, num_classes=2).to(DEVICE)
    edge_pred_gru = EdgePredictor(hidden_channels=64).to(DEVICE)
    model_gru, edge_pred_gru = pretrain_ssl(model_gru, edge_pred_gru, x_list,
                                             edge_index_list_dev, epochs=100, lr=0.001)

    print("\n  >> GIN-Only Model:")
    model_no_gru = GINOnly(in_channels=5, hidden_channels=64, num_classes=2).to(DEVICE)
    edge_pred_no_gru = EdgePredictor(hidden_channels=64).to(DEVICE)
    model_no_gru, edge_pred_no_gru = pretrain_ssl(model_no_gru, edge_pred_no_gru, x_list,
                                                    edge_index_list_dev, epochs=100, lr=0.001)

    # --- Supervised Fine-tuning ---
    print("\n[4/5] Fine-tuning with 200 labeled samples...")

    print("\n  >> GIN+GRU:")
    model_gru, val_hist_gru = train_supervised(model_gru, x_list, edge_index_list_dev, y,
                                                train_mask, val_mask, epochs=150, lr=0.005)

    print("\n  >> GIN-Only:")
    model_no_gru, val_hist_no_gru = train_supervised(model_no_gru, x_list, edge_index_list_dev, y,
                                                      train_mask, val_mask, epochs=150, lr=0.005)

    # --- Evaluation ---
    print("\n[5/5] Evaluation on test set...")
    results_gru = evaluate_model(model_gru, x_list, edge_index_list_dev, y, test_mask, low_activity_mask)
    results_no_gru = evaluate_model(model_no_gru, x_list, edge_index_list_dev, y, test_mask, low_activity_mask)

    # --- Results ---
    print("\n" + "=" * 60)
    print("RESULTS: GIN+GRU vs GIN-Only (200 labeled samples)")
    print("=" * 60)
    print(f"{'Metric':<28} {'GIN+GRU':<12} {'GIN-Only':<12}")
    print("-" * 52)
    print(f"{'Overall Recall (Bot)':<28} {results_gru['overall_recall_bot']:<12.4f} {results_no_gru['overall_recall_bot']:<12.4f}")
    print(f"{'Low-Activity Bot Recall':<28} {results_gru['low_activity_recall']:<12.4f} {results_no_gru['low_activity_recall']:<12.4f}")
    print(f"{'Overall F1 (Macro)':<28} {results_gru['overall_f1']:<12.4f} {results_no_gru['overall_f1']:<12.4f}")
    print(f"{'Precision (Bot)':<28} {results_gru['overall_precision_bot']:<12.4f} {results_no_gru['overall_precision_bot']:<12.4f}")
    print("=" * 52)

    delta_recall = results_gru['low_activity_recall'] - results_no_gru['low_activity_recall']
    print(f"\n  GRU advantage on low-activity bots: {delta_recall:+.4f} recall")

    print("\n\nClassification Report (GIN+GRU):")
    print(results_gru['report'])
    print("Classification Report (GIN-Only):")
    print(results_no_gru['report'])

    # --- Visualization ---
    print("\nGenerating comparison visualization...")
    visualize_comparison(results_gru, results_no_gru, val_hist_gru, val_hist_no_gru,
                         model_gru, x_list, edge_index_list_dev, y, low_activity_mask, test_mask)

    print("\n" + "=" * 60)
    print("Done! All outputs saved to ./outputs/")
    print("=" * 60)


if __name__ == '__main__':
    main()
