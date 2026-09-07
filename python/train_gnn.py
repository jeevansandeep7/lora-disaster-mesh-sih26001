"""
train_gnn.py — trains the multi-hop link-quality GNN (lora_mesh.gnn.LinkGNN) using
labels from lora_mesh.link_quality.LinkQualityModel, sampled across random
mesh topologies with an 80/20 train/validation split.

Usage:
    python train_gnn.py [--episodes 1500] [--out lora_mesh_gnn.npz]
"""

import argparse
import numpy as np

from lora_mesh.mesh import generate_mesh
from lora_mesh.link_quality import LinkQualityModel
from lora_mesh.gnn import LinkGNN, edge_feature_vec, FADING_TYPES


def build_training_set(n_topologies=60, seed=0, gnn=None):
    rng = np.random.default_rng(seed)
    lqm = LinkQualityModel()
    gnn = gnn or LinkGNN()
    print(f"Using {'MATLAB-grounded' if lqm.available else 'ANALYTIC FALLBACK'} link-quality dataset "
          f"({'found' if lqm.available else 'not found -- run runLoRaDisasterSim.m and copy the CSV into matlab/'} "
          f"lora_disaster_link_quality_dataset.csv)")

    X, y = [], []
    for t in range(n_topologies):
        G = generate_mesh(n_nodes=rng.integers(12, 30), seed=int(rng.integers(1e9)))
        node_to_idx, h2, edge_feats = gnn.compute_node_embeddings(G)
        for i, j in G.edges:
            fading = rng.choice(FADING_TYPES)
            collision = bool(rng.random() < 0.2)
            snr = G.edges[i, j]["snr_db"]
            ef = edge_feature_vec(snr, fading, collision)

            i_idx, j_idx = node_to_idx[i], node_to_idx[j]
            x_link = np.concatenate([h2[i_idx], h2[j_idx], ef])
            label = lqm.pdr(snr, fading, collision)
            X.append(x_link)
            y.append(label)
    return np.array(X), np.array(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=1500)
    ap.add_argument("--out", type=str, default="lora_mesh_gnn.npz")
    ap.add_argument("--topologies", type=int, default=60)
    args = ap.parse_args()

    gnn = LinkGNN(hidden_dim=16, lr=0.1)
    X, y = build_training_set(n_topologies=args.topologies, gnn=gnn)

    # 80/20 Train / Validation split
    rng = np.random.default_rng(42)
    n_samples = len(X)
    indices = rng.permutation(n_samples)
    split_idx = int(0.8 * n_samples)
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    print(f"Dataset split: {len(X_train)} train links, {len(X_val)} val links")

    batch = 64
    for ep in range(args.episodes):
        idx = np.random.choice(len(X_train), size=min(batch, len(X_train)), replace=False)
        train_mse = gnn.train_step(X_train[idx], y_train[idx])

        if ep % 200 == 0 or ep == args.episodes - 1:
            val_preds = gnn.predict(X_val)
            val_mse = float(np.mean((val_preds - y_val) ** 2))
            val_mae = float(np.mean(np.abs(val_preds - y_val)))
            print(f"  epoch {ep:5d} | Train MSE: {train_mse:.4f} | Val MSE: {val_mse:.4f} | Val MAE: {val_mae:.4f}")

    gnn.save(args.out)
    print(f"Saved trained multi-hop GNN weights to {args.out}")


if __name__ == "__main__":
    main()

