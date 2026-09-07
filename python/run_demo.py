"""
run_demo.py — multi-seed evaluation script for LoRa Disaster-Mesh prototype.

1. Builds random disaster-area meshes (25 nodes, gateway=0).
2. Loads/trains multi-hop LinkGNN.
3. Trains generalizable RL routing+SF agent over multiple seeds (10 independent runs).
4. Evaluates pre-disaster, post-disaster (15% node loss + online adaptation), and
   zero-shot generalization on completely UNSEEN topologies.
5. Reports mean +/- std for PDR, average hops, and relative energy per policy.
6. Saves summary CSV and multi-panel error bar plot to results/.

Usage:
    python run_demo.py [--seeds 10]
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lora_mesh.mesh import generate_mesh, apply_disaster_event
from lora_mesh.gnn import LinkGNN
from lora_mesh.rl_agent import RoutingRLAgent, SF_AIRTIME_REL
from lora_mesh.baselines import fixed_sf_shortest_path, adr_like_shortest_path

GNN_WEIGHTS = "lora_mesh_gnn.npz"
GATEWAY = 0
N_NODES = 25
N_TRIALS_PER_NODE = 10
RESULTS_DIR = "../results"


def get_gnn():
    if os.path.exists(GNN_WEIGHTS):
        print(f"Loading trained GNN from {GNN_WEIGHTS}")
        return LinkGNN.load(GNN_WEIGHTS)
    print("No trained GNN weights found — training a quick one now "
          "(run train_gnn.py separately for a more thorough fit)...")
    import train_gnn
    gnn = LinkGNN(hidden_dim=16, lr=0.1)
    X, y = train_gnn.build_training_set(n_topologies=30, gnn=gnn)
    for ep in range(800):
        idx = np.random.choice(len(X), size=min(64, len(X)), replace=False)
        gnn.train_step(X[idx], y[idx])
    gnn.save(GNN_WEIGHTS)
    return gnn


def evaluate_policy(name, fn, G, gnn, agent, sources, rng):
    delivered, hops_list, energy_list = 0, [], []
    dist_map = agent._get_dist_map(G)
    pdr_map = {}
    if hasattr(gnn, "compute_node_embeddings"):
        node_to_idx, h2, edge_feats = gnn.compute_node_embeddings(G)
        idx_to_node = {idx: n for n, idx in node_to_idx.items()}
        for (u_idx, v_idx), ef in edge_feats.items():
            u, v = idx_to_node[u_idx], idx_to_node[v_idx]
            x_link = np.concatenate([h2[u_idx], h2[v_idx], ef])
            z1 = np.tanh(x_link @ gnn.W1 + gnn.b1)
            z2 = z1 @ gnn.W2 + gnn.b2
            pdr_map[(u, v)] = float(1.0 / (1.0 + np.exp(-np.clip(z2[0], -30, 30))))


    for src in sources:
        for _ in range(N_TRIALS_PER_NODE):
            if name == "GNN+RL (proposed)":
                ok, path, hops, sf_list = agent.route_packet(G, gnn, src, rng=rng, dist_map=dist_map, pdr_map=pdr_map)
            else:
                ok, path, hops, sf_list = fn(G, gnn, src, GATEWAY, rng=rng, pdr_map=pdr_map)


            delivered += int(ok)
            if ok:
                hops_list.append(hops)
                energy = sum(SF_AIRTIME_REL[sf] for sf in sf_list) if sf_list else hops * SF_AIRTIME_REL[9]
                energy_list.append(energy)

    n_total = len(sources) * N_TRIALS_PER_NODE
    return {
        "policy": name,
        "PDR": delivered / max(1, n_total),
        "avg_hops": float(np.mean(hops_list)) if hops_list else float("nan"),
        "rel_energy": float(np.mean(energy_list)) if energy_list else float("nan"),
    }



def run_single_seed(seed, gnn):
    rng = np.random.default_rng(seed)

    # 1. Base mesh
    G = generate_mesh(n_nodes=N_NODES, seed=seed)
    agent = RoutingRLAgent(gateway=GATEWAY)
    agent.train(G, gnn, n_episodes=400, rng=rng)

    # Pre-disaster
    alive_pre = [n for n in G.nodes if n != GATEWAY and G.nodes[n]["alive"]]
    pre_rows = [
        evaluate_policy("GNN+RL (proposed)", None, G, gnn, agent, alive_pre, rng),
        evaluate_policy("Fixed-SF shortest path (AODV-like)", fixed_sf_shortest_path, G, gnn, agent, alive_pre, rng),
        evaluate_policy("ADR-like shortest path", adr_like_shortest_path, G, gnn, agent, alive_pre, rng),
    ]

    # 2. Post-disaster
    G_dis = apply_disaster_event(G, "node_loss", rng)
    agent.train(G_dis, gnn, n_episodes=150, rng=rng)
    alive_post = [n for n in G_dis.nodes if n != GATEWAY and G_dis.nodes[n]["alive"]]
    post_rows = [
        evaluate_policy("GNN+RL (proposed)", None, G_dis, gnn, agent, alive_post, rng),
        evaluate_policy("Fixed-SF shortest path (AODV-like)", fixed_sf_shortest_path, G_dis, gnn, agent, alive_post, rng),
        evaluate_policy("ADR-like shortest path", adr_like_shortest_path, G_dis, gnn, agent, alive_post, rng),
    ]

    # 3. Unseen generalization topology
    G_unseen = generate_mesh(n_nodes=N_NODES, seed=seed + 1000)
    alive_unseen = [n for n in G_unseen.nodes if n != GATEWAY and G_unseen.nodes[n]["alive"]]
    unseen_rows = [
        evaluate_policy("GNN+RL (proposed)", None, G_unseen, gnn, agent, alive_unseen, rng),
        evaluate_policy("Fixed-SF shortest path (AODV-like)", fixed_sf_shortest_path, G_unseen, gnn, agent, alive_unseen, rng),
        evaluate_policy("ADR-like shortest path", adr_like_shortest_path, G_unseen, gnn, agent, alive_unseen, rng),
    ]

    results = []
    for r in pre_rows:
        r["scenario"] = "pre-disaster"
        results.append(r)
    for r in post_rows:
        r["scenario"] = "post-disaster"
        results.append(r)
    for r in unseen_rows:
        r["scenario"] = "unseen-generalization"
        results.append(r)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    gnn = get_gnn()

    print(f"Running multi-seed evaluation across {args.seeds} independent seeds...")
    all_runs = []
    for s in range(args.seeds):
        seed_val = 42 + s
        run_results = run_single_seed(seed_val, gnn)
        for r in run_results:
            r["seed"] = seed_val
        all_runs.extend(run_results)

    df_raw = pd.DataFrame(all_runs)

    # Compute mean and std dev grouped by scenario and policy
    summary = (
        df_raw.groupby(["scenario", "policy"])
        .agg(
            PDR_mean=("PDR", "mean"),
            PDR_std=("PDR", "std"),
            avg_hops_mean=("avg_hops", "mean"),
            rel_energy_mean=("rel_energy", "mean"),
        )
        .reset_index()
    )

    print("\n=== Multi-Seed Evaluation Summary (Mean +/- Std Dev) ===")
    print(summary.to_string(index=False))

    csv_path = os.path.join(RESULTS_DIR, "comparison_results.csv")
    summary.to_csv(csv_path, index=False)
    print(f"\nSaved summary dataset to {csv_path}")

    # Plot results with error bars
    scenarios = ["pre-disaster", "post-disaster", "unseen-generalization"]
    titles = ["Pre-Disaster", "Post-Disaster (Node Loss)", "Unseen Topology Generalization"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    colors = ["#2e7d32", "#9e9e9e", "#607d8b"]

    for sc, title, ax in zip(scenarios, titles, axes):
        sub = summary[summary.scenario == sc]
        bars = ax.bar(
            sub.policy,
            sub.PDR_mean,
            yerr=sub.PDR_std,
            capsize=5,
            color=colors,
            alpha=0.85,
        )
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Packet Delivery Ratio (PDR)")
        ax.tick_params(axis="x", rotation=25)
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                h + 0.03,
                f"{h:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "pdr_comparison.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Saved error-bar comparison plot to {plot_path}")


if __name__ == "__main__":
    main()

