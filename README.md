# LoRa Disaster-Mesh — SIH26001 Prototype

AI-based routing + spreading-factor selection for a LoRa mesh under
disaster conditions (landslide/NER early-warning scenario). Proposed
approach: **GNN link-quality prediction + RL routing/SF policy**,
benchmarked against fixed-SF shortest-path (AODV-like) and an
SNR-threshold adaptive-SF baseline (ADR-like).

## Key Completed Features

1. **Multi-hop Message-Passing GNN (`python/lora_mesh/gnn.py`)**:
   - 2 rounds of graph message passing in pure NumPy (zero install friction).
   - Aggregates multi-hop node and edge features before edge PDR regression head.
   - Evaluated with 80/20 train/validation split in `train_gnn.py` reporting Train/Val MSE & MAE.

2. **Topology-Generalizable RL Agent (`python/lora_mesh/rl_agent.py`)**:
   - Q-table keyed by graph-derived state features `((dist_to_gateway_bucket, pdr_bucket), (delta_dist, sf))` instead of raw node IDs.
   - Enables zero-shot generalization to unseen topologies without retraining.

3. **Physics-Grounded Airtime Metrics**:
   - Spreading factor airtime relative multipliers derived directly from $T_s = \frac{2^{SF}}{BW}$ ($SF7: 1.0, SF9: 4.0, SF12: 32.0$).

4. **Automated PHY & Battery Verification**:
   - Automated MATLAB test (`matlab/test_battery_scale.m`) numerically verifying that battery depletion increases Symbol Error Rate (SER).
   - Python consistency test (`python/verify_phy_consistency.py`) verifying directional consistency between path-loss SNR model and MATLAB link quality dataset curves.

5. **Multi-Seed Statistical Evaluation (`python/run_demo.py`)**:
   - Evaluated across 10 independent random seeds.
   - Reports Mean $\pm$ Std Dev for PDR, average hops, and relative energy across pre-disaster, post-disaster (node loss + online adaptation), and zero-shot unseen topology generalization.
   - Generates summary CSV (`results/comparison_results.csv`) and error-bar comparison plots (`results/pdr_comparison.png`).

## Repo structure

```
matlab/    LoRa CSS PHY (SF7-SF12) + disaster channel model (Rician/
           Rayleigh fading, collisions, battery decay). Includes automated test_battery_scale.m.
python/    Mesh topology sim, multi-hop GNN link-quality predictor, generalizable
           Q-learning routing+SF agent, baselines, PHY verification, and end-to-end multi-seed benchmark.
results/   Summary CSV + multi-panel error-bar plot from run_demo.py.
```

## Quickstart

```bash
cd python
pip install -r requirements.txt

# 1. Train GNN with train/val metrics
python train_gnn.py --episodes 1000

# 2. Verify PHY consistency
python verify_phy_consistency.py

# 3. Run multi-seed benchmark & generate plots
python run_demo.py --seeds 10
```

Outputs land in `results/`.

