"""
mesh.py — disaster-scenario mesh topology generator.

Generates a random multi-node LoRa mesh (simulating scattered
survivor-beacon / rescue-team nodes across a disaster area), assigns
each node a battery level and mobility, and computes a path-loss-based
SNR estimate for every link within radio range. This SNR estimate is
what feeds the link-quality model (link_quality.py) which in turn is
grounded in the MATLAB PHY+channel simulation's dataset.
"""

import numpy as np
import networkx as nx

# --- Radio / propagation constants (LoRa 125 kHz, SX1276-class node) ---
TX_POWER_DBM = 14.0          # typical LoRa TX power
NOISE_FLOOR_DBM = -174 + 10 * np.log10(125e3)  # thermal noise floor, 125kHz BW
PATH_LOSS_EXP = 3.0          # rubble/collapsed-structure NLOS environment
REF_DISTANCE_M = 1.0
REF_LOSS_DB = 40.0           # path loss at reference distance
MAX_RANGE_M = 1000.0         # max link range worth considering (forces multi-hop)


def path_loss_db(distance_m: float) -> float:
    d = max(distance_m, REF_DISTANCE_M)
    return REF_LOSS_DB + 10 * PATH_LOSS_EXP * np.log10(d / REF_DISTANCE_M)


def estimate_snr_db(distance_m: float, battery_scale: float = 1.0) -> float:
    """Rough link-budget SNR estimate for a node at given distance,
    with battery_scale in (0,1] derating the effective TX power."""
    tx_dbm = TX_POWER_DBM + 10 * np.log10(max(battery_scale, 1e-3))
    rx_dbm = tx_dbm - path_loss_db(distance_m)
    return rx_dbm - NOISE_FLOOR_DBM


def generate_mesh(
    n_nodes: int = 20,
    area_m: float = 2400.0,
    seed: int | None = None,
) -> nx.Graph:
    """Create a random disaster-mesh topology.

    Node attrs: pos (x,y), battery (0-1), mobile (bool)
    Edge attrs: distance_m, snr_db  (added for every pair within range)
    """
    rng = np.random.default_rng(seed)
    G = nx.Graph()
    for i in range(n_nodes):
        pos = rng.uniform(0, area_m, size=2)
        battery = float(rng.uniform(0.3, 1.0))
        mobile = bool(rng.random() < 0.3)
        G.add_node(i, pos=pos, battery=battery, mobile=mobile, alive=True)

    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            d = float(np.linalg.norm(G.nodes[i]["pos"] - G.nodes[j]["pos"]))
            if d <= MAX_RANGE_M:
                avg_batt = (G.nodes[i]["battery"] + G.nodes[j]["battery"]) / 2
                snr = estimate_snr_db(d, avg_batt)
                G.add_edge(i, j, distance_m=d, snr_db=snr)
    return G


def apply_disaster_event(G: nx.Graph, event: str, rng: np.random.Generator) -> nx.Graph:
    """Mutate a copy of G to simulate a disaster event.

    event: 'node_loss' (random nodes go offline, e.g. crushed/out of
           battery), 'traffic_burst' (no topology change, handled by
           the traffic generator upstream), 'mobility' (nodes marked
           mobile move, edges recomputed).
    """
    H = G.copy()
    if event == "node_loss":
        n_alive = [n for n in H.nodes if H.nodes[n]["alive"]]
        n_drop = max(1, int(0.15 * len(n_alive)))
        dropped = rng.choice(n_alive, size=n_drop, replace=False)
        for n in dropped:
            H.nodes[n]["alive"] = False
    elif event == "mobility":
        for n in H.nodes:
            if H.nodes[n]["mobile"]:
                H.nodes[n]["pos"] = H.nodes[n]["pos"] + rng.normal(0, 50, size=2)
        # recompute affected edges' distance/snr
        for i, j in H.edges:
            d = float(np.linalg.norm(H.nodes[i]["pos"] - H.nodes[j]["pos"]))
            avg_batt = (H.nodes[i]["battery"] + H.nodes[j]["battery"]) / 2
            H.edges[i, j]["distance_m"] = d
            H.edges[i, j]["snr_db"] = estimate_snr_db(d, avg_batt)
    return H
