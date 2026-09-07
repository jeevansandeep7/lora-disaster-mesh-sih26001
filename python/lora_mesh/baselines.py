"""
baselines.py — two non-learned comparison policies.

1. fixed_sf_shortest_path: classic AODV-style shortest-hop-count
   routing with a single fixed SF for the whole network (SF9, a
   common default) — no adaptation to channel conditions at all.

2. adr_like_shortest_path: same shortest-path routing, but per-hop SF
   is chosen the way LoRaWAN ADR does in spirit — pick the LOWEST SF
   (fastest, least airtime) whose required-SNR threshold is still met
   by the link's estimated SNR, falling back to the most robust SF if
   none qualify. This adapts SF to conditions but does NOT adapt the
   route itself to reliability (still pure shortest path).
"""

import networkx as nx
import numpy as np

# Approximate required SNR (dB) for each SF
SF_REQUIRED_SNR_DB = {7: -7.5, 9: -12.5, 12: -20.0}
# relative airtime multiplier per SF derived from Ts = 2^SF / BW
SF_AIRTIME_REL = {7: 1.0, 9: 4.0, 12: 32.0}


def _link_pdr(gnn, G, i, j, sf, pdr_map=None):
    if pdr_map is not None and (i, j) in pdr_map:
        pdr = pdr_map[(i, j)]
    elif hasattr(gnn, "predict_link"):
        pdr = gnn.predict_link(G, i, j)
    else:
        from .gnn import edge_features
        feat = edge_features(G.nodes[i], G.nodes[j], G.edges[i, j]["snr_db"], "rician", False)
        pdr = float(gnn.predict(feat)[0])
    return min(1.0, pdr + 0.05 * list(SF_REQUIRED_SNR_DB).index(sf))


def fixed_sf_shortest_path(G, gnn, source, gateway, fixed_sf=7, rng=None, max_hops=8, pdr_map=None):
    rng = rng or np.random.default_rng()
    alive = G.subgraph([n for n in G.nodes if G.nodes[n]["alive"]])
    if source not in alive or gateway not in alive or not nx.has_path(alive, source, gateway):
        return False, [], 0, []
    path = nx.shortest_path(alive, source, gateway)
    sf_list = [fixed_sf] * (len(path) - 1)
    for i, j in zip(path[:-1], path[1:]):
        pdr = _link_pdr(gnn, G, i, j, fixed_sf, pdr_map=pdr_map)
        if rng.random() >= pdr:
            return False, path, len(path) - 1, sf_list
    return True, path, len(path) - 1, sf_list


def adr_like_shortest_path(G, gnn, source, gateway, rng=None, max_hops=8, pdr_map=None):
    rng = rng or np.random.default_rng()
    alive = G.subgraph([n for n in G.nodes if G.nodes[n]["alive"]])
    if source not in alive or gateway not in alive or not nx.has_path(alive, source, gateway):
        return False, [], 0, []
    path = nx.shortest_path(alive, source, gateway)
    sf_list = []
    for i, j in zip(path[:-1], path[1:]):
        snr = G.edges[i, j]["snr_db"]
        chosen_sf = 12
        for sf in sorted(SF_REQUIRED_SNR_DB, reverse=False):
            if snr >= SF_REQUIRED_SNR_DB[sf]:
                chosen_sf = sf
                break
        sf_list.append(chosen_sf)
        pdr = _link_pdr(gnn, G, i, j, chosen_sf, pdr_map=pdr_map)
        if rng.random() >= pdr:
            return False, path, len(path) - 1, sf_list
    return True, path, len(path) - 1, sf_list


