"""
rl_agent.py — Q-learning agent that jointly selects the next hop
AND spreading factor toward a gateway node.

Uses graph-derived state features (hop distance bucket to gateway + GNN-predicted
link quality bucket) so the Q-table generalizes zero-shot across different mesh
topologies.
"""

from collections import defaultdict
import numpy as np
import networkx as nx

SF_OPTIONS = [7, 9, 12]  # low SF = fast/short range, high SF = slow/robust
# relative airtime multiplier derived from Ts = 2^SF / BW (1.024ms, 4.096ms, 32.768ms)
SF_AIRTIME_REL = {7: 1.0, 9: 4.0, 12: 32.0}


class RoutingRLAgent:
    def __init__(self, gateway: int, alpha=0.3, gamma=0.9, epsilon=0.2):
        self.gateway = gateway
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = defaultdict(float)  # key: ((dist_bucket, pdr_bucket), (delta_dist, sf))

    def _actions(self, G, node, alive_map=None):
        if alive_map is not None and node in alive_map:
            return alive_map[node]
        return [(nbr, sf) for nbr in G.neighbors(node)
                if G.nodes[nbr]["alive"] for sf in SF_OPTIONS]

    def _get_dist_map(self, G):
        alive = G.subgraph([n for n in G.nodes if G.nodes[n]["alive"]])
        if self.gateway not in alive:
            return {}
        return dict(nx.single_source_shortest_path_length(alive, self.gateway))

    def _get_key(self, G, gnn, node, next_hop, sf, dist_map=None, pdr_map=None):
        if dist_map is None:
            dist_map = self._get_dist_map(G)

        d_curr = dist_map.get(node, 99)
        d_next = dist_map.get(next_hop, 99)

        delta_d = d_next - d_curr
        if pdr_map is not None and (node, next_hop) in pdr_map:
            pdr = pdr_map[(node, next_hop)]
        elif hasattr(gnn, "predict_link"):
            pdr = gnn.predict_link(G, node, next_hop)
        else:
            from .gnn import edge_features
            edge = G.edges[node, next_hop]
            feat = edge_features(G.nodes[node], G.nodes[next_hop], edge["snr_db"], "rician", False)
            pdr = float(gnn.predict(feat)[0])

        pdr_bucket = int(np.clip(pdr * 5, 0, 4))
        dist_bucket = min(d_curr, 5)

        state = (dist_bucket, pdr_bucket)
        action = (delta_d, sf)
        return (state, action)

    def choose_action(self, G, gnn, node, greedy=False, dist_map=None, pdr_map=None, alive_map=None):
        actions = self._actions(G, node, alive_map=alive_map)
        if not actions:
            return None
        if (not greedy) and np.random.random() < self.epsilon:
            return actions[np.random.randint(len(actions))]

        qs = [self.Q[self._get_key(G, gnn, node, a[0], a[1], dist_map, pdr_map)] for a in actions]
        return actions[int(np.argmax(qs))]

    def _reward(self, gnn, G, node, next_hop, sf, delivered):
        if next_hop == self.gateway and delivered:
            return 10.0
        base = -0.5 * SF_AIRTIME_REL[sf]  # latency + energy cost of this hop
        if not delivered:
            base -= 5.0  # dropped packet penalty
        return base

    def train_episode(self, G, gnn, source: int, max_hops: int = 8, rng=None, dist_map=None, pdr_map=None, alive_map=None):
        rng = rng or np.random.default_rng()
        node = source
        dist_map = dist_map if dist_map is not None else self._get_dist_map(G)

        for _ in range(max_hops):
            action = self.choose_action(G, gnn, node, dist_map=dist_map, pdr_map=pdr_map, alive_map=alive_map)
            if action is None:
                break
            next_hop, sf = action

            if pdr_map is not None and (node, next_hop) in pdr_map:
                pdr = pdr_map[(node, next_hop)]
            elif hasattr(gnn, "predict_link"):
                pdr = gnn.predict_link(G, node, next_hop)
            else:
                edge = G.edges[node, next_hop]
                from .gnn import edge_features
                feat = edge_features(G.nodes[node], G.nodes[next_hop], edge["snr_db"], "rician", False)
                pdr = float(gnn.predict(feat)[0])

            pdr_eff = min(1.0, pdr + 0.05 * (SF_OPTIONS.index(sf)))
            delivered = rng.random() < pdr_eff

            r = self._reward(gnn, G, node, next_hop, sf, delivered)
            key = self._get_key(G, gnn, node, next_hop, sf, dist_map, pdr_map)

            next_actions = self._actions(G, next_hop, alive_map=alive_map) if delivered else []
            next_max = max([self.Q[self._get_key(G, gnn, next_hop, a[0], a[1], dist_map, pdr_map)] for a in next_actions], default=0.0)
            self.Q[key] += self.alpha * (r + self.gamma * next_max - self.Q[key])

            if not delivered:
                break
            node = next_hop
            if node == self.gateway:
                break
        return node == self.gateway

    def train(self, G, gnn, n_episodes=300, rng=None):
        rng = rng or np.random.default_rng(0)
        nodes = [n for n in G.nodes if n != self.gateway]
        if not nodes:
            return 0.0
        success = 0
        eps0 = self.epsilon

        dist_map = self._get_dist_map(G)
        alive_map = {n: [(nbr, sf) for nbr in G.neighbors(n) if G.nodes[nbr]["alive"] for sf in SF_OPTIONS] for n in G.nodes}
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


        for ep in range(n_episodes):
            self.epsilon = max(0.02, eps0 * (1 - ep / n_episodes))
            src = int(rng.choice(nodes))
            success += self.train_episode(G, gnn, src, rng=rng, dist_map=dist_map, pdr_map=pdr_map, alive_map=alive_map)
        self.epsilon = 0.02
        return success / max(1, n_episodes)

    def route_packet(self, G, gnn, source: int, max_hops: int = 8, rng=None, dist_map=None, pdr_map=None, alive_map=None):
        """Greedy rollout (no exploration) — used at evaluation time."""
        rng = rng or np.random.default_rng()
        dist_map = dist_map if dist_map is not None else self._get_dist_map(G)
        alive_map = alive_map if alive_map is not None else {n: [(nbr, sf) for nbr in G.neighbors(n) if G.nodes[nbr]["alive"] for sf in SF_OPTIONS] for n in G.nodes}
        node = source
        path = [node]
        hops = 0
        sf_list = []
        for _ in range(max_hops):
            action = self.choose_action(G, gnn, node, greedy=True, dist_map=dist_map, pdr_map=pdr_map, alive_map=alive_map)
            if action is None:
                return False, path, hops, sf_list
            next_hop, sf = action
            sf_list.append(sf)

            if pdr_map is not None and (node, next_hop) in pdr_map:
                pdr = pdr_map[(node, next_hop)]
            elif hasattr(gnn, "predict_link"):
                pdr = gnn.predict_link(G, node, next_hop)
            else:
                edge = G.edges[node, next_hop]
                from .gnn import edge_features
                feat = edge_features(G.nodes[node], G.nodes[next_hop], edge["snr_db"], "rician", False)
                pdr = float(gnn.predict(feat)[0])

            pdr_eff = min(1.0, pdr + 0.05 * (SF_OPTIONS.index(sf)))
            delivered = rng.random() < pdr_eff
            hops += 1
            if not delivered:
                return False, path, hops, sf_list
            node = next_hop
            path.append(node)
            if node == self.gateway:
                return True, path, hops, sf_list
        return False, path, hops, sf_list



