"""
gnn.py — 2-round message-passing GNN for per-link PDR prediction.

Implemented in plain NumPy (forward pass + manual backprop) for zero install
friction.

Architecture (2 message-passing rounds + link regression head):
    Round 0: Initial node features h_i^(0) = [battery_i, mobile_i] (dim 2)
             Edge features e_ij = [snr_db/30, fading_onehot (3), collision] (dim 5)
    Round 1: Message m_ji^(1) = tanh([h_j^(0), e_ji] * Wm1 + bm1)
             Aggregated m_i^(1) = mean_{j in N(i)}(m_ji^(1))
             Updated node embedding h_i^(1) = tanh([h_i^(0), m_i^(1)] * Wu1 + bu1) (dim 8)
    Round 2: Message m_ji^(2) = tanh([h_j^(1), e_ji] * Wm2 + bm2)
             Aggregated m_i^(2) = mean_{j in N(i)}(m_ji^(2))
             Updated node embedding h_i^(2) = tanh([h_i^(1), m_i^(2)] * Wu2 + bu2) (dim 8)
    Link Head: Concatenate [h_i^(2), h_j^(2), e_ij] (dim 21)
               z1 = tanh(x * W1 + b1) (dim 16)
               y_hat = sigmoid(z1 * W2 + b2) -> PDR prediction in (0, 1)

Trained on labels produced by link_quality.LinkQualityModel (grounded in the MATLAB
PHY+channel Monte Carlo dataset), across random mesh topologies.
"""

from dataclasses import dataclass, field
import numpy as np

FADING_TYPES = ["none", "rician", "rayleigh"]


def node_feature_vec(node_dict):
    return np.array([node_dict.get("battery", 1.0), float(node_dict.get("mobile", False))], dtype=np.float64)


def edge_feature_vec(snr_db, fading, collision):
    fading_onehot = [1.0 if f == fading else 0.0 for f in FADING_TYPES]
    return np.array([snr_db / 30.0, *fading_onehot, float(collision)], dtype=np.float64)


def edge_features(node_i, node_j, snr_db, fading, collision):
    """Backwards-compatible legacy feature function."""
    fading_onehot = [1.0 if f == fading else 0.0 for f in FADING_TYPES]
    return np.array(
        [
            node_i["battery"], float(node_i["mobile"]),
            node_j["battery"], float(node_j["mobile"]),
            snr_db / 30.0,
            *fading_onehot,
            float(collision),
        ],
        dtype=np.float64,
    )


NODE_DIM = 2
EDGE_DIM = 5
HIDDEN_NODE_DIM = 8
LINK_HEAD_IN = 2 * HIDDEN_NODE_DIM + EDGE_DIM  # 8 + 8 + 5 = 21


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


@dataclass
class LinkGNN:
    hidden_dim: int = 16
    lr: float = 0.05
    # Message passing parameters (Round 1 & 2)
    Wm1: np.ndarray = field(default=None)
    bm1: np.ndarray = field(default=None)
    Wu1: np.ndarray = field(default=None)
    bu1: np.ndarray = field(default=None)

    Wm2: np.ndarray = field(default=None)
    bm2: np.ndarray = field(default=None)
    Wu2: np.ndarray = field(default=None)
    bu2: np.ndarray = field(default=None)

    # Link head MLP parameters
    W1: np.ndarray = field(default=None)
    b1: np.ndarray = field(default=None)
    W2: np.ndarray = field(default=None)
    b2: np.ndarray = field(default=None)

    def __post_init__(self):
        if self.W1 is None:
            rng = np.random.default_rng(0)
            # Round 1
            self.Wm1 = rng.normal(0, 0.2, size=(NODE_DIM + EDGE_DIM, HIDDEN_NODE_DIM))
            self.bm1 = np.zeros(HIDDEN_NODE_DIM)
            self.Wu1 = rng.normal(0, 0.2, size=(NODE_DIM + HIDDEN_NODE_DIM, HIDDEN_NODE_DIM))
            self.bu1 = np.zeros(HIDDEN_NODE_DIM)

            # Round 2
            self.Wm2 = rng.normal(0, 0.2, size=(HIDDEN_NODE_DIM + EDGE_DIM, HIDDEN_NODE_DIM))
            self.bm2 = np.zeros(HIDDEN_NODE_DIM)
            self.Wu2 = rng.normal(0, 0.2, size=(HIDDEN_NODE_DIM + HIDDEN_NODE_DIM, HIDDEN_NODE_DIM))
            self.bu2 = np.zeros(HIDDEN_NODE_DIM)

            # Head
            self.W1 = rng.normal(0, 0.2, size=(LINK_HEAD_IN, self.hidden_dim))
            self.b1 = np.zeros(self.hidden_dim)
            self.W2 = rng.normal(0, 0.2, size=(self.hidden_dim, 1))
            self.b2 = np.zeros(1)

    def compute_node_embeddings(self, G):
        """2 rounds of message passing over graph G."""
        nodes = list(G.nodes)
        node_to_idx = {n: idx for idx, n in enumerate(nodes)}
        n_count = len(nodes)

        # Round 0 node states
        h0 = np.zeros((n_count, NODE_DIM), dtype=np.float64)
        for n, idx in node_to_idx.items():
            h0[idx] = node_feature_vec(G.nodes[n])

        # Precompute edge features
        adj = {idx: [] for idx in range(n_count)}
        edge_feats = {}
        for u, v in G.edges:
            u_idx, v_idx = node_to_idx[u], node_to_idx[v]
            snr = G.edges[u, v].get("snr_db", 0.0)
            fading = G.edges[u, v].get("fading", "rician")
            collision = G.edges[u, v].get("collision", False)
            ef = edge_feature_vec(snr, fading, collision)
            edge_feats[(u_idx, v_idx)] = ef
            edge_feats[(v_idx, u_idx)] = ef
            adj[u_idx].append(v_idx)
            adj[v_idx].append(u_idx)

        # Round 1
        m1_agg = np.zeros((n_count, HIDDEN_NODE_DIM), dtype=np.float64)
        for i in range(n_count):
            nbrs = adj[i]
            if nbrs:
                msgs = []
                for j in nbrs:
                    in_feat = np.concatenate([h0[j], edge_feats[(j, i)]])
                    m_ji = np.tanh(in_feat @ self.Wm1 + self.bm1)
                    msgs.append(m_ji)
                m1_agg[i] = np.mean(msgs, axis=0)

        h1_in = np.hstack([h0, m1_agg])
        h1 = np.tanh(h1_in @ self.Wu1 + self.bu1)

        # Round 2
        m2_agg = np.zeros((n_count, HIDDEN_NODE_DIM), dtype=np.float64)
        for i in range(n_count):
            nbrs = adj[i]
            if nbrs:
                msgs = []
                for j in nbrs:
                    in_feat = np.concatenate([h1[j], edge_feats[(j, i)]])
                    m_ji = np.tanh(in_feat @ self.Wm2 + self.bm2)
                    msgs.append(m_ji)
                m2_agg[i] = np.mean(msgs, axis=0)

        h2_in = np.hstack([h1, m2_agg])
        h2 = np.tanh(h2_in @ self.Wu2 + self.bu2)

        return node_to_idx, h2, edge_feats

    def predict_link(self, G, i, j, snr_db=None, fading="rician", collision=False):
        """Predict link PDR for edge (i, j) using 2-round GNN message passing over G."""
        node_to_idx, h2, edge_feats = self.compute_node_embeddings(G)
        i_idx, j_idx = node_to_idx[i], node_to_idx[j]

        if snr_db is None:
            ef = edge_feats.get((i_idx, j_idx), edge_feature_vec(G.edges[i, j]["snr_db"], fading, collision))
        else:
            ef = edge_feature_vec(snr_db, fading, collision)

        x_link = np.concatenate([h2[i_idx], h2[j_idx], ef])
        z1 = np.tanh(x_link @ self.W1 + self.b1)
        z2 = z1 @ self.W2 + self.b2
        pdr = _sigmoid(z2)[0]
        return float(pdr)

    def forward_link_batch(self, X_link):
        """Forward pass for link head inputs (batch_size, 21)."""
        z1 = X_link @ self.W1 + self.b1
        h1 = np.tanh(z1)
        z2 = h1 @ self.W2 + self.b2
        y = _sigmoid(z2).ravel()
        return y, (X_link, z1, h1, z2)

    def predict(self, X) -> np.ndarray:
        """Legacy prediction interface accepting feature vectors or link head vectors."""
        X_arr = np.atleast_2d(X)
        if X_arr.shape[1] == 9:
            # Legacy 9-dim edge feature vector: pad dummy node states to reach 21 dims
            h_dummy = np.zeros((X_arr.shape[0], 2 * HIDDEN_NODE_DIM))
            X_link = np.hstack([h_dummy, X_arr[:, 4:]])
        else:
            X_link = X_arr
        y, _ = self.forward_link_batch(X_link)
        return y

    def train_step(self, X_link, y_true):
        """Train step for GNN link regression head."""
        X_arr = np.atleast_2d(X_link)
        if X_arr.shape[1] == 9:
            h_dummy = np.zeros((X_arr.shape[0], 2 * HIDDEN_NODE_DIM))
            X_link = np.hstack([h_dummy, X_arr[:, 4:]])

        y_pred, (X_, z1, h1, z2) = self.forward_link_batch(X_link)
        n = X_link.shape[0]

        dz2 = (2.0 / n) * (y_pred - y_true) * y_pred * (1.0 - y_pred)
        dz2 = dz2.reshape(-1, 1)
        dW2 = h1.T @ dz2
        db2 = dz2.sum(axis=0)

        dh1 = dz2 @ self.W2.T
        dz1 = dh1 * (1.0 - np.tanh(z1) ** 2)
        dW1 = X_.T @ dz1
        db1 = dz1.sum(axis=0)

        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

        return float(np.mean((y_pred - y_true) ** 2))

    def save(self, path):
        np.savez(
            path,
            Wm1=self.Wm1, bm1=self.bm1, Wu1=self.Wu1, bu1=self.bu1,
            Wm2=self.Wm2, bm2=self.bm2, Wu2=self.Wu2, bu2=self.bu2,
            W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2,
        )

    @classmethod
    def load(cls, path):
        d = np.load(path)
        return cls(
            hidden_dim=d["b1"].shape[0],
            Wm1=d["Wm1"], bm1=d["bm1"], Wu1=d["Wu1"], bu1=d["bu1"],
            Wm2=d["Wm2"], bm2=d["bm2"], Wu2=d["Wu2"], bu2=d["bu2"],
            W1=d["W1"], b1=d["b1"], W2=d["W2"], b2=d["b2"],
        )

