import numpy as np
import pandas as pd

class MF_SGD:
    """
    Basic Matrix Factorization for explicit ratings:
      pred(u,i) = global_mean + bu[u] + bi[i] + P[u]·Q[i]
    Trained with SGD on squared error + L2 regularization.
    """

    def __init__(self, n_factors=50, lr=0.01, reg=0.02, epochs=20, random_state=42):
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.epochs = epochs
        self.random_state = random_state

        self.user_to_idx = {}
        self.item_to_idx = {}
        self.idx_to_user = None
        self.idx_to_item = None

        self.P = None
        self.Q = None
        self.bu = None
        self.bi = None
        self.global_mean = 0.0

    def _index(self, df: pd.DataFrame):
        users = df["user_id"].unique()
        items = df["item_id"].unique()
        self.user_to_idx = {u: i for i, u in enumerate(users)}
        self.item_to_idx = {it: i for i, it in enumerate(items)}
        self.idx_to_user = np.array(users)
        self.idx_to_item = np.array(items)

    def fit(self, ratings_df: pd.DataFrame):
        if "rating" not in ratings_df.columns:
            raise ValueError("MF_SGD requires explicit 'rating' column.")

        self._index(ratings_df)

        rng = np.random.default_rng(self.random_state)
        n_users = len(self.user_to_idx)
        n_items = len(self.item_to_idx)

        self.P = 0.1 * rng.standard_normal((n_users, self.n_factors))
        self.Q = 0.1 * rng.standard_normal((n_items, self.n_factors))
        self.bu = np.zeros(n_users)
        self.bi = np.zeros(n_items)

        self.global_mean = float(ratings_df["rating"].mean())

        # SGD loop
        data = ratings_df[["user_id", "item_id", "rating"]].to_numpy()
        for epoch in range(self.epochs):
            rng.shuffle(data)
            for u_id, i_id, r in data:
                u = self.user_to_idx[u_id]
                i = self.item_to_idx[i_id]
                r = float(r)

                pred = self.predict_single(u_id, i_id)
                err = r - pred

                # gradients
                pu = self.P[u].copy()
                qi = self.Q[i].copy()

                self.bu[u] += self.lr * (err - self.reg * self.bu[u])
                self.bi[i] += self.lr * (err - self.reg * self.bi[i])

                self.P[u] += self.lr * (err * qi - self.reg * pu)
                self.Q[i] += self.lr * (err * pu - self.reg * qi)

        return self

    def predict_single(self, user_id, item_id) -> float:
        if user_id not in self.user_to_idx or item_id not in self.item_to_idx:
            # cold-start fallback
            return self.global_mean

        u = self.user_to_idx[user_id]
        i = self.item_to_idx[item_id]
        return (
            self.global_mean +
            self.bu[u] + self.bi[i] +
            float(self.P[u].dot(self.Q[i]))
        )

    def recommend_for_user(self, user_id, seen_item_ids=None, top_n=10):
        if user_id not in self.user_to_idx:
            return pd.DataFrame(columns=["item_id", "score"])

        seen = set(seen_item_ids or [])
        scores = []
        for item_id in self.idx_to_item:
            if item_id in seen:
                continue
            scores.append((item_id, self.predict_single(user_id, item_id)))

        scores.sort(key=lambda x: x[1], reverse=True)
        top = scores[:top_n]
        return pd.DataFrame(top, columns=["item_id", "score"])