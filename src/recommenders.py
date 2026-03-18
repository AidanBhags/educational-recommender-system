import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class PopularityRecommender:
    def __init__(self):
        self.popularity = None

    def fit(self, train_df: pd.DataFrame):
        pop = (
            train_df.groupby("item_id")
            .agg(
                popularity_score=("strength", "sum"),
                interaction_count=("interaction_count", "sum")
            )
            .reset_index()
        )
        pop["score"] = pop["popularity_score"] + np.log1p(pop["interaction_count"])
        self.popularity = pop.sort_values("score", ascending=False).reset_index(drop=True)
        return self

    def recommend(self, user_id, seen_items, top_n=10):
        recs = self.popularity[~self.popularity["item_id"].isin(seen_items)].head(top_n)
        return recs[["item_id", "score"]].reset_index(drop=True)


class ContentRecommender:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=8000,
            ngram_range=(1, 2),
            stop_words="english"
        )
        self.item_matrix = None
        self.items_df = None
        self.item_ids = None
        self.item_index = None

    def fit(self, items: pd.DataFrame):
        self.items_df = items.copy().reset_index(drop=True)
        self.item_ids = self.items_df["item_id"].astype(str).tolist()
        self.item_index = {item_id: idx for idx, item_id in enumerate(self.item_ids)}
        self.item_matrix = self.vectorizer.fit_transform(self.items_df["text"].fillna("").astype(str))
        return self

    def _diversity_rerank(self, recs: pd.DataFrame, top_n: int = 10):
        if recs.empty:
            return recs

        selected_rows = []
        activity_counts = {}
        module_counts = {}

        for row in recs.itertuples(index=False):
            activity = getattr(row, "activity_type", "")
            module = getattr(row, "code_module", "")

            if activity_counts.get(activity, 0) >= 4:
                continue
            if module_counts.get(module, 0) >= 8:
                continue

            selected_rows.append(row)
            activity_counts[activity] = activity_counts.get(activity, 0) + 1
            module_counts[module] = module_counts.get(module, 0) + 1

            if len(selected_rows) >= top_n:
                break

        if len(selected_rows) < top_n:
            already = {r.item_id for r in selected_rows}
            for row in recs.itertuples(index=False):
                if row.item_id in already:
                    continue
                selected_rows.append(row)
                if len(selected_rows) >= top_n:
                    break

        return pd.DataFrame(selected_rows)

    def recommend(self, user_id, seen_items, top_n=10):
        seen_idxs = [self.item_index[i] for i in seen_items if i in self.item_index]

        if not seen_idxs:
            return pd.DataFrame(columns=["item_id", "score"])

        user_profile = np.asarray(self.item_matrix[seen_idxs].mean(axis=0))
        sims = cosine_similarity(user_profile, self.item_matrix).ravel()

        recs = self.items_df.copy()
        recs["score"] = sims
        recs = recs[~recs["item_id"].isin(seen_items)].copy()
        recs = recs.sort_values("score", ascending=False).reset_index(drop=True)

        # Pull a larger candidate pool, then rerank for diversity
        recs = recs.head(top_n * 10).copy()
        recs = self._diversity_rerank(recs, top_n=top_n)

        return recs[["item_id", "score"]].reset_index(drop=True)


class MFSGD:
    def __init__(self, n_factors=40, lr=0.01, reg=0.02, epochs=15, random_state=42):
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.epochs = epochs
        self.random_state = random_state

        self.user_to_idx = {}
        self.item_to_idx = {}
        self.idx_to_item = None

        self.P = None
        self.Q = None
        self.bu = None
        self.bi = None
        self.global_mean = 0.0

    def fit(self, train_df: pd.DataFrame):
        users = train_df["user_id"].unique()
        items = train_df["item_id"].unique()

        self.user_to_idx = {u: i for i, u in enumerate(users)}
        self.item_to_idx = {i: j for j, i in enumerate(items)}
        self.idx_to_item = np.array(items)

        rng = np.random.default_rng(self.random_state)

        n_users = len(users)
        n_items = len(items)

        self.P = 0.1 * rng.standard_normal((n_users, self.n_factors))
        self.Q = 0.1 * rng.standard_normal((n_items, self.n_factors))
        self.bu = np.zeros(n_users)
        self.bi = np.zeros(n_items)
        self.global_mean = float(train_df["rating"].mean())

        rows = train_df[["user_id", "item_id", "rating"]].to_numpy()

        for _ in range(self.epochs):
            rng.shuffle(rows)

            for user_id, item_id, rating in rows:
                u = self.user_to_idx[user_id]
                i = self.item_to_idx[item_id]
                r = float(rating)

                pred = self.predict_single(user_id, item_id)
                err = r - pred

                pu = self.P[u].copy()
                qi = self.Q[i].copy()

                self.bu[u] += self.lr * (err - self.reg * self.bu[u])
                self.bi[i] += self.lr * (err - self.reg * self.bi[i])

                self.P[u] += self.lr * (err * qi - self.reg * pu)
                self.Q[i] += self.lr * (err * pu - self.reg * qi)

        return self

    def predict_single(self, user_id, item_id):
        if user_id not in self.user_to_idx or item_id not in self.item_to_idx:
            return self.global_mean

        u = self.user_to_idx[user_id]
        i = self.item_to_idx[item_id]
        return self.global_mean + self.bu[u] + self.bi[i] + float(self.P[u].dot(self.Q[i]))

    def recommend(self, user_id, seen_items, top_n=10):
        if user_id not in self.user_to_idx:
            return pd.DataFrame(columns=["item_id", "score"])

        rows = []
        for item_id in self.idx_to_item:
            if item_id in seen_items:
                continue
            rows.append((item_id, self.predict_single(user_id, item_id)))

        recs = pd.DataFrame(rows, columns=["item_id", "score"])
        recs = recs.sort_values("score", ascending=False).head(top_n).reset_index(drop=True)
        return recs


class HybridRecommender:
    def __init__(self, popularity, content, mf, alpha=0.15, beta=0.75, gamma=0.10):
        self.popularity = popularity
        self.content = content
        self.mf = mf
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    @staticmethod
    def normalize_scores(df):
        if df.empty:
            return df
        df = df.copy()
        s = df["score"].astype(float)
        rng = s.max() - s.min()
        df["score"] = 0.0 if rng == 0 else (s - s.min()) / rng
        return df

    def recommend(self, user_id, seen_items, top_n=10):
        pop = self.normalize_scores(self.popularity.recommend(user_id, seen_items, top_n=top_n * 5))
        con = self.normalize_scores(self.content.recommend(user_id, seen_items, top_n=top_n * 5))
        mf = self.normalize_scores(self.mf.recommend(user_id, seen_items, top_n=top_n * 5))

        merged = pd.merge(pop, con, on="item_id", how="outer", suffixes=("_pop", "_con"))
        merged = pd.merge(merged, mf, on="item_id", how="outer")
        merged = merged.rename(columns={"score": "score_mf"}).fillna(0.0)

        merged["score"] = (
            self.gamma * merged.get("score_pop", 0.0)
            + self.beta * merged.get("score_con", 0.0)
            + self.alpha * merged.get("score_mf", 0.0)
        )

        merged = merged[~merged["item_id"].isin(seen_items)]
        merged = merged.sort_values("score", ascending=False).head(top_n).reset_index(drop=True)
        return merged[["item_id", "score"]]
