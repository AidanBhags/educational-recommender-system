import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


INTERACTIONS_PATH = "data/processed/interactions_agg.csv"
ITEMS_PATH = "data/processed/items_final.csv"
METRICS_DIR = "metrics"
FIGURES_DIR = "figures"


def ensure_dirs():
    os.makedirs(METRICS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)


def load_data():
    interactions = pd.read_csv(INTERACTIONS_PATH)
    items = pd.read_csv(ITEMS_PATH)

    interactions["user_id"] = interactions["user_id"].astype(str)
    interactions["item_id"] = interactions["item_id"].astype(str)
    items["item_id"] = items["item_id"].astype(str)

    return interactions, items


def make_user_folds(interactions: pd.DataFrame, n_folds: int = 3, random_state: int = 42):
    rng = np.random.default_rng(random_state)
    users = interactions["user_id"].astype(str).unique()
    users = np.array(users, dtype=object)
    rng.shuffle(users)
    folds = np.array_split(users, n_folds)
    return [set(f) for f in folds]


def leave_some_out_split(test_users_df: pd.DataFrame, test_ratio: float = 0.2, random_state: int = 42):
    train_parts = []
    test_parts = []

    for user_id, group in test_users_df.groupby("user_id"):
        group = group.sample(frac=1, random_state=random_state)

        if len(group) < 5:
            train_parts.append(group)
            continue

        n_test = max(1, int(len(group) * test_ratio))
        test_group = group.iloc[:n_test]
        train_group = group.iloc[n_test:]

        if len(train_group) == 0:
            train_group = group.iloc[:-1]
            test_group = group.iloc[-1:]

        train_parts.append(train_group)
        test_parts.append(test_group)

    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else pd.DataFrame(columns=test_users_df.columns)
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame(columns=test_users_df.columns)
    return train_df, test_df


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
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
        self.item_matrix = None
        self.item_ids = None
        self.item_index = None

    def fit(self, items: pd.DataFrame):
        self.item_ids = items["item_id"].tolist()
        self.item_index = {item_id: idx for idx, item_id in enumerate(self.item_ids)}
        self.item_matrix = self.vectorizer.fit_transform(items["text"].fillna(""))
        return self

    def recommend(self, user_id, seen_items, top_n=10):
        seen_idxs = [self.item_index[i] for i in seen_items if i in self.item_index]

        if not seen_idxs:
            return pd.DataFrame(columns=["item_id", "score"])

        user_profile = np.asarray(self.item_matrix[seen_idxs].mean(axis=0))
        sims = cosine_similarity(user_profile, self.item_matrix).ravel()

        recs = pd.DataFrame({
            "item_id": self.item_ids,
            "score": sims
        })

        recs = recs[~recs["item_id"].isin(seen_items)]
        recs = recs.sort_values("score", ascending=False).head(top_n).reset_index(drop=True)
        return recs


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


def precision_recall_ndcg_at_k(recommended, relevant_set, k=10):
    rec_k = recommended[:k]
    hits = [1 if item in relevant_set else 0 for item in rec_k]

    precision = sum(hits) / k if k > 0 else 0.0
    recall = sum(hits) / len(relevant_set) if len(relevant_set) > 0 else 0.0

    dcg = 0.0
    for idx, rel in enumerate(hits, start=1):
        if rel:
            dcg += 1.0 / np.log2(idx + 1)

    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, ideal_hits + 1))
    ndcg = dcg / idcg if idcg > 0 else 0.0

    return precision, recall, ndcg


def evaluate_model(model, train_df: pd.DataFrame, test_df: pd.DataFrame, k=10):
    train_hist = train_df.groupby("user_id")["item_id"].apply(set).to_dict()
    test_truth = test_df.groupby("user_id")["item_id"].apply(set).to_dict()

    precisions = []
    recalls = []
    ndcgs = []

    for user_id, relevant_items in test_truth.items():
        seen_items = train_hist.get(user_id, set())
        recs = model.recommend(user_id, seen_items, top_n=k)
        recommended = recs["item_id"].tolist() if not recs.empty else []

        p, r, n = precision_recall_ndcg_at_k(recommended, relevant_items, k=k)
        precisions.append(p)
        recalls.append(r)
        ndcgs.append(n)

    return {
        f"precision@{k}": float(np.mean(precisions)) if precisions else 0.0,
        f"recall@{k}": float(np.mean(recalls)) if recalls else 0.0,
        f"ndcg@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "evaluated_users": int(len(test_truth))
    }


def evaluate_rmse(model, test_df: pd.DataFrame):
    y_true = []
    y_pred = []

    for row in test_df.itertuples(index=False):
        pred = model.predict_single(str(row.user_id), str(row.item_id))
        y_true.append(float(row.rating))
        y_pred.append(float(pred))

    if not y_true:
        return None

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def run_fold(train_df, test_df, items):
    popularity = PopularityRecommender().fit(train_df)
    content = ContentRecommender().fit(items)
    mf = MFSGD(n_factors=40, lr=0.01, reg=0.02, epochs=15, random_state=42).fit(train_df)
    hybrid = HybridRecommender(popularity, content, mf, alpha=0.15, beta=0.75, gamma=0.10)

    results = {
        "popularity": evaluate_model(popularity, train_df, test_df, k=10),
        "content": evaluate_model(content, train_df, test_df, k=10),
        "mf": evaluate_model(mf, train_df, test_df, k=10),
        "hybrid": evaluate_model(hybrid, train_df, test_df, k=10),
    }

    results["mf"]["rmse"] = evaluate_rmse(mf, test_df)
    return results


def aggregate_cv_results(fold_results):
    models = fold_results[0].keys()
    summary = {}

    for model_name in models:
        metric_names = fold_results[0][model_name].keys()
        summary[model_name] = {}

        for metric in metric_names:
            values = [fr[model_name][metric] for fr in fold_results if fr[model_name][metric] is not None]
            summary[model_name][metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }

    return summary


def plot_metric_bars(summary, metric_name, filename):
    eligible_models = [m for m in summary.keys() if metric_name in summary[m]]

    if not eligible_models:
        print(f"Skipping plot for {metric_name}: metric not found in any model.")
        return

    means = [summary[m][metric_name]["mean"] for m in eligible_models]
    stds = [summary[m][metric_name]["std"] for m in eligible_models]

    plt.figure(figsize=(8, 5))
    plt.bar(eligible_models, means, yerr=stds, capsize=4)
    plt.ylabel(metric_name)
    plt.title(f"Model Comparison: {metric_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=200)
    plt.close()


def main():
    ensure_dirs()
    interactions, items = load_data()

    folds = make_user_folds(interactions, n_folds=3, random_state=42)
    fold_results = []

    for fold_idx, test_users in enumerate(folds, start=1):
        print(f"Running fold {fold_idx}/3...")

        fold_df = interactions[interactions["user_id"].isin(test_users)].copy()
        train_part, test_part = leave_some_out_split(fold_df, test_ratio=0.2, random_state=42 + fold_idx)

        other_users_df = interactions[~interactions["user_id"].isin(test_users)].copy()
        train_df = pd.concat([other_users_df, train_part], ignore_index=True)
        test_df = test_part.copy()

        valid_train_items = set(train_df["item_id"])
        test_df = test_df[test_df["item_id"].isin(valid_train_items)].copy()

        fold_result = run_fold(train_df, test_df, items)
        fold_results.append(fold_result)

    summary = aggregate_cv_results(fold_results)

    with open(os.path.join(METRICS_DIR, "cv_results.json"), "w", encoding="utf-8") as f:
        json.dump({
            "fold_results": fold_results,
            "summary": summary
        }, f, indent=2)

    plot_metric_bars(summary, "precision@10", "precision_at_10.png")
    plot_metric_bars(summary, "recall@10", "recall_at_10.png")
    plot_metric_bars(summary, "ndcg@10", "ndcg_at_10.png")
    plot_metric_bars(summary, "rmse", "rmse.png")

    print("Cross-validation complete.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
