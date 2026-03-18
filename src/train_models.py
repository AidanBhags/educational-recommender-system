import json
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from recommenders import PopularityRecommender, ContentRecommender, MFSGD, HybridRecommender


INTERACTIONS_PATH = "data/processed/interactions_agg.csv"
ITEMS_PATH = "data/processed/items_final.csv"
MODELS_DIR = "models"
METRICS_DIR = "metrics"


def ensure_dirs():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)


def load_data():
    interactions = pd.read_csv(INTERACTIONS_PATH)
    items = pd.read_csv(ITEMS_PATH)

    interactions["user_id"] = interactions["user_id"].astype(str)
    interactions["item_id"] = interactions["item_id"].astype(str)
    items["item_id"] = items["item_id"].astype(str)

    return interactions, items


def user_train_test_split(interactions: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    rng = np.random.default_rng(random_state)

    train_parts = []
    test_parts = []

    for user_id, group in interactions.groupby("user_id"):
        group = group.sample(frac=1, random_state=random_state)

        if len(group) < 5:
            train_parts.append(group)
            continue

        n_test = max(1, int(len(group) * test_size))
        test_group = group.iloc[:n_test]
        train_group = group.iloc[n_test:]

        if len(train_group) == 0:
            train_group = group.iloc[:-1]
            test_group = group.iloc[-1:]

        train_parts.append(train_group)
        test_parts.append(test_group)

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame(columns=interactions.columns)

    return train_df, test_df

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
    if not hasattr(model, "predict_single"):
        return None

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


def main():
    ensure_dirs()

    interactions, items = load_data()
    train_df, test_df = user_train_test_split(interactions, test_size=0.2, random_state=42)

    popularity = PopularityRecommender().fit(train_df)
    content = ContentRecommender().fit(items)
    mf = MFSGD(n_factors=40, lr=0.01, reg=0.02, epochs=15, random_state=42).fit(train_df)
    hybrid = HybridRecommender(popularity, content, mf, alpha=0.15, beta=0.75, gamma=0.10)

    metrics = {
        "popularity": evaluate_model(popularity, train_df, test_df, k=10),
        "content": evaluate_model(content, train_df, test_df, k=10),
        "mf": evaluate_model(mf, train_df, test_df, k=10),
        "hybrid": evaluate_model(hybrid, train_df, test_df, k=10),
    }

    mf_rmse = evaluate_rmse(mf, test_df)
    if mf_rmse is not None:
        metrics["mf"]["rmse"] = mf_rmse

    joblib.dump(popularity, os.path.join(MODELS_DIR, "popularity.pkl"))
    joblib.dump(content, os.path.join(MODELS_DIR, "content.pkl"))
    joblib.dump(mf, os.path.join(MODELS_DIR, "mf.pkl"))
    joblib.dump(hybrid, os.path.join(MODELS_DIR, "hybrid.pkl"))

    with open(os.path.join(METRICS_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Training complete.")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()