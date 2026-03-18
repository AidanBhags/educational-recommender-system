import numpy as np
import pandas as pd

def precision_recall_at_k(recommended, relevant_set, k):
    rec_k = recommended[:k]
    rec_set = set(rec_k)
    if k == 0:
        return 0.0, 0.0
    hits = len(rec_set & relevant_set)
    precision = hits / k
    recall = hits / max(1, len(relevant_set))
    return precision, recall

def ndcg_at_k(recommended, relevant_set, k):
    rec_k = recommended[:k]
    dcg = 0.0
    for i, item in enumerate(rec_k, start=1):
        if item in relevant_set:
            dcg += 1.0 / np.log2(i + 1)
    ideal_hits = min(k, len(relevant_set))
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, ideal_hits + 1))
    return 0.0 if idcg == 0 else dcg / idcg

def rmse(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def eval_ranking(model_recommend_fn, test_df: pd.DataFrame, train_df: pd.DataFrame, k: int = 10):
    """
    model_recommend_fn: function(user_id, history_item_ids, top_n)-> DataFrame(item_id, score)
    For each user in test set:
      - relevant items = test items
      - history = train items
      - recommend top_n=k
    """
    train_hist = train_df.groupby("user_id")["item_id"].apply(list).to_dict()
    test_items = test_df.groupby("user_id")["item_id"].apply(set).to_dict()

    precisions, recalls, ndcgs = [], [], []
    for user_id, relevant in test_items.items():
        history = train_hist.get(user_id, [])
        recs = model_recommend_fn(user_id, history, top_n=k)
        recommended = recs["item_id"].tolist() if not recs.empty else []

        p, r = precision_recall_at_k(recommended, relevant, k)
        n = ndcg_at_k(recommended, relevant, k)

        precisions.append(p)
        recalls.append(r)
        ndcgs.append(n)

    return {
        f"precision@{k}": float(np.mean(precisions)) if precisions else 0.0,
        f"recall@{k}": float(np.mean(recalls)) if recalls else 0.0,
        f"ndcg@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "n_users_eval": int(len(test_items))
    }