import json
import os
import joblib
import pandas as pd

from config import Paths, TrainConfig
from data import load_interactions, load_items
from split import user_holdout_split
from features import build_tfidf
from models_content import ContentRecommender
from models_mf_sgd import MF_SGD
from models_hybrid import HybridRecommender
from evaluate import eval_ranking, rmse

def ensure_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def main():
    paths = Paths()
    cfg = TrainConfig()

    ensure_dirs(paths.processed_dir, paths.models_dir, paths.metrics_dir, paths.figures_dir, paths.recs_dir)

    interactions = load_interactions(paths.interactions_raw)
    items = load_items(paths.items_raw)

    train_df, test_df = user_holdout_split(interactions, cfg.test_ratio, cfg.random_state)

    # Content model
    tfidf_vec, tfidf_X = build_tfidf(items)
    content_model = ContentRecommender(items["item_id"].tolist(), tfidf_X)

    # Collaborative model (explicit ratings only)
    cf_model = None
    if "rating" in interactions.columns:
        cf_model = MF_SGD(
            n_factors=cfg.mf_factors,
            lr=cfg.mf_lr,
            reg=cfg.mf_reg,
            epochs=cfg.mf_epochs,
            random_state=cfg.random_state
        ).fit(train_df)

    # Evaluate baselines
    metrics = {}

    # Content ranking eval
    def content_recs(user_id, history, top_n):
        return content_model.recommend_for_user(history, seen_filter=True, top_n=top_n)

    metrics["content"] = eval_ranking(content_recs, test_df, train_df, k=cfg.k)

    # CF eval (ranking + RMSE if explicit)
    if cf_model is not None:
        def cf_recs(user_id, history, top_n):
            return cf_model.recommend_for_user(user_id, seen_item_ids=history, top_n=top_n)

        metrics["cf"] = eval_ranking(cf_recs, test_df, train_df, k=cfg.k)

        # RMSE on explicit ratings
        y_true, y_pred = [], []
        for row in test_df.itertuples(index=False):
            if hasattr(row, "rating"):
                y_true.append(float(row.rating))
                y_pred.append(float(cf_model.predict_single(row.user_id, row.item_id)))
        if y_true:
            metrics["cf"]["rmse"] = rmse(y_true, y_pred)

        # Hybrid
        hybrid = HybridRecommender(cf_model, content_model, alpha=cfg.hybrid_alpha)

        def hybrid_recs(user_id, history, top_n):
            return hybrid.recommend(user_id, history, top_n=top_n)

        metrics["hybrid"] = eval_ranking(hybrid_recs, test_df, train_df, k=cfg.k)

        joblib.dump(hybrid, os.path.join(paths.models_dir, "hybrid.pkl"))

    # Save artifacts
    joblib.dump(content_model, os.path.join(paths.models_dir, "content.pkl"))
    joblib.dump(tfidf_vec, os.path.join(paths.models_dir, "tfidf_vectorizer.pkl"))

    if cf_model is not None:
        joblib.dump(cf_model, os.path.join(paths.models_dir, "mf_sgd.pkl"))

    with open(os.path.join(paths.metrics_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Training complete. Metrics:")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()