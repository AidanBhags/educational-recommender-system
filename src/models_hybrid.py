import pandas as pd

class HybridRecommender:
    def __init__(self, cf_model, content_model, alpha: float = 0.7):
        self.cf = cf_model
        self.content = content_model
        self.alpha = float(alpha)

    def recommend(self, user_id, user_history_item_ids, top_n=10):
        cf = self.cf.recommend_for_user(user_id, seen_item_ids=user_history_item_ids, top_n=top_n * 5)
        cb = self.content.recommend_for_user(user_history_item_ids, seen_filter=True, top_n=top_n * 5)

        # normalize scores to [0,1] for blending
        def norm(df):
            if df.empty:
                return df
            s = df["score"].astype(float)
            rng = s.max() - s.min()
            df = df.copy()
            df["score"] = 0.0 if rng == 0 else (s - s.min()) / rng
            return df

        cf = norm(cf)
        cb = norm(cb)

        merged = pd.merge(cf, cb, on="item_id", how="outer", suffixes=("_cf", "_cb")).fillna(0.0)
        merged["score"] = self.alpha * merged["score_cf"] + (1 - self.alpha) * merged["score_cb"]
        merged = merged.sort_values("score", ascending=False).head(top_n)
        return merged[["item_id", "score"]].reset_index(drop=True)