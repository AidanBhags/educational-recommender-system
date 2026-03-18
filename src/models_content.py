import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class ContentRecommender:
    def __init__(self, item_ids, tfidf_matrix):
        self.item_ids = np.array(item_ids)
        self.X = tfidf_matrix
        self.id_to_idx = {iid: i for i, iid in enumerate(self.item_ids)}

    def score_similar_items(self, seed_item_ids, top_n=50):
        # Average the vectors of seed items, compute cosine similarity to all items
        idxs = [self.id_to_idx[i] for i in seed_item_ids if i in self.id_to_idx]
        if not idxs:
            return pd.DataFrame(columns=["item_id", "score"])

        seed_vec = self.X[idxs].mean(axis=0)
        sims = cosine_similarity(seed_vec, self.X).ravel()

        out = pd.DataFrame({"item_id": self.item_ids, "score": sims})
        out = out.sort_values("score", ascending=False).head(top_n)
        return out

    def recommend_for_user(self, user_history_item_ids, seen_filter=True, top_n=10):
        scores = self.score_similar_items(user_history_item_ids, top_n=top_n + len(user_history_item_ids) + 20)
        if seen_filter:
            seen = set(user_history_item_ids)
            scores = scores[~scores["item_id"].isin(seen)]
        return scores.head(top_n).reset_index(drop=True)