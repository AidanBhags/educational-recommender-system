import os
import sys
import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException


# -------------------------------------------------------
# Make sure Python can find the /src folder
# -------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")

if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)


# Import models AFTER path fix
from recommenders import ContentRecommender, HybridRecommender


# -------------------------------------------------------
# Paths
# -------------------------------------------------------
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
ITEMS_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "items_final.csv")
INTERACTIONS_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "interactions_agg.csv")


# -------------------------------------------------------
# FastAPI
# -------------------------------------------------------
app = FastAPI(
    title="Educational Recommendation API",
    description="API for personalised learning resource recommendations",
    version="1.0.0"
)


# -------------------------------------------------------
# Load data
# -------------------------------------------------------
items = pd.read_csv(ITEMS_PATH)
interactions = pd.read_csv(INTERACTIONS_PATH)

items["item_id"] = items["item_id"].astype(str)
interactions["user_id"] = interactions["user_id"].astype(str)
interactions["item_id"] = interactions["item_id"].astype(str)

item_lookup = items.set_index("item_id").to_dict(orient="index")


# -------------------------------------------------------
# Load trained models
# -------------------------------------------------------
content_model = joblib.load(os.path.join(MODELS_DIR, "content.pkl"))
hybrid_model = joblib.load(os.path.join(MODELS_DIR, "hybrid.pkl"))


# -------------------------------------------------------
# Helper
# -------------------------------------------------------
def format_item(item_id, score):

    meta = item_lookup.get(item_id, {})

    return {
        "item_id": item_id,
        "score": float(score),
        "title": meta.get("title", ""),
        "activity_type": meta.get("activity_type", ""),
        "code_module": meta.get("code_module", ""),
        "code_presentation": meta.get("code_presentation", ""),
        "description": meta.get("description", ""),
        "tags": meta.get("tags", "")
    }


# -------------------------------------------------------
# Root endpoint
# -------------------------------------------------------
@app.get("/")
def root():

    return {
        "message": "Educational Recommendation API running",
        "endpoints": {
            "user_history": "/users/{user_id}/history",
            "content_recommendations": "/recommend/content/{user_id}",
            "hybrid_recommendations": "/recommend/hybrid/{user_id}"
        }
    }


# -------------------------------------------------------
# User history
# -------------------------------------------------------
@app.get("/users/{user_id}/history")
def get_user_history(user_id: str, limit: int = 10):

    user_df = interactions[interactions["user_id"] == str(user_id)].copy()

    if user_df.empty:
        raise HTTPException(status_code=404, detail="User not found")

    sort_cols = [c for c in ["last_date", "mean_date", "first_date"] if c in user_df.columns]

    if sort_cols:
        user_df = user_df.sort_values(sort_cols, ascending=False)

    cols = [
        c for c in [
            "item_id",
            "code_module",
            "activity_type",
            "strength",
            "interaction_count",
            "last_date"
        ] if c in user_df.columns
    ]

    return {
        "user_id": user_id,
        "history": user_df[cols].head(limit).to_dict(orient="records")
    }


# -------------------------------------------------------
# Content recommender
# -------------------------------------------------------
@app.get("/recommend/content/{user_id}")
def recommend_content(user_id: str, top_n: int = 10):

    history = interactions[interactions["user_id"] == str(user_id)]["item_id"].astype(str).tolist()

    if not history:
        raise HTTPException(status_code=404, detail="User not found or no history")

    seen_items = set(history)

    recs = content_model.recommend(str(user_id), seen_items, top_n=top_n)

    return {
        "user_id": user_id,
        "model": "content",
        "recommendations": [
            format_item(str(row.item_id), float(row.score))
            for row in recs.itertuples(index=False)
        ]
    }


# -------------------------------------------------------
# Hybrid recommender
# -------------------------------------------------------
@app.get("/recommend/hybrid/{user_id}")
def recommend_hybrid(user_id: str, top_n: int = 10):

    history = interactions[interactions["user_id"] == str(user_id)]["item_id"].astype(str).tolist()

    if not history:
        raise HTTPException(status_code=404, detail="User not found or no history")

    seen_items = set(history)

    recs = hybrid_model.recommend(str(user_id), seen_items, top_n=top_n)

    return {
        "user_id": user_id,
        "model": "hybrid",
        "recommendations": [
            format_item(str(row.item_id), float(row.score))
            for row in recs.itertuples(index=False)
        ]
    }