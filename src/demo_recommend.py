import os
import json
import joblib
import pandas as pd
from recommenders import ContentRecommender, HybridRecommender


MODELS_DIR = "models"
ITEMS_PATH = "data/processed/items_final.csv"
INTERACTIONS_PATH = "data/processed/interactions_agg.csv"
OUTPUT_DIR = "recommendations"


def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_assets():
    items = pd.read_csv(ITEMS_PATH)
    interactions = pd.read_csv(INTERACTIONS_PATH)

    items["item_id"] = items["item_id"].astype(str)
    interactions["user_id"] = interactions["user_id"].astype(str)
    interactions["item_id"] = interactions["item_id"].astype(str)

    item_lookup = items.set_index("item_id").to_dict(orient="index")

    content = joblib.load(os.path.join(MODELS_DIR, "content.pkl"))
    hybrid = joblib.load(os.path.join(MODELS_DIR, "hybrid.pkl"))

    return items, interactions, item_lookup, content, hybrid


def get_user_history(interactions: pd.DataFrame, user_id: str, top_n_recent: int = 10):
    user_df = interactions[interactions["user_id"] == str(user_id)].copy()

    if user_df.empty:
        return user_df

    sort_cols = [c for c in ["last_date", "mean_date", "first_date"] if c in user_df.columns]
    if sort_cols:
        user_df = user_df.sort_values(sort_cols, ascending=False)

    return user_df.head(top_n_recent)


def format_item(item_id: str, item_lookup: dict):
    meta = item_lookup.get(item_id, {})

    return {
        "item_id": item_id,
        "title": meta.get("title", ""),
        "activity_type": meta.get("activity_type", ""),
        "code_module": meta.get("code_module", ""),
        "code_presentation": meta.get("code_presentation", ""),
        "description": meta.get("description", ""),
        "tags": meta.get("tags", "")
    }


def make_recommendations(user_id: str, interactions: pd.DataFrame, item_lookup: dict, content_model, hybrid_model, top_n: int = 10):
    history = interactions[interactions["user_id"] == str(user_id)]["item_id"].astype(str).tolist()
    seen_items = set(history)

    content_recs = content_model.recommend(str(user_id), seen_items, top_n=top_n)
    hybrid_recs = hybrid_model.recommend(str(user_id), seen_items, top_n=top_n)

    content_out = []
    for row in content_recs.itertuples(index=False):
        item = format_item(str(row.item_id), item_lookup)
        item["score"] = float(row.score)
        content_out.append(item)

    hybrid_out = []
    for row in hybrid_recs.itertuples(index=False):
        item = format_item(str(row.item_id), item_lookup)
        item["score"] = float(row.score)
        hybrid_out.append(item)

    return content_out, hybrid_out


def save_outputs(user_id: str, history_df: pd.DataFrame, content_out: list, hybrid_out: list):
    history_path = os.path.join(OUTPUT_DIR, f"user_{user_id}_history.csv")
    content_path = os.path.join(OUTPUT_DIR, f"user_{user_id}_content_recommendations.csv")
    hybrid_path = os.path.join(OUTPUT_DIR, f"user_{user_id}_hybrid_recommendations.csv")
    summary_path = os.path.join(OUTPUT_DIR, f"user_{user_id}_recommendation_summary.json")

    history_df.to_csv(history_path, index=False)
    pd.DataFrame(content_out).to_csv(content_path, index=False)
    pd.DataFrame(hybrid_out).to_csv(hybrid_path, index=False)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "user_id": str(user_id),
                "history_file": history_path,
                "content_recommendations_file": content_path,
                "hybrid_recommendations_file": hybrid_path,
                "content_recommendations_count": len(content_out),
                "hybrid_recommendations_count": len(hybrid_out)
            },
            f,
            indent=2
        )

    return history_path, content_path, hybrid_path, summary_path


def print_section(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def print_history(history_df: pd.DataFrame):
    if history_df.empty:
        print("No interaction history found for this user.")
        return

    cols = [c for c in ["item_id", "code_module", "activity_type", "strength", "interaction_count", "last_date"] if c in history_df.columns]
    print(history_df[cols].to_string(index=False))


def print_recommendations(title: str, recs: list):
    print_section(title)
    if not recs:
        print("No recommendations available.")
        return

    for idx, rec in enumerate(recs, start=1):
        print(f"{idx}. {rec['title']}")
        print(f"   item_id: {rec['item_id']}")
        print(f"   score: {rec['score']:.4f}")
        print(f"   module: {rec['code_module']} | presentation: {rec['code_presentation']} | type: {rec['activity_type']}")
        print(f"   tags: {rec['tags']}")
        print()


def main():
    ensure_dirs()

    user_id = input("Enter user_id: ").strip()

    items, interactions, item_lookup, content_model, hybrid_model = load_assets()

    history_df = get_user_history(interactions, user_id, top_n_recent=10)
    content_out, hybrid_out = make_recommendations(
        user_id=user_id,
        interactions=interactions,
        item_lookup=item_lookup,
        content_model=content_model,
        hybrid_model=hybrid_model,
        top_n=10
    )

    print_section(f"Recent History for User {user_id}")
    print_history(history_df)

    print_recommendations("Top 10 Content-Based Recommendations", content_out)
    print_recommendations("Top 10 Hybrid Recommendations", hybrid_out)

    history_path, content_path, hybrid_path, summary_path = save_outputs(
        user_id=user_id,
        history_df=history_df,
        content_out=content_out,
        hybrid_out=hybrid_out
    )

    print_section("Saved Outputs")
    print(history_path)
    print(content_path)
    print(hybrid_path)
    print(summary_path)


if __name__ == "__main__":
    main()
