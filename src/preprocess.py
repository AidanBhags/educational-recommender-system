import os
import numpy as np
import pandas as pd


RAW_INTERACTIONS = "data/raw/interactions.csv"
RAW_ITEMS = "data/raw/items.csv"
PROCESSED_DIR = "data/processed"


def ensure_dirs():
    os.makedirs(PROCESSED_DIR, exist_ok=True)


def min_max_scale(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    min_val = s.min()
    max_val = s.max()
    if max_val == min_val:
        return pd.Series(np.full(len(s), 3.0), index=s.index)
    scaled = 1.0 + 4.0 * (s - min_val) / (max_val - min_val)
    return scaled.clip(1.0, 5.0)


def load_data():
    interactions = pd.read_csv(
        RAW_INTERACTIONS,
        usecols=[
            "user_id",
            "item_id",
            "strength",
            "date",
            "activity_type",
            "code_module",
            "code_presentation",
        ],
        dtype={
            "user_id": "string",
            "item_id": "string",
            "activity_type": "string",
            "code_module": "string",
            "code_presentation": "string",
        },
    )

    items = pd.read_csv(
        RAW_ITEMS,
        dtype={
            "item_id": "string",
            "activity_type": "string",
            "code_module": "string",
            "code_presentation": "string",
            "title": "string",
            "description": "string",
            "tags": "string",
        },
    )

    interactions["strength"] = pd.to_numeric(interactions["strength"], errors="coerce").fillna(0)
    interactions["date"] = pd.to_numeric(interactions["date"], errors="coerce")

    return interactions, items


def bucket_week(value):
    if pd.isna(value):
        return "unknown_week"
    value = float(value)
    if value < 0:
        return "prestart"
    if value <= 4:
        return "early"
    if value <= 12:
        return "mid"
    return "late"


def bucket_length(value):
    if pd.isna(value):
        return "unknown_length"
    value = float(value)
    if value < 120:
        return "short_presentation"
    if value < 240:
        return "medium_presentation"
    return "long_presentation"


def bucket_popularity(rank_pct):
    if pd.isna(rank_pct):
        return "unknown_popularity"
    if rank_pct <= 0.2:
        return "very_popular"
    if rank_pct <= 0.5:
        return "popular"
    if rank_pct <= 0.8:
        return "moderate_popularity"
    return "niche"


def preprocess_items(items: pd.DataFrame, interactions: pd.DataFrame) -> pd.DataFrame:
    items = items.copy()

    for col in ["title", "description", "tags", "activity_type", "code_module", "code_presentation"]:
        if col not in items.columns:
            items[col] = ""

    items["week_from"] = pd.to_numeric(items.get("week_from"), errors="coerce")
    items["week_to"] = pd.to_numeric(items.get("week_to"), errors="coerce")
    items["module_presentation_length"] = pd.to_numeric(
        items.get("module_presentation_length"), errors="coerce"
    )

    # Aggregate raw interactions at item level for richer features
    item_stats = (
        interactions.groupby("item_id", as_index=False)
        .agg(
            item_total_strength=("strength", "sum"),
            item_event_count=("strength", "size"),
            item_avg_strength=("strength", "mean"),
            item_last_date=("date", "max"),
        )
    )

    item_stats["item_strength_log"] = np.log1p(item_stats["item_total_strength"])
    item_stats["item_event_log"] = np.log1p(item_stats["item_event_count"])

    # Percentile-based popularity band
    item_stats["popularity_rank_pct"] = item_stats["item_total_strength"].rank(pct=True, ascending=False)
    item_stats["popularity_band"] = item_stats["popularity_rank_pct"].apply(bucket_popularity)

    items = items.merge(item_stats, on="item_id", how="left")

    # Fill missing stats
    for col in [
        "item_total_strength",
        "item_event_count",
        "item_avg_strength",
        "item_last_date",
        "item_strength_log",
        "item_event_log",
    ]:
        items[col] = pd.to_numeric(items[col], errors="coerce").fillna(0)

    items["popularity_band"] = items["popularity_band"].fillna("unknown_popularity")

    # Derived labels
    items["week_from_bucket"] = items["week_from"].apply(bucket_week)
    items["week_to_bucket"] = items["week_to"].apply(bucket_week)
    items["presentation_length_bucket"] = items["module_presentation_length"].apply(bucket_length)

    items["resource_family"] = (
        items["activity_type"].fillna("").astype(str)
        + "_"
        + items["code_module"].fillna("").astype(str)
    )

    items["text"] = (
        "title " + items["title"].fillna("").astype(str) + " "
        + "description " + items["description"].fillna("").astype(str) + " "
        + "tags " + items["tags"].fillna("").astype(str) + " "
        + "activity_type " + items["activity_type"].fillna("").astype(str).str.replace("_", " ", regex=False) + " "
        + "module " + items["code_module"].fillna("").astype(str) + " "
        + "presentation " + items["code_presentation"].fillna("").astype(str) + " "
        + "week_from_bucket " + items["week_from_bucket"].fillna("").astype(str) + " "
        + "week_to_bucket " + items["week_to_bucket"].fillna("").astype(str) + " "
        + "presentation_length_bucket " + items["presentation_length_bucket"].fillna("").astype(str) + " "
        + "resource_family " + items["resource_family"].fillna("").astype(str) + " "
        + "popularity_band " + items["popularity_band"].fillna("").astype(str) + " "
        + "item_strength_log " + items["item_strength_log"].round(2).astype(str) + " "
        + "item_event_log " + items["item_event_log"].round(2).astype(str)
    ).str.strip()

    items = items.drop_duplicates(subset=["item_id"]).reset_index(drop=True)
    return items


def aggregate_interactions(interactions: pd.DataFrame) -> pd.DataFrame:
    df = interactions.copy()
    df = df.sort_values(["user_id", "item_id", "date"], kind="mergesort")

    grouped = (
        df.groupby(["user_id", "item_id"], as_index=False, sort=False)
        .agg(
            strength=("strength", "sum"),
            interaction_count=("strength", "size"),
            first_date=("date", "min"),
            last_date=("date", "max"),
            mean_date=("date", "mean"),
            activity_type=("activity_type", "first"),
            code_module=("code_module", "first"),
            code_presentation=("code_presentation", "first"),
        )
    )

    grouped["strength_log"] = np.log1p(grouped["strength"])
    grouped["engagement_score"] = grouped["strength_log"] + np.log1p(grouped["interaction_count"]) * 0.25
    grouped["rating"] = min_max_scale(grouped["engagement_score"]).round(3)

    return grouped


def filter_sparse_once(interactions: pd.DataFrame, min_user_interactions: int = 5, min_item_interactions: int = 5):
    user_counts = interactions["user_id"].value_counts()
    valid_users = user_counts[user_counts >= min_user_interactions].index
    df = interactions[interactions["user_id"].isin(valid_users)].copy()

    item_counts = df["item_id"].value_counts()
    valid_items = item_counts[item_counts >= min_item_interactions].index
    df = df[df["item_id"].isin(valid_items)].copy()

    return df


def main():
    ensure_dirs()

    print("Loading raw files...")
    interactions, items = load_data()
    print(f"Loaded interactions: {interactions.shape}")
    print(f"Loaded items: {items.shape}")

    print("Aggregating repeated user-item events...")
    interactions_agg = aggregate_interactions(interactions)
    print(f"After aggregation: {interactions_agg.shape}")

    print("Filtering sparse users/items...")
    interactions_agg = filter_sparse_once(interactions_agg, min_user_interactions=5, min_item_interactions=5)
    print(f"After sparsity filter: {interactions_agg.shape}")

    print("Building enriched item features...")
    items = preprocess_items(items, interactions)
    valid_item_ids = set(interactions_agg["item_id"])
    items = items[items["item_id"].isin(valid_item_ids)].copy()
    interactions_agg = interactions_agg[interactions_agg["item_id"].isin(set(items["item_id"]))].copy()

    interactions_out = os.path.join(PROCESSED_DIR, "interactions_agg.csv")
    items_out = os.path.join(PROCESSED_DIR, "items_final.csv")

    print("Saving files...")
    interactions_agg.to_csv(interactions_out, index=False)
    items.to_csv(items_out, index=False)

    print("Preprocessing complete.")
    print(f"Saved: {interactions_out}")
    print(f"Saved: {items_out}")
    print(f"Interactions shape: {interactions_agg.shape}")
    print(f"Items shape: {items.shape}")
    print(f"Unique users: {interactions_agg['user_id'].nunique()}")
    print(f"Unique items: {interactions_agg['item_id'].nunique()}")
    print(f"Average interactions per user: {round(len(interactions_agg) / max(1, interactions_agg['user_id'].nunique()), 2)}")


if __name__ == "__main__":
    main()