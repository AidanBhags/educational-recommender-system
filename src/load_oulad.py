import os
import numpy as np
import pandas as pd


OULAD_DIR = "data/oulad"
OUTPUT_DIR = "data/raw"


def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def min_max_scale(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    min_val = s.min()
    max_val = s.max()
    if max_val == min_val:
        return pd.Series(np.full(len(s), 3.0), index=s.index)
    scaled = 1.0 + 4.0 * (s - min_val) / (max_val - min_val)
    return scaled.clip(1.0, 5.0)


def build_items(vle: pd.DataFrame, courses: pd.DataFrame) -> pd.DataFrame:
    # Merge VLE resources with course/module metadata
    items = vle.merge(
        courses,
        on=["code_module", "code_presentation"],
        how="left"
    ).copy()

    # Build unique item id
    items["item_id"] = (
        items["code_module"].astype(str) + "_" +
        items["code_presentation"].astype(str) + "_" +
        items["id_site"].astype(str)
    )

    # Build human-readable text fields for content-based filtering
    items["title"] = (
        items["activity_type"].astype(str).str.replace("_", " ", regex=False).str.title()
        + " Resource"
    )

    items["description"] = (
        "Educational resource of type "
        + items["activity_type"].astype(str).str.replace("_", " ", regex=False)
        + " for module "
        + items["code_module"].astype(str)
        + " in presentation "
        + items["code_presentation"].astype(str)
        + ". Available from week "
        + items["week_from"].fillna(-1).astype(int).astype(str)
        + " to week "
        + items["week_to"].fillna(-1).astype(int).astype(str)
        + ". Module presentation length: "
        + items["module_presentation_length"].fillna(-1).astype(int).astype(str)
        + " days."
    )

    items["tags"] = (
        items["activity_type"].astype(str) + ","
        + items["code_module"].astype(str) + ","
        + items["code_presentation"].astype(str)
    )

    keep_cols = [
        "item_id",
        "id_site",
        "code_module",
        "code_presentation",
        "activity_type",
        "week_from",
        "week_to",
        "module_presentation_length",
        "title",
        "description",
        "tags"
    ]

    items = items[keep_cols].drop_duplicates(subset=["item_id"]).reset_index(drop=True)
    return items


def build_interactions(student_vle: pd.DataFrame, vle: pd.DataFrame) -> pd.DataFrame:
    # Join student interactions to VLE resource metadata so each interaction maps to a unique item
    interactions = student_vle.merge(
        vle[["id_site", "code_module", "code_presentation", "activity_type"]],
        on=["id_site", "code_module", "code_presentation"],
        how="left"
    ).copy()

    interactions["user_id"] = interactions["id_student"].astype(str)

    interactions["item_id"] = (
        interactions["code_module"].astype(str) + "_" +
        interactions["code_presentation"].astype(str) + "_" +
        interactions["id_site"].astype(str)
    )

    # Raw interaction strength from clicks
    interactions["strength"] = pd.to_numeric(interactions["sum_click"], errors="coerce").fillna(0)

    # Remove zero/negative interactions just in case
    interactions = interactions[interactions["strength"] > 0].copy()

    # Optional log transform for highly skewed click counts
    interactions["strength_log"] = np.log1p(interactions["strength"])

    # Create an explicit-style rating for MF-SGD
    # This is a pragmatic transformation from implicit signal -> pseudo-rating
    interactions["rating"] = min_max_scale(interactions["strength_log"]).round(3)

    keep_cols = [
        "user_id",
        "item_id",
        "strength",
        "strength_log",
        "rating",
        "date",
        "activity_type",
        "code_module",
        "code_presentation"
    ]

    interactions = interactions[keep_cols].drop_duplicates().reset_index(drop=True)
    return interactions


def main():
    ensure_dirs()

    courses_path = os.path.join(OULAD_DIR, "courses.csv")
    vle_path = os.path.join(OULAD_DIR, "vle.csv")
    student_vle_path = os.path.join(OULAD_DIR, "studentVle.csv")

    courses = pd.read_csv(courses_path)
    vle = pd.read_csv(vle_path)
    student_vle = pd.read_csv(student_vle_path)

    items = build_items(vle, courses)
    interactions = build_interactions(student_vle, vle)

    # Filter interactions to only items that exist in items.csv
    valid_items = set(items["item_id"])
    interactions = interactions[interactions["item_id"].isin(valid_items)].copy()

    # Optional: remove users with too few interactions
    user_counts = interactions["user_id"].value_counts()
    valid_users = user_counts[user_counts >= 5].index
    interactions = interactions[interactions["user_id"].isin(valid_users)].copy()

    # Optional: remove items with too few interactions
    item_counts = interactions["item_id"].value_counts()
    valid_item_ids = item_counts[item_counts >= 5].index
    interactions = interactions[interactions["item_id"].isin(valid_item_ids)].copy()
    items = items[items["item_id"].isin(set(valid_item_ids))].copy()

    interactions_out = os.path.join(OUTPUT_DIR, "interactions.csv")
    items_out = os.path.join(OUTPUT_DIR, "items.csv")

    interactions.to_csv(interactions_out, index=False)
    items.to_csv(items_out, index=False)

    print("Created recommender-ready files:")
    print(f" - {interactions_out}")
    print(f" - {items_out}")
    print()
    print("Shapes:")
    print(" - interactions:", interactions.shape)
    print(" - items:", items.shape)
    print()
    print("Sample interaction columns:")
    print(interactions.columns.tolist())
    print()
    print("Sample item columns:")
    print(items.columns.tolist())


if __name__ == "__main__":
    main()