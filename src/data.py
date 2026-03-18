import pandas as pd

def load_interactions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"user_id", "item_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"interactions missing columns: {missing}")

    # If implicit, create a strength column; if explicit, keep rating
    if "rating" in df.columns:
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
        df = df.dropna(subset=["rating"])
    else:
        # fallback: if strength exists use it; else assume each row == 1 interaction
        if "strength" not in df.columns:
            df["strength"] = 1.0
        df["strength"] = pd.to_numeric(df["strength"], errors="coerce").fillna(1.0)

    return df

def load_items(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "item_id" not in df.columns:
        raise ValueError("items.csv must contain item_id")

    # Build a single text field for NLP
    title = df["title"] if "title" in df.columns else ""
    desc = df["description"] if "description" in df.columns else ""
    tags = df["tags"] if "tags" in df.columns else ""

    df["text"] = (
        title.astype(str).fillna("") + " " +
        desc.astype(str).fillna("") + " " +
        tags.astype(str).fillna("")
    ).str.strip()

    return df