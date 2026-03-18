import argparse
import joblib
import pandas as pd

from config import Paths

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True, help="user_id to recommend for")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--interactions", default=Paths().interactions_raw)
    args = ap.parse_args()

    paths = Paths()

    # Load models
    content = joblib.load(f"{paths.models_dir}/content.pkl")
    hybrid_path = f"{paths.models_dir}/hybrid.pkl"

    interactions = pd.read_csv(args.interactions)
    history = interactions[interactions["user_id"].astype(str) == str(args.user)]["item_id"].tolist()

    if history:
        if os.path.exists(hybrid_path):
            hybrid = joblib.load(hybrid_path)
            recs = hybrid.recommend(str(args.user), history, top_n=args.k)
            print(recs)
        else:
            recs = content.recommend_for_user(history, top_n=args.k)
            print(recs)
    else:
        print("No history for this user (cold-start). Provide a few seed items or use popularity baseline.")

if __name__ == "__main__":
    import os
    main()