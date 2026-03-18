import numpy as np
import pandas as pd

def user_holdout_split(interactions: pd.DataFrame, test_ratio: float, random_state: int):
    rng = np.random.default_rng(random_state)
    users = interactions["user_id"].unique()
    rng.shuffle(users)

    n_test_users = int(len(users) * test_ratio)
    test_users = set(users[:n_test_users])

    test = interactions[interactions["user_id"].isin(test_users)].copy()
    train = interactions[~interactions["user_id"].isin(test_users)].copy()

    return train, test