from dataclasses import dataclass

@dataclass(frozen=True)
class Paths:
    interactions_raw: str = "data/raw/interactions.csv"
    items_raw: str = "data/raw/items.csv"
    processed_dir: str = "data/processed"
    models_dir: str = "models"
    metrics_dir: str = "metrics"
    figures_dir: str = "figures"
    recs_dir: str = "recommendations"

@dataclass(frozen=True)
class TrainConfig:
    k: int = 10
    test_ratio: float = 0.2
    random_state: int = 42

    # MF-SGD (explicit ratings)
    mf_factors: int = 50
    mf_lr: float = 0.01
    mf_reg: float = 0.02
    mf_epochs: int = 25

    # Hybrid blend (0..1): final_score = alpha*cf + (1-alpha)*content
    hybrid_alpha: float = 0.7