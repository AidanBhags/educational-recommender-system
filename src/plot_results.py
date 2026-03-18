import os
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# Real cross-validation results from your project
# --------------------------------------------------
results = {
    "popularity": {
        "precision@10": {"mean": 0.017617582070707072, "std": 0.006758822303938864},
        "recall@10": {"mean": 0.01144488356137112, "std": 0.0031563816215926643},
        "ndcg@10": {"mean": 0.018777144758065512, "std": 0.0058163965044893},
    },
    "content": {
        "precision@10": {"mean": 0.09626736111111112, "std": 0.002129198007933904},
        "recall@10": {"mean": 0.07607654958355649, "std": 0.00039708823369944616},
        "ndcg@10": {"mean": 0.10741373049196906, "std": 0.0022125781933930266},
    },
    "mf": {
        "precision@10": {"mean": 0.01706123737373737, "std": 0.004463388303113792},
        "recall@10": {"mean": 0.008643696676313591, "std": 0.0012954623808453566},
        "ndcg@10": {"mean": 0.02125193411566242, "std": 0.0059258454976052945},
        "rmse": {"mean": 0.30141402880512475, "std": 0.002431329552683104},
    },
    "hybrid": {
        "precision@10": {"mean": 0.09612136994949495, "std": 0.0021011992327163915},
        "recall@10": {"mean": 0.07594424964003053, "std": 0.0004065615835110935},
        "ndcg@10": {"mean": 0.1072794392326734, "std": 0.0021308211326288213},
    },
}

FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)


def plot_metric(metric_name: str, title: str, ylabel: str, filename: str) -> None:
    eligible_models = [m for m in results if metric_name in results[m]]

    means = [results[m][metric_name]["mean"] for m in eligible_models]
    stds = [results[m][metric_name]["std"] for m in eligible_models]

    plt.figure(figsize=(8, 5))
    plt.bar(eligible_models, means, yerr=stds, capsize=4)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("Model")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


# --------------------------------------------------
# Create the graphs
# --------------------------------------------------
plot_metric(
    metric_name="precision@10",
    title="Cross-Validation Precision@10 by Model",
    ylabel="Precision@10",
    filename="cv_precision_at_10.png"
)

plot_metric(
    metric_name="recall@10",
    title="Cross-Validation Recall@10 by Model",
    ylabel="Recall@10",
    filename="cv_recall_at_10.png"
)

plot_metric(
    metric_name="ndcg@10",
    title="Cross-Validation NDCG@10 by Model",
    ylabel="NDCG@10",
    filename="cv_ndcg_at_10.png"
)

plot_metric(
    metric_name="rmse",
    title="Cross-Validation RMSE by Model",
    ylabel="RMSE",
    filename="cv_rmse.png"
)

print("Saved graphs to:", FIGURES_DIR)