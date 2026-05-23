import os
import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy.linalg import sqrtm
import numpy as np
import os
import pandas as pd


CONFIG_FOLDER_PATH = r'C:\Users\97433\Knowing_the_difference\configs'

DATASET_FOLDER_NAMES = ['apples_and_oranges', 'horses_and_zebra', 'pneumonia_cxr', 'skin_lesion', 'watermark_dataset']

CONFIG_FILES = {dataset_name: os.path.join(CONFIG_FOLDER_PATH, dataset_name, 'analysis', 'linear_combination_config.py') for dataset_name in DATASET_FOLDER_NAMES}

for config_file in CONFIG_FILES.values():
    print(f"Does {config_file} exist ? {os.path.exists(config_file)}")

ACC_SCORES = {}

F1_SCORES = {'apples_and_oranges': 0.8850, 'horses_and_zebra': 0.9812,
                            'pneumonia_cxr': 0.6392, 'skin_lesion': 0.9767, 'watermark_dataset': 0.9889}

FOUND_MODEL_NAMES = ['clip', 'dinov2', 'resnet18']


import os
import importlib.util
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity

from scipy.linalg import sqrtm


# ---------------------------------------------------------
# CONFIG LOADER
# ---------------------------------------------------------

def load_config(config_path):

    print(config_path)

    spec = importlib.util.spec_from_file_location(
        "config_module",
        config_path
    )

    config_module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(config_module)

    return config_module


# ---------------------------------------------------------
# EMBEDDING LOADER
# ---------------------------------------------------------

def load_embeddings(folder_path):

    embeddings = []

    for fname in sorted(os.listdir(folder_path)):

        if not fname.endswith(".npy"):
            continue

        fpath = os.path.join(folder_path, fname)

        emb = np.load(fpath).flatten()

        embeddings.append(emb)

    embeddings = np.stack(embeddings, axis=0)

    return embeddings


# ---------------------------------------------------------
# FID
# ---------------------------------------------------------

def compute_fid(real_embeddings, synthetic_embeddings):

    mu1 = np.mean(real_embeddings, axis=0)
    mu2 = np.mean(synthetic_embeddings, axis=0)

    sigma1 = np.cov(real_embeddings, rowvar=False)
    sigma2 = np.cov(synthetic_embeddings, rowvar=False)

    diff = mu1 - mu2

    covmean = sqrtm(sigma1 @ sigma2)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = (
        diff @ diff
        + np.trace(sigma1)
        + np.trace(sigma2)
        - 2 * np.trace(covmean)
    )

    return float(fid)


# ---------------------------------------------------------
# LINEAR PROBE
# ---------------------------------------------------------

def compute_linear_probe_score(
    real_A_embeddings,
    real_B_embeddings
):

    """
    Linear separability baseline.

    Train linear classifier on:
        class A embeddings
        class B embeddings
    """

    X = np.concatenate(
        [
            real_A_embeddings,
            real_B_embeddings
        ],
        axis=0
    )

    y = np.concatenate(
        [
            np.zeros(len(real_A_embeddings)),
            np.ones(len(real_B_embeddings))
        ]
    )

    clf = LogisticRegression(
        max_iter=5000
    )

    clf.fit(X, y)

    preds = clf.predict(X)

    acc = accuracy_score(y, preds)

    return float(acc)


# ---------------------------------------------------------
# EMBEDDING SIMILARITY
# ---------------------------------------------------------

def compute_embedding_similarity(
    real_B_embeddings,
    synthetic_B_embeddings
):

    """
    Measures similarity between:
        real positives
        synthetic positives
    """

    real_mean = np.mean(real_B_embeddings, axis=0)

    synthetic_mean = np.mean(
        synthetic_B_embeddings,
        axis=0
    )

    similarity = cosine_similarity(
        real_mean.reshape(1, -1),
        synthetic_mean.reshape(1, -1)
    )[0][0]

    return float(similarity)


# ---------------------------------------------------------
# Main Metric Loader
# ---------------------------------------------------------

def get_scores(
    found_model_name,
    dataset_name,
    config_file=None,
    scaling_key="raw"
):

    config = load_config(config_file)

    # -----------------------------------------------------
    # ACCESS CONFIG
    # -----------------------------------------------------

    embedding_config = config.EMBEDDINGS[
        scaling_key
    ][
        found_model_name
    ]

    # -----------------------------------------------------
    # PATHS
    # -----------------------------------------------------

    real_A_dir = embedding_config["real_A"]
    real_B_dir = embedding_config["real_B"]
    synthetic_B_dir = embedding_config["synthetic_B"]

    # -----------------------------------------------------
    # LOAD EMBEDDINGS
    # -----------------------------------------------------

    real_A_embeddings = load_embeddings(
        real_A_dir
    )

    real_B_embeddings = load_embeddings(
        real_B_dir
    )

    synthetic_B_embeddings = load_embeddings(
        synthetic_B_dir
    )

    # -----------------------------------------------------
    # FID
    # -----------------------------------------------------

    fid_score = compute_fid(
        real_B_embeddings,
        synthetic_B_embeddings
    )

    # -----------------------------------------------------
    # LINEAR PROBE
    # -----------------------------------------------------

    linear_probe_score = compute_linear_probe_score(
        real_A_embeddings,
        synthetic_B_embeddings
    )

    # -----------------------------------------------------
    # EMBEDDING SIMILARITY
    # -----------------------------------------------------

    similarity_score = compute_embedding_similarity(
        real_B_embeddings,
        synthetic_B_embeddings
    )

    return (
        fid_score,
        linear_probe_score,
        similarity_score
    )

# ---------------------------------------------------------
# Correlation Utility
# ---------------------------------------------------------

def compute_correlations(metric_scores, target_scores):

    metric_values = []
    target_values = []

    for dataset_name in metric_scores.keys():

        if dataset_name not in target_scores:
            continue

        metric_values.append(metric_scores[dataset_name])
        target_values.append(target_scores[dataset_name])

    metric_values = np.array(metric_values)
    target_values = np.array(target_values)

    pearson_r, pearson_p = pearsonr(metric_values, target_values)
    spearman_rho, spearman_p = spearmanr(metric_values, target_values)

    return {
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_rho": spearman_rho,
        "spearman_p": spearman_p
    }


# ---------------------------------------------------------
# GLOBAL COLORS
# ---------------------------------------------------------

FOUNDATION_COLORS = {
    "clip": "blue",
    "dinov2": "green",
    "resnet18": "red"
}


# ---------------------------------------------------------
# COMBINED PLOTTING
# ---------------------------------------------------------

def correlation_plot(
    all_metric_scores,
    target_scores,
    metric_name,
    save_dir="plots"
):

    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(8, 6))

    rows = []

    # -----------------------------------------------------
    # PLOT EACH FOUNDATION MODEL
    # -----------------------------------------------------

    for found_model_name in FOUND_MODEL_NAMES:

        metric_scores = all_metric_scores[
            found_model_name
        ]

        x = []
        y = []
        labels = []

        for dataset_name in metric_scores.keys():

            if dataset_name not in target_scores:
                continue

            x.append(
                metric_scores[dataset_name]
            )

            y.append(
                target_scores[dataset_name]
            )

            labels.append(dataset_name)

        x = np.array(x)
        y = np.array(y)

        # -------------------------------------------------
        # CORRELATIONS
        # -------------------------------------------------

        correlations = compute_correlations(
            metric_scores,
            target_scores
        )

        pearson_r = correlations["pearson_r"]

        rows.append({
            "Foundation Model": found_model_name,
            "Metric": metric_name,
            "Pearson r": round(
                pearson_r,
                4
            ),
            "Spearman rho": round(
                correlations["spearman_rho"],
                4
            )
        })

        # -------------------------------------------------
        # SORT FOR CLEAN LINES
        # -------------------------------------------------

        sorted_pairs = sorted(
            zip(x, y, labels),
            key=lambda z: z[0]
        )

        x_sorted = [p[0] for p in sorted_pairs]
        y_sorted = [p[1] for p in sorted_pairs]
        labels_sorted = [p[2] for p in sorted_pairs]

        # -------------------------------------------------
        # PLOT
        # -------------------------------------------------

        plt.plot(
            x_sorted,
            y_sorted,
            marker='o',
            linestyle='-',
            linewidth=2,
            markersize=8,
            color=FOUNDATION_COLORS[
                found_model_name
            ],
            label=(
                f"{found_model_name.upper()} "
                f"(r={pearson_r:.2f})"
            )
        )

        # -------------------------------------------------
        # LABELS
        # -------------------------------------------------

        for xi, yi, label in zip(
            x_sorted,
            y_sorted,
            labels_sorted
        ):

            plt.annotate(
                label,
                (xi, yi),
                fontsize=8,
                alpha=0.8
            )

    # -----------------------------------------------------
    # FID SPECIAL CASE
    # -----------------------------------------------------

    if metric_name == "FID":

        # Lower FID is better
        plt.gca().invert_xaxis()

    # -----------------------------------------------------
    # STYLING
    # -----------------------------------------------------

    plt.xlabel(
        metric_name,
        fontsize=12
    )

    plt.ylabel(
        "Best Test F1 Score",
        fontsize=12
    )

    plt.title(
        f"{metric_name} Correlation Analysis",
        fontsize=14
    )

    plt.grid(alpha=0.3)

    plt.legend()

    plt.tight_layout()

    # -----------------------------------------------------
    # SAVE
    # -----------------------------------------------------

    save_path = os.path.join(
        save_dir,
        f"{metric_name.lower().replace(' ', '_')}_combined.png"
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches='tight'
    )

    plt.close()

    print(f"Saved plot: {save_path}")

    # -----------------------------------------------------
    # PRINT TABLE
    # -----------------------------------------------------

    df = pd.DataFrame(rows)

    print("\n" + "=" * 70)
    print(f"{metric_name.upper()} CORRELATIONS")
    print("=" * 70)

    print(df.to_string(index=False))


# ---------------------------------------------------------
# MAIN BASELINE COMPARISON
# ---------------------------------------------------------

def compare_baselines(
    config_files,
    acc_scores,
    f1_scores
):

    # -----------------------------------------------------
    # STORE ALL SCORES
    # -----------------------------------------------------

    all_fid_scores = dict()
    all_linear_probe_scores = dict()
    all_similarity_scores = dict()

    # -----------------------------------------------------
    # COMPUTE SCORES
    # -----------------------------------------------------

    for found_model_name in FOUND_MODEL_NAMES:

        fid_scores = dict()

        linear_probe_scores = dict()

        similarity_scores = dict()

        for dataset_name in f1_scores.keys():

            config_file = config_files[
                dataset_name
            ]

            (
                fid_score,
                linear_probe_score,
                similarity_score

            ) = get_scores(
                found_model_name=found_model_name,
                dataset_name=dataset_name,
                config_file=config_file
            )

            fid_scores[
                dataset_name
            ] = fid_score

            linear_probe_scores[
                dataset_name
            ] = linear_probe_score

            similarity_scores[
                dataset_name
            ] = similarity_score

        # -------------------------------------------------
        # STORE
        # -------------------------------------------------

        all_fid_scores[
            found_model_name
        ] = fid_scores

        all_linear_probe_scores[
            found_model_name
        ] = linear_probe_scores

        all_similarity_scores[
            found_model_name
        ] = similarity_scores

    # -----------------------------------------------------
    # PLOT ALL TOGETHER
    # -----------------------------------------------------

    correlation_plot(
        all_metric_scores=all_fid_scores,
        target_scores=f1_scores,
        metric_name="FID"
    )

    correlation_plot(
        all_metric_scores=all_linear_probe_scores,
        target_scores=f1_scores,
        metric_name="Linear Probe Accuracy"
    )

    correlation_plot(
        all_metric_scores=all_similarity_scores,
        target_scores=f1_scores,
        metric_name="Embedding Similarity"
    )


# ---------------------------------------------------------
# Entry
# ---------------------------------------------------------

if __name__ == "__main__":

    compare_baselines(
        config_files=CONFIG_FILES,
        acc_scores=ACC_SCORES,
        f1_scores=F1_SCORES
    )