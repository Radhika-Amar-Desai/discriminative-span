import os
import time
import importlib.util
import numpy as np
import pandas as pd

from scipy.stats import pearsonr, spearmanr
from scipy.linalg import sqrtm

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib.pyplot as plt


# =========================================================
# CONFIGURATION
# =========================================================

CONFIG_FOLDER_PATH = r'C:\Users\97433\Knowing_the_difference\configs'

DATASET_FOLDER_NAMES = [
    'apples_and_oranges',
    'horses_and_zebra',
    'pneumonia_cxr',
    'skin_lesion',
    'watermark_dataset'
]

CONFIG_FILES = {
    dataset_name: os.path.join(
        CONFIG_FOLDER_PATH,
        dataset_name,
        'analysis',
        'linear_combination_config.py'
    )
    for dataset_name in DATASET_FOLDER_NAMES
}

# ---------------------------------------------------------
# DOWNSTREAM PERFORMANCE
# ---------------------------------------------------------

F1_SCORES = {
    'apples_and_oranges': 0.8850,
    'horses_and_zebra': 0.9812,
    'pneumonia_cxr': 0.6392,
    'skin_lesion': 0.9767,
    'watermark_dataset': 0.9889
}

FOUND_MODEL_NAMES = [
    'clip',
    'dinov2',
    'resnet18'
]

# =========================================================
# COLORS
# =========================================================

FOUNDATION_COLORS = {
    "clip": "blue",
    "dinov2": "green",
    "resnet18": "red"
}

# =========================================================
# OUTPUT DIRECTORY
# =========================================================

OUTPUT_DIR = "baseline_plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================
# CONFIG LOADER
# =========================================================

def load_config(config_path):

    print(f"\n[CONFIG] Loading config:")
    print(config_path)

    start = time.time()

    spec = importlib.util.spec_from_file_location(
        "config_module",
        config_path
    )

    config_module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(config_module)

    end = time.time()

    print(f"[CONFIG] Loaded in {end - start:.2f} sec")

    return config_module


# =========================================================
# EMBEDDING LOADER
# =========================================================

def load_embeddings(folder_path):

    print(f"\n[EMBEDDINGS] Loading embeddings from:")
    print(folder_path)

    start = time.time()

    embeddings = []

    npy_files = sorted([
        fname for fname in os.listdir(folder_path)
        if fname.endswith(".npy")
    ])

    print(f"[EMBEDDINGS] Found {len(npy_files)} files")

    for idx, fname in enumerate(npy_files):

        if idx % 50 == 0:
            print(
                f"[EMBEDDINGS] Loaded "
                f"{idx}/{len(npy_files)}"
            )

        fpath = os.path.join(folder_path, fname)

        emb = np.load(fpath).flatten()

        embeddings.append(emb)

    embeddings = np.stack(embeddings, axis=0)

    end = time.time()

    print(
        f"[EMBEDDINGS] Shape = {embeddings.shape}"
    )

    print(
        f"[EMBEDDINGS] Finished in "
        f"{end - start:.2f} sec"
    )

    return embeddings


# =========================================================
# FID
# =========================================================

def compute_fid(real_embeddings, synthetic_embeddings):

    print("\n[FID] Computing FID...")

    start = time.time()

    mu1 = np.mean(real_embeddings, axis=0)
    mu2 = np.mean(synthetic_embeddings, axis=0)

    print("[FID] Mean vectors computed")

    sigma1 = np.cov(real_embeddings, rowvar=False)
    sigma2 = np.cov(synthetic_embeddings, rowvar=False)

    print("[FID] Covariance matrices computed")

    diff = mu1 - mu2

    print("[FID] Computing sqrtm...")

    sqrtm_start = time.time()

    covmean = sqrtm(sigma1 @ sigma2)

    sqrtm_end = time.time()

    print(
        f"[FID] sqrtm completed in "
        f"{sqrtm_end - sqrtm_start:.2f} sec"
    )

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = (
        diff @ diff
        + np.trace(sigma1)
        + np.trace(sigma2)
        - 2 * np.trace(covmean)
    )

    end = time.time()

    print(f"[FID] Final score = {fid:.4f}")

    print(
        f"[FID] Completed in "
        f"{end - start:.2f} sec"
    )

    return float(fid)


# =========================================================
# LINEAR PROBE
# =========================================================

def compute_linear_probe_score(
    real_A_embeddings,
    synthetic_B_embeddings
):

    print("\n[LINEAR PROBE] Training classifier...")

    start = time.time()

    X = np.concatenate(
        [
            real_A_embeddings,
            synthetic_B_embeddings
        ],
        axis=0
    )

    y = np.concatenate(
        [
            np.zeros(len(real_A_embeddings)),
            np.ones(len(synthetic_B_embeddings))
        ]
    )

    print(f"[LINEAR PROBE] X shape = {X.shape}")

    clf = LogisticRegression(
        max_iter=5000
    )

    fit_start = time.time()

    clf.fit(X, y)

    fit_end = time.time()

    print(
        f"[LINEAR PROBE] Training completed in "
        f"{fit_end - fit_start:.2f} sec"
    )

    preds = clf.predict(X)

    acc = accuracy_score(y, preds)

    end = time.time()

    print(f"[LINEAR PROBE] Accuracy = {acc:.4f}")

    print(
        f"[LINEAR PROBE] Total time = "
        f"{end - start:.2f} sec"
    )

    return float(acc)


# =========================================================
# EMBEDDING SIMILARITY
# =========================================================

def compute_embedding_similarity(
    real_B_embeddings,
    synthetic_B_embeddings
):

    print("\n[SIMILARITY] Computing cosine similarity...")

    start = time.time()

    real_mean = np.mean(real_B_embeddings, axis=0)

    synthetic_mean = np.mean(
        synthetic_B_embeddings,
        axis=0
    )

    similarity = cosine_similarity(
        real_mean.reshape(1, -1),
        synthetic_mean.reshape(1, -1)
    )[0][0]

    end = time.time()

    print(
        f"[SIMILARITY] Score = {similarity:.4f}"
    )

    print(
        f"[SIMILARITY] Completed in "
        f"{end - start:.2f} sec"
    )

    return float(similarity)


# =========================================================
# SCORE EXTRACTION
# =========================================================

def get_scores(
    found_model_name,
    dataset_name,
    config_file=None,
    scaling_key="raw"
):

    print("\n" + "=" * 80)
    print(f"[START] {found_model_name.upper()} | {dataset_name}")
    print("=" * 80)

    total_start = time.time()

    config = load_config(config_file)

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

    total_end = time.time()

    print("\n[SUMMARY]")
    print(f"FID                 : {fid_score:.4f}")
    print(f"Linear Probe        : {linear_probe_score:.4f}")
    print(f"Embedding Similarity: {similarity_score:.4f}")

    print(
        f"\n[TOTAL TIME] "
        f"{found_model_name.upper()} | {dataset_name}"
    )

    print(
        f"{total_end - total_start:.2f} sec"
    )

    return (
        fid_score,
        linear_probe_score,
        similarity_score
    )


# =========================================================
# CORRELATION UTILITY
# =========================================================

def compute_correlations(metric_scores, target_scores):

    metric_values = []
    target_values = []

    for dataset_name in metric_scores.keys():

        if dataset_name not in target_scores:
            continue

        metric_values.append(
            metric_scores[dataset_name]
        )

        target_values.append(
            target_scores[dataset_name]
        )

    metric_values = np.array(metric_values)
    target_values = np.array(target_values)

    pearson_r, pearson_p = pearsonr(
        metric_values,
        target_values
    )

    spearman_rho, spearman_p = spearmanr(
        metric_values,
        target_values
    )

    return {
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_rho": spearman_rho,
        "spearman_p": spearman_p
    }


# =========================================================
# SAVE FULL METRIC VALUES
# =========================================================

def save_metric_values_csv(
    all_metric_scores,
    target_scores,
    metric_name,
    save_dir=OUTPUT_DIR
):

    print("\n[SAVING FULL METRIC VALUES CSV]")

    rows = []

    for found_model_name in FOUND_MODEL_NAMES:

        metric_scores = all_metric_scores[
            found_model_name
        ]

        for dataset_name in metric_scores.keys():

            row = {
                "Foundation Model": found_model_name,
                "Dataset": dataset_name,
                "Metric": metric_name,
                "Metric Value": metric_scores[
                    dataset_name
                ],
                "Best Test F1": target_scores[
                    dataset_name
                ]
            }

            rows.append(row)

    df = pd.DataFrame(rows)

    csv_path = os.path.join(
        save_dir,
        f"{metric_name.lower().replace(' ', '_')}_all_values.csv"
    )

    df.to_csv(
        csv_path,
        index=False
    )

    print(f"[SAVED] {csv_path}")


# =========================================================
# PLOTTING
# =========================================================

def correlation_plot(
    all_metric_scores,
    target_scores,
    metric_name,
    save_dir=OUTPUT_DIR
):

    print("\n" + "#" * 80)
    print(f"[PLOTTING] {metric_name}")
    print("#" * 80)

    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(8, 6))

    rows = []

    # -----------------------------------------------------
    # FOUNDATION MODEL LOOP
    # -----------------------------------------------------

    for found_model_name in FOUND_MODEL_NAMES:

        print(
            f"\n[PLOT] Processing "
            f"{found_model_name.upper()}"
        )

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

        correlations = compute_correlations(
            metric_scores,
            target_scores
        )

        pearson_r = correlations["pearson_r"]

        spearman_rho = correlations[
            "spearman_rho"
        ]

        # -------------------------------------------------
        # STORE CORRELATION TABLE
        # -------------------------------------------------

        rows.append({
            "Foundation Model": found_model_name,
            "Metric": metric_name,
            "Pearson r": round(
                correlations["pearson_r"],
                4
            ),
            "Pearson p": round(
                correlations["pearson_p"],
                4
            ),
            "Spearman rho": round(
                correlations["spearman_rho"],
                4
            ),
            "Spearman p": round(
                correlations["spearman_p"],
                4
            )
        })

        # -------------------------------------------------
        # SORT
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
            linewidth=2.5,
            markersize=8,
            alpha=0.9,
            color=FOUNDATION_COLORS[
                found_model_name
            ],
            label=(
                f"{found_model_name.upper()} "
                f"(r={pearson_r:.2f}, "
                f"ρ={spearman_rho:.2f})"
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
                alpha=0.85
            )

    # -----------------------------------------------------
    # FID SPECIAL CASE
    # -----------------------------------------------------

    if metric_name == "FID":

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

    plt.grid(
        alpha=0.25,
        linestyle='--'
    )

    plt.legend(
        fontsize=10
    )

    plt.tight_layout()

    # -----------------------------------------------------
    # SAVE PLOT
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

    print(f"\n[SAVED] Plot -> {save_path}")

    # -----------------------------------------------------
    # SAVE CORRELATION CSV
    # -----------------------------------------------------

    df = pd.DataFrame(rows)

    csv_path = os.path.join(
        save_dir,
        f"{metric_name.lower().replace(' ', '_')}_correlations.csv"
    )

    df.to_csv(
        csv_path,
        index=False
    )

    print(f"[SAVED] Correlation CSV -> {csv_path}")

    # -----------------------------------------------------
    # SAVE FULL VALUES CSV
    # -----------------------------------------------------

    save_metric_values_csv(
        all_metric_scores=all_metric_scores,
        target_scores=target_scores,
        metric_name=metric_name,
        save_dir=save_dir
    )

    # -----------------------------------------------------
    # PRINT TABLE
    # -----------------------------------------------------

    print("\n" + "=" * 80)
    print(f"{metric_name.upper()} CORRELATIONS")
    print("=" * 80)

    print(df.to_string(index=False))


# =========================================================
# MAIN BASELINE COMPARISON
# =========================================================

def compare_baselines(
    config_files,
    f1_scores
):

    overall_start = time.time()

    all_fid_scores = dict()

    all_linear_probe_scores = dict()

    all_similarity_scores = dict()

    # -----------------------------------------------------
    # COMPUTE SCORES
    # -----------------------------------------------------

    for found_model_name in FOUND_MODEL_NAMES:

        print("\n" + "#" * 80)
        print(
            f"[FOUNDATION MODEL] "
            f"{found_model_name.upper()}"
        )
        print("#" * 80)

        foundation_start = time.time()

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

            # -------------------------------------------------
            # STORE
            # -------------------------------------------------

            fid_scores[
                dataset_name
            ] = fid_score

            linear_probe_scores[
                dataset_name
            ] = linear_probe_score

            similarity_scores[
                dataset_name
            ] = similarity_score

        # -----------------------------------------------------
        # STORE FOUNDATION RESULTS
        # -----------------------------------------------------

        all_fid_scores[
            found_model_name
        ] = fid_scores

        all_linear_probe_scores[
            found_model_name
        ] = linear_probe_scores

        all_similarity_scores[
            found_model_name
        ] = similarity_scores

        foundation_end = time.time()

        print(
            f"\n[FOUNDATION COMPLETE] "
            f"{found_model_name.upper()}"
        )

        print(
            f"Time taken: "
            f"{foundation_end - foundation_start:.2f} sec"
        )

    # =====================================================
    # PLOTS
    # =====================================================

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

    overall_end = time.time()

    print("\n" + "#" * 80)
    print("[ALL COMPLETE]")
    print("#" * 80)

    print(
        f"Total execution time: "
        f"{overall_end - overall_start:.2f} sec"
    )


# =========================================================
# ENTRY
# =========================================================

if __name__ == "__main__":

    compare_baselines(
        config_files=CONFIG_FILES,
        f1_scores=F1_SCORES
    )