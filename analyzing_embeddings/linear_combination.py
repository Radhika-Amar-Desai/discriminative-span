import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

sys.path.append(PROJECT_ROOT)

import csv
import argparse
import logging
import importlib.util

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from utils.auto_logger import log_run


# -----------------------------------------------------
# Load config
# -----------------------------------------------------

def load_config(path):

    spec = importlib.util.spec_from_file_location("config", path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    return config


# -----------------------------------------------------
# Load embeddings
# -----------------------------------------------------

def load_embeddings(folder):

    if not os.path.exists(folder):
        raise ValueError(f"Folder does not exist: {folder}")

    embeddings = []
    paths = []

    files = [
        f for f in os.listdir(folder)
        if f.endswith(".npy")
    ]

    for file in files:

        path = os.path.join(folder, file)

        emb = np.load(path)

        embeddings.append(emb)
        paths.append(file)

    embeddings = np.array(embeddings)

    return embeddings, paths

# -----------------------------------------------------
# Effective Rank
# -----------------------------------------------------

def effective_rank(singular_values):

    p = singular_values / np.sum(singular_values)

    entropy = -np.sum(p * np.log(p + 1e-12))

    return int(np.exp(entropy))


# -----------------------------------------------------
# Reduce matrix using SVD
# -----------------------------------------------------

# def reduce_matrix(Vt, S, k):

#     return (S[:k, None] * Vt[:k])


def reduce_matrix(U, S_svd, Vt, k):
    return (U[:, :k] * S_svd[:k]) @ Vt[:k]

# -----------------------------------------------------
# Least squares solver
# -----------------------------------------------------

def solve_least_squares(D, w):

    Dt = D.T

    alpha, *_ = np.linalg.lstsq(Dt, w, rcond=None)

    return alpha


# -----------------------------------------------------
# Ridge solver
# -----------------------------------------------------

def solve_ridge(D, w, lam):

    Dt = D.T

    A = D @ Dt + lam * np.eye(D.shape[0])
    b = D @ w

    alpha = np.linalg.solve(A, b)

    return alpha


# -----------------------------------------------------
# Metric computation
# -----------------------------------------------------

def compute_metrics(D, w, alpha):

    Dt = D.T

    w_proj = Dt @ alpha

    rel_error = np.linalg.norm(w - w_proj) / (np.linalg.norm(w) + 1e-8)

    explained_fraction = 1 - rel_error

    return rel_error, explained_fraction


# -----------------------------------------------------
# Core experiment
# -----------------------------------------------------

def run_span_analysis(

    real_A_folder,
    synthetic_B_folder,
    real_B_folder,
    ridge_lambda,
    logger

):

    logger.info(f"Loading embeddings")

    real_A_emb, real_A_files = load_embeddings(real_A_folder)
    syn_emb, syn_files = load_embeddings(synthetic_B_folder)
    real_B_emb, _ = load_embeddings(real_B_folder)

    logger.info(f"real_A embeddings: {real_A_emb.shape}")
    logger.info(f"synthetic embeddings: {syn_emb.shape}")
    logger.info(f"real_B embeddings: {real_B_emb.shape}")

    if len(real_A_emb) == 0 or len(syn_emb) == 0 or len(real_B_emb) == 0:
        raise ValueError("One of the embedding folders is empty")

    real_A_dict = {f: e for f, e in zip(real_A_files, real_A_emb)}
    syn_dict = {f: e for f, e in zip(syn_files, syn_emb)}

    # -------------------------------------------------
    # Train classifier
    # -------------------------------------------------

    logger.info("Training classifier")

    X = np.vstack([real_A_emb, real_B_emb])
    y = np.hstack([
        np.zeros(len(real_A_emb)),
        np.ones(len(real_B_emb))
    ])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_scaled, y)

    w = clf.coef_.flatten()

    logger.info(f"Classifier weight dimension: {w.shape}")

    # -------------------------------------------------
    # Build difference matrix
    # -------------------------------------------------

    common_files = sorted(
        set(real_A_dict.keys()).intersection(set(syn_dict.keys()))
    )

    if len(common_files) == 0:
        raise ValueError("No matching filenames between real and synthetic embeddings")

    A = np.vstack([real_A_dict[f] for f in common_files])
    B = np.vstack([syn_dict[f] for f in common_files])

    A_scaled = scaler.transform(A)
    B_scaled = scaler.transform(B)

    D = B_scaled - A_scaled

    logger.info(f"Difference matrix shape: {D.shape}")

    # -------------------------------------------------
    # SVD
    # -------------------------------------------------

    U, S_svd, Vt = np.linalg.svd(D, full_matrices=False)

    span_rank = np.linalg.matrix_rank(D)
    eff_rank = effective_rank(S_svd)

    frob_sq = np.sum(S_svd ** 2)
    spectral_sq = S_svd[0] ** 2
    stable_rank = int(round(frob_sq / spectral_sq))

    ranks = {
        "effective_rank": eff_rank,
        "span_rank": span_rank,
        "stable_rank": stable_rank
    }

    solvers = ["least_squares", "ridge"]

    results = []

    # -------------------------------------------------
    # Run pipelines
    # -------------------------------------------------

    for rank_name, k in ranks.items():

        logger.info(f"Running rank method: {rank_name}, k={k}")

        D_reduced = reduce_matrix(U, S_svd, Vt, k)

        for solver in solvers:

            logger.info(f"Solver: {solver}")

            try:
                if solver == "least_squares":
                    alpha = solve_least_squares(D_reduced, w)
                    rel_error, explained_fraction = compute_metrics(D_reduced, w, alpha)

                elif solver == "ridge":
                    alpha = solve_ridge(D_reduced, w, ridge_lambda)
                    rel_error, explained_fraction = compute_metrics(D_reduced, w, alpha)

            except Exception as e:
                logger.error(f"Solver failed: {e}")
                raise

            results.append({
                "rank_method": rank_name,
                "solver": solver,
                "k": int(k),
                "relative_error": float(rel_error),
                "explained_fraction": float(explained_fraction),
                "num_pairs": len(common_files),
                "embedding_dim": D.shape[1]
            })

    return results


# -----------------------------------------------------
# CSV writer
# -----------------------------------------------------

def write_results_csv(results, output_file):

    keys = results[0].keys()

    with open(output_file, "w", newline="") as f:

        writer = csv.DictWriter(f, fieldnames=keys)

        writer.writeheader()
        writer.writerows(results)


# -----------------------------------------------------
# CLI
# -----------------------------------------------------

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)

    args = parser.parse_args()

    config = load_config(args.config)

    run_id, output_dir, log_file = log_run(
        code_filepath=__file__,
        dataset_name=config.DATASET_NAME,
        base_output_dir=config.OUTPUT_DIR
    )

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("span_analysis")

    all_results = []

    for embedding_type, model_configs in config.EMBEDDINGS.items():

        if embedding_type != "raw":
            continue

        logger.info(f"Embedding type: {embedding_type}")

        for model_name, paths in model_configs.items():

            logger.info(f"Model: {model_name}")

            results = run_span_analysis(
                real_A_folder=paths["real_A"],
                synthetic_B_folder=paths["synthetic_B"],
                real_B_folder=paths["real_B"],
                ridge_lambda=config.ridge_lambda,
                logger=logger
            )

            for r in results:
                r["embedding_type"] = embedding_type
                r["model"] = model_name
                all_results.append(r)

    csv_path = os.path.join(output_dir, "span_analysis_results.csv")

    write_results_csv(all_results, csv_path)

    logger.info(f"Results written to {csv_path}")


# -----------------------------------------------------

if __name__ == "__main__":
    main()