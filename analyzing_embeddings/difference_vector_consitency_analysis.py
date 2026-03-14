# import os
# import sys

# CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

# sys.path.append(PROJECT_ROOT)

# import argparse
# import logging
# import numpy as np

# from sklearn.decomposition import PCA
# from sklearn.metrics.pairwise import cosine_similarity

# from utils.auto_logger import BASE_DIR, log_run


# # --------------------------------------------------
# # Logger
# # --------------------------------------------------

# def setup_logger(log_file):

#     logger = logging.getLogger("difference_vector_logger")
#     logger.setLevel(logging.INFO)

#     if logger.hasHandlers():
#         logger.handlers.clear()

#     formatter = logging.Formatter("%(asctime)s - %(message)s")

#     file_handler = logging.FileHandler(log_file, encoding="utf-8")
#     file_handler.setFormatter(formatter)

#     console_handler = logging.StreamHandler(sys.stdout)
#     console_handler.setFormatter(formatter)

#     logger.addHandler(file_handler)
#     logger.addHandler(console_handler)

#     return logger


# # --------------------------------------------------
# # Load embeddings
# # --------------------------------------------------

# def load_embeddings(class_dir, logger):

#     logger.info(f"Loading embeddings from: {class_dir}")

#     files = sorted([
#         f for f in os.listdir(class_dir)
#         if f.endswith(".npy") and os.path.isfile(os.path.join(class_dir, f))
#     ])

#     logger.info(f"Found {len(files)} embedding files")

#     vectors = []

#     for i, f in enumerate(files):

#         path = os.path.join(class_dir, f)

#         if i % 100 == 0:
#             logger.info(f"Loading file {i}/{len(files)} : {f}")

#         vec = np.load(path)

#         vectors.append(vec)

#     logger.info("Stacking embeddings into matrix")

#     V = np.stack(vectors)

#     logger.info(f"Finished loading embeddings with shape {V.shape}")

#     return V


# # --------------------------------------------------
# # Compute difference vectors between classes
# # --------------------------------------------------

# def compute_class_difference_vectors(model_root, logger):

#     logger.info(f"Computing class difference vectors for {model_root}")

#     classes = sorted([
#         d for d in os.listdir(model_root)
#         if os.path.isdir(os.path.join(model_root, d))
#     ])

#     logger.info(f"Classes detected: {classes}")

#     if len(classes) < 2:
#         raise ValueError("Need at least two class folders")

#     classA = os.path.join(model_root, classes[0])
#     classB = os.path.join(model_root, classes[1])

#     logger.info(f"Loading class A embeddings: {classes[0]}")
#     A = load_embeddings(classA, logger)

#     logger.info(f"Loading class B embeddings: {classes[1]}")
#     B = load_embeddings(classB, logger)

#     logger.info(f"A shape: {A.shape}")
#     logger.info(f"B shape: {B.shape}")

#     n = min(len(A), len(B))

#     logger.info(f"Using {n} paired samples")

#     A = A[:n]
#     B = B[:n]

#     logger.info("Computing difference vectors")

#     V = B - A

#     logger.info(f"Difference vector matrix shape: {V.shape}")

#     return V, classes


# # --------------------------------------------------
# # Metrics
# # --------------------------------------------------

# def analyze(V, logger):

#     logger.info("Starting analysis...")

#     n, d = V.shape

#     logger.info(f"\nSamples: {n}")
#     logger.info(f"Feature dim: {d}")

#     norms = np.linalg.norm(V, axis=1)

#     logger.info("\nNorm statistics")
#     logger.info(f"mean: {norms.mean():.4f}")
#     logger.info(f"std: {norms.std():.4f}")

#     logger.info("Computing cosine similarity matrix...")

#     cos = cosine_similarity(V)
#     upper = cos[np.triu_indices_from(cos, k=1)]

#     logger.info("\nCosine similarity")
#     logger.info(f"mean: {upper.mean():.4f}")
#     logger.info(f"std: {upper.std():.4f}")
#     logger.info(f"min: {upper.min():.4f}")
#     logger.info(f"max: {upper.max():.4f}")

#     mean_vec = np.mean(V, axis=0)
#     mean_vec /= np.linalg.norm(mean_vec)

#     alignment = []

#     for v in V:
#         v = v / np.linalg.norm(v)
#         alignment.append(np.dot(v, mean_vec))

#     logger.info("\nAlignment with mean Δ")
#     logger.info(f"mean: {np.mean(alignment):.4f}")
#     logger.info(f"std: {np.std(alignment):.4f}")

#     rank = np.linalg.matrix_rank(V)
#     logger.info(f"\nMatrix rank: {rank}")

#     logger.info("Running PCA...")

#     pca = PCA()
#     pca.fit(V)

#     explained = pca.explained_variance_ratio_

#     logger.info("\nPCA spectrum (first 10 PCs)")

#     for i in range(min(10, len(explained))):
#         logger.info(f"PC{i+1}: {explained[i]:.4f}")

#     logger.info(f"\nVariance first 3 PCs: {explained[:3].sum():.4f}")
#     logger.info(f"Variance first 5 PCs: {explained[:5].sum():.4f}")
#     logger.info(f"Variance first 10 PCs: {explained[:10].sum():.4f}")


# # --------------------------------------------------
# # Model analysis
# # --------------------------------------------------

# def run_model_analysis(model_root, model_name, logger):

#     logger.info("\n================================")
#     logger.info(f"MODEL: {model_name}")
#     logger.info("================================")

#     V, classes = compute_class_difference_vectors(model_root, logger)

#     logger.info(f"Class difference: {classes[1]} - {classes[0]}")

#     analyze(V, logger)


# # --------------------------------------------------
# # Dataset structure validation
# # --------------------------------------------------

# def validate_dataset_structure(dataset_root, logger):

#     if not os.path.exists(dataset_root):
#         raise ValueError(f"Dataset root does not exist: {dataset_root}")

#     logger.info("\nValidating dataset structure...")

#     models = ["dinov2", "resnet18", "clip"]

#     for model_name in models:

#         model_root = os.path.join(dataset_root, model_name)

#         if not os.path.exists(model_root):
#             logger.warning(f"{model_name} folder missing (will skip)")
#             continue

#         classes = [
#             d for d in os.listdir(model_root)
#             if os.path.isdir(os.path.join(model_root, d))
#         ]

#         if len(classes) < 2:
#             raise ValueError(
#                 f"{model_name} must contain at least two class folders"
#             )

#         logger.info(f"{model_name}: classes detected -> {classes}")

#         for cls in classes:

#             cls_dir = os.path.join(model_root, cls)

#             files = [
#                 f for f in os.listdir(cls_dir)
#                 if f.endswith(".npy")
#                 and os.path.isfile(os.path.join(cls_dir, f))
#             ]

#             if len(files) == 0:
#                 raise ValueError(
#                     f"No .npy embeddings found in {cls_dir}"
#                 )

#             sample = np.load(os.path.join(cls_dir, files[0]))
#             dim = sample.shape[0]

#             for f in files[1:]:

#                 emb = np.load(os.path.join(cls_dir, f))

#                 if emb.shape[0] != dim:
#                     raise ValueError(
#                         f"Inconsistent embedding dimensions in {cls_dir}"
#                     )

#         logger.info(f"{model_name}: structure OK")

#     logger.info("Dataset validation completed successfully\n")


# # --------------------------------------------------
# # CLI
# # --------------------------------------------------

# def main():

#     parser = argparse.ArgumentParser()

#     parser.add_argument(
#         "--dataset_root",
#         type=str,
#         required=True
#     )

#     parser.add_argument(
#         "--dataset_name",
#         type=str,
#         default="embedding_difference_dataset"
#     )

#     args = parser.parse_args()

#     run_id, output_dir, log_file = log_run(
#         code_filepath=__file__,
#         dataset_name=args.dataset_name,
#         base_output_dir=os.path.join(BASE_DIR, "output")
#     )

#     logger = setup_logger(log_file)

#     logger.info("================================")
#     logger.info("Difference Vector Analysis")
#     logger.info(f"Run ID: {run_id}")
#     logger.info(f"Dataset Root: {args.dataset_root}")
#     logger.info("================================")

#     validate_dataset_structure(args.dataset_root, logger)

#     models = ["dinov2", "resnet18", "clip"]

#     for model_name in models:

#         model_root = os.path.join(args.dataset_root, model_name)

#         if not os.path.exists(model_root):

#             logger.info(f"\nSkipping {model_name} (not found)")
#             continue

#         run_model_analysis(model_root, model_name, logger)


# if __name__ == "__main__":
#     main()

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

sys.path.append(PROJECT_ROOT)

import argparse
import logging
import numpy as np

from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

from utils.auto_logger import BASE_DIR, log_run


# --------------------------------------------------
# Logger
# --------------------------------------------------

def setup_logger(log_file):

    logger = logging.getLogger("difference_vector_logger")
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(message)s")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# --------------------------------------------------
# Load embeddings
# --------------------------------------------------

def load_embeddings(class_dir, logger):

    logger.info(f"Loading embeddings from: {class_dir}")

    files = sorted([
        f for f in os.listdir(class_dir)
        if f.endswith(".npy") and os.path.isfile(os.path.join(class_dir, f))
    ])

    logger.info(f"Found {len(files)} embedding files")

    vectors = []

    for i, f in enumerate(files):

        path = os.path.join(class_dir, f)

        if i % 100 == 0:
            logger.info(f"Loading file {i}/{len(files)} : {f}")

        vec = np.load(path)

        vectors.append(vec)

    logger.info("Stacking embeddings into matrix")

    V = np.stack(vectors)

    logger.info(f"Finished loading embeddings with shape {V.shape}")

    return V


# --------------------------------------------------
# Compute difference vectors between classes
# --------------------------------------------------

def compute_class_difference_vectors(model_root, logger):

    logger.info(f"Computing class difference vectors for {model_root}")

    classes = sorted([
        d for d in os.listdir(model_root)
        if os.path.isdir(os.path.join(model_root, d))
    ])

    logger.info(f"Classes detected: {classes}")

    if len(classes) < 2:
        raise ValueError("Need at least two class folders")

    classA = os.path.join(model_root, classes[0])
    classB = os.path.join(model_root, classes[1])

    logger.info(f"Loading class A embeddings: {classes[0]}")
    A = load_embeddings(classA, logger)

    logger.info(f"Loading class B embeddings: {classes[1]}")
    B = load_embeddings(classB, logger)

    logger.info(f"A shape: {A.shape}")
    logger.info(f"B shape: {B.shape}")

    # --------------------------------------------------
    # Embedding dimensionality analysis
    # --------------------------------------------------

    X = np.vstack([A, B])

    total_dims = X.shape[1]

    nonzero_mask = np.any(X != 0, axis=0)

    active_dims = np.sum(nonzero_mask)

    zero_dims = total_dims - active_dims

    logger.info("\nEmbedding dimensional analysis")
    logger.info(f"Total embedding dimensions : {total_dims}")
    logger.info(f"Active (non-zero) dims     : {active_dims}")
    logger.info(f"Completely zero dims       : {zero_dims}")

    # --------------------------------------------------

    n = min(len(A), len(B))

    logger.info(f"Using {n} paired samples")

    A = A[:n]
    B = B[:n]

    logger.info("Computing difference vectors")

    V = B - A

    logger.info(f"Difference vector matrix shape: {V.shape}")

    return V, classes

# --------------------------------------------------
# Metrics
# --------------------------------------------------

def analyze(V, logger):

    logger.info("Starting analysis...")

    n, d = V.shape

    logger.info(f"\nSamples: {n}")
    logger.info(f"Feature dim: {d}")

    norms = np.linalg.norm(V, axis=1)

    logger.info("\nNorm statistics")
    logger.info(f"mean: {norms.mean():.4f}")
    logger.info(f"std: {norms.std():.4f}")

    logger.info("Computing cosine similarity matrix...")

    cos = cosine_similarity(V)
    upper = cos[np.triu_indices_from(cos, k=1)]

    logger.info("\nCosine similarity")
    logger.info(f"mean: {upper.mean():.4f}")
    logger.info(f"std: {upper.std():.4f}")
    logger.info(f"min: {upper.min():.4f}")
    logger.info(f"max: {upper.max():.4f}")

    mean_vec = np.mean(V, axis=0)
    mean_vec /= np.linalg.norm(mean_vec)

    alignment = []

    for v in V:
        v = v / np.linalg.norm(v)
        alignment.append(np.dot(v, mean_vec))

    logger.info("\nAlignment with mean delta")
    logger.info(f"mean: {np.mean(alignment):.4f}")
    logger.info(f"std: {np.std(alignment):.4f}")

    rank = np.linalg.matrix_rank(V)
    logger.info(f"\nMatrix rank: {rank}")

    logger.info("Running PCA...")

    pca = PCA()
    pca.fit(V)

    explained = pca.explained_variance_ratio_

    logger.info("\nPCA spectrum (first 10 PCs)")

    for i in range(min(10, len(explained))):
        logger.info(f"PC{i+1}: {explained[i]:.4f}")

    logger.info(f"\nVariance first 3 PCs: {explained[:3].sum():.4f}")
    logger.info(f"Variance first 5 PCs: {explained[:5].sum():.4f}")
    logger.info(f"Variance first 10 PCs: {explained[:10].sum():.4f}")

    logger.info("\nResidual transformation analysis")

    mean_vec = np.mean(V, axis=0)

    V_res = V - mean_vec

    cos_res = cosine_similarity(V_res)
    upper_res = cos_res[np.triu_indices_from(cos_res, k=1)]

    logger.info("Residual cosine similarity")
    logger.info(f"mean: {upper_res.mean():.4f}")
    logger.info(f"std: {upper_res.std():.4f}")

# --------------------------------------------------
# Model analysis
# --------------------------------------------------

def run_model_analysis(model_root, model_name, logger):

    logger.info("\n================================")
    logger.info(f"MODEL: {model_name}")
    logger.info("================================")

    V, classes = compute_class_difference_vectors(model_root, logger)

    logger.info(f"Class difference: {classes[1]} - {classes[0]}")

    analyze(V, logger)


# --------------------------------------------------
# Dataset structure validation
# --------------------------------------------------

def validate_dataset_structure(dataset_root, logger):

    if not os.path.exists(dataset_root):
        raise ValueError(f"Dataset root does not exist: {dataset_root}")

    logger.info("\nValidating dataset structure...")

    models = ["dinov2", "resnet18", "clip"]

    for model_name in models:

        model_root = os.path.join(dataset_root, model_name)

        if not os.path.exists(model_root):
            logger.warning(f"{model_name} folder missing (will skip)")
            continue

        classes = [
            d for d in os.listdir(model_root)
            if os.path.isdir(os.path.join(model_root, d))
        ]

        if len(classes) < 2:
            raise ValueError(
                f"{model_name} must contain at least two class folders"
            )

        logger.info(f"{model_name}: classes detected -> {classes}")

        for cls in classes:

            cls_dir = os.path.join(model_root, cls)

            files = [
                f for f in os.listdir(cls_dir)
                if f.endswith(".npy")
                and os.path.isfile(os.path.join(cls_dir, f))
            ]

            if len(files) == 0:
                raise ValueError(
                    f"No .npy embeddings found in {cls_dir}"
                )

            sample = np.load(os.path.join(cls_dir, files[0]))
            dim = sample.shape[0]

            for f in files[1:]:

                emb = np.load(os.path.join(cls_dir, f))

                if emb.shape[0] != dim:
                    raise ValueError(
                        f"Inconsistent embedding dimensions in {cls_dir}"
                    )

        logger.info(f"{model_name}: structure OK")

    logger.info("Dataset validation completed successfully\n")


# --------------------------------------------------
# CLI
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        default="embedding_difference_dataset"
    )

    args = parser.parse_args()

    run_id, output_dir, log_file = log_run(
        code_filepath=__file__,
        dataset_name=args.dataset_name,
        base_output_dir=os.path.join(BASE_DIR, "output")
    )

    logger = setup_logger(log_file)

    logger.info("================================")
    logger.info("Difference Vector Analysis")
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Dataset Root: {args.dataset_root}")
    logger.info("================================")

    validate_dataset_structure(args.dataset_root, logger)

    models = ["dinov2", "resnet18", "clip"]

    for model_name in models:

        model_root = os.path.join(args.dataset_root, model_name)

        if not os.path.exists(model_root):

            logger.info(f"\nSkipping {model_name} (not found)")
            continue

        run_model_analysis(model_root, model_name, logger)


if __name__ == "__main__":
    main()