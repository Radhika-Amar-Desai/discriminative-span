import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

sys.path.append(PROJECT_ROOT)

import argparse
import logging
import importlib.util
import shutil

import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from utils.auto_logger import log_run


# -----------------------------------------------------
# Logger
# -----------------------------------------------------

def setup_logger(log_file):

    logger = logging.getLogger("l1_classifier_logger")
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


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

    embeddings = []
    labels = []
    paths = []

    classes = [
        d for d in sorted(os.listdir(folder))
        if os.path.isdir(os.path.join(folder, d))
    ]

    for label_idx, class_name in enumerate(classes):

        class_dir = os.path.join(folder, class_name)

        files = [
            f for f in os.listdir(class_dir)
            if f.endswith(".npy")
        ]

        for file in files:

            path = os.path.join(class_dir, file)

            emb = np.load(path)

            embeddings.append(emb)
            labels.append(label_idx)
            paths.append((class_name, file))

    embeddings = np.array(embeddings)
    labels = np.array(labels)

    return embeddings, labels, classes, paths


# -----------------------------------------------------
# Save scaled embeddings
# -----------------------------------------------------

def save_scaled_embeddings(paths, scaled_embeddings, output_folder):

    os.makedirs(output_folder, exist_ok=True)

    for (class_name, filename), emb in zip(paths, scaled_embeddings):

        class_dir = os.path.join(output_folder, class_name)

        os.makedirs(class_dir, exist_ok=True)

        save_path = os.path.join(class_dir, filename)

        np.save(save_path, emb)


# -----------------------------------------------------
# Train + scale pipeline
# -----------------------------------------------------

def process_embedding_folder(embedding_folder, output_folder, C_VALUE, logger):

    logger.info(f"Processing embeddings: {embedding_folder}")

    X, y, classes, paths = load_embeddings(embedding_folder)

    logger.info(f"Loaded embeddings shape: {X.shape}")

    # ------------------------------------------------
    # Embedding dimensionality analysis
    # ------------------------------------------------

    total_dims = X.shape[1]

    nonzero_mask = np.any(X != 0, axis=0)

    active_dims = np.sum(nonzero_mask)

    zero_dims = total_dims - active_dims

    logger.info("Embedding dimensional analysis")
    logger.info(f"Total embedding dimensions : {total_dims}")
    logger.info(f"Active (non-zero) dims     : {active_dims}")
    logger.info(f"Completely zero dims       : {zero_dims}")

    # ------------------------------------------------
    # Select 50% subset for L1 training
    # ------------------------------------------------

    rng = np.random.default_rng(42)

    total_samples = len(X)

    train_indices = rng.choice(total_samples, size=total_samples // 2, replace=False)

    X_train = X[train_indices]
    y_train = y[train_indices]

    logger.info("L1 training subset selection")
    logger.info(f"Total samples              : {total_samples}")
    logger.info(f"Training samples used for L1 : {len(train_indices)}")

    # ------------------------------------------------
    # Train L1 Logistic Regression
    # ------------------------------------------------

    clf = LogisticRegression(
        penalty="l1",
        solver="liblinear",
        C=C_VALUE,
        max_iter=1000
    )

    clf.fit(X_train, y_train)

    weights = clf.coef_[0]

    non_zero = np.sum(weights != 0)

    logger.info(f"Classifier feature dimensions : {len(weights)}")
    logger.info(f"Non-zero classifier weights   : {non_zero}")

    # ------------------------------------------------
    # Evaluate classifier on ALL data
    # ------------------------------------------------

    preds = clf.predict(X)

    acc = accuracy_score(y, preds)
    precision = precision_score(y, preds)
    recall = recall_score(y, preds)
    f1 = f1_score(y, preds)

    logger.info("Evaluation on full dataset")
    logger.info(f"Accuracy  : {acc:.4f}")
    logger.info(f"Precision : {precision:.4f}")
    logger.info(f"Recall    : {recall:.4f}")
    logger.info(f"F1 Score  : {f1:.4f}")

    # ------------------------------------------------
    # Scale embeddings using classifier weights
    # ------------------------------------------------

    scaled_embeddings = X * weights

    save_scaled_embeddings(paths, scaled_embeddings, output_folder)

    logger.info(f"Scaled embeddings saved to {output_folder}")

    # ------------------------------------------------
    # Save classifier + weights
    # ------------------------------------------------

    joblib.dump(clf, os.path.join(output_folder, "classifier.joblib"))
    np.save(os.path.join(output_folder, "weights.npy"), weights)

    logger.info("Classifier and weights saved")


# -----------------------------------------------------
# Main
# -----------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=True)

    args = parser.parse_args()

    config = load_config(args.config)

    DATASET_NAME = getattr(config, "DATASET_NAME", "embedding_dataset")

    run_id, _, log_file = log_run(
        code_filepath=__file__,
        dataset_name=DATASET_NAME,
        base_output_dir=os.path.dirname(config.BASE_DIR)
    )

    logger = setup_logger(log_file)

    logger.info("====================================")
    logger.info("L1 Logistic Regression Scaling")
    logger.info(f"Run ID: {run_id}")
    logger.info("====================================")

    C_VALUE = getattr(config, "C_VALUE", 1.0)

    # ------------------------------------------------
    # Process each embedding space
    # ------------------------------------------------

    process_embedding_folder(
        config.DINOV2_EMBEDDING_FOLDER,
        config.L1_SCALED_DINOV2_EMBEDDING_FOLDER,
        C_VALUE,
        logger
    )

    process_embedding_folder(
        config.RESNET18_EMBEDDING_FOLDER,
        config.L1_SCALED_RESNET18_EMBEDDING_FOLDER,
        C_VALUE,
        logger
    )

    process_embedding_folder(
        config.CLIP_EMBEDDING_FOLDER,
        config.L1_SCALED_CLIP_EMBEDDING_FOLDER,
        C_VALUE,
        logger
    )

    logger.info("All embedding spaces processed successfully")


if __name__ == "__main__":
    main()