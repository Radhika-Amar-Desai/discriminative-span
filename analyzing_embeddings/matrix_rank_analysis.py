import os
import csv
import numpy as np
import argparse


# -----------------------------------------------------
# Utils
# -----------------------------------------------------

def get_intersection(list_of_sets):
    return list(set.intersection(*list_of_sets))


def load_npy(path):
    return np.load(path)


# -----------------------------------------------------
# Rank Metrics
# -----------------------------------------------------

def get_span_rank(D):
    return np.linalg.matrix_rank(D)


def get_stable_rank(D):
    # stable rank = ||D||_F^2 / ||D||_2^2
    fro_norm = np.linalg.norm(D, 'fro') ** 2
    spectral_norm = np.linalg.norm(D, 2) ** 2
    return fro_norm / spectral_norm if spectral_norm != 0 else 0


def get_effective_rank(D):
    # based on entropy of singular values
    U, S, Vt = np.linalg.svd(D, full_matrices=False)
    S = S / np.sum(S)  # normalize
    entropy = -np.sum(S * np.log(S + 1e-12))
    return np.exp(entropy)


# -----------------------------------------------------
# Core Logic
# -----------------------------------------------------

def construct_difference_vectors(embedding_fldr):
    class_names = sorted(os.listdir(embedding_fldr))
    class_paths = list(filter(os.path.isdir, [os.path.join(embedding_fldr, c) for c in class_names]))

    # get common filenames across all classes
    file_sets = [set(os.listdir(p)) for p in class_paths]
    common_filenames = sorted(get_intersection(file_sets))

    D_list = []

    for fname in common_filenames:
        embeddings = []

        for path in class_paths:
            fpath = os.path.join(path, fname)
            emb = load_npy(fpath).flatten()
            embeddings.append(emb)

        # pairwise differences
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                diff = embeddings[i] - embeddings[j]
                D_list.append(diff)

    if len(D_list) == 0:
        raise ValueError("No difference vectors constructed")

    D = np.stack(D_list, axis=0)
    return D


def get_matrix_ranks(embedding_fldr):

    D = construct_difference_vectors(embedding_fldr)

    effective_rank = get_effective_rank(D)
    span_rank = get_span_rank(D)
    stable_rank = get_stable_rank(D)

    return effective_rank, span_rank, stable_rank


# -----------------------------------------------------
# CSV Writer
# -----------------------------------------------------

def write_to_csv_file(results, results_csv_path):

    header = ["embedding_type", "model_type", "effective_rank", "stable_rank", "span_rank"]

    with open(results_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(results)


# -----------------------------------------------------
# Main Analysis
# -----------------------------------------------------

def matrix_rank_analysis(embedding_parent_fldr, results_csv_path):

    results = []

    for embedding_type in sorted(os.listdir(embedding_parent_fldr)):

        embedding_subfldr_path = os.path.join(embedding_parent_fldr, embedding_type)

        if not os.path.isdir(embedding_subfldr_path):
            continue

        for model_type in sorted(os.listdir(embedding_subfldr_path)):

            embedding_model_path = os.path.join(
                embedding_parent_fldr, embedding_type, model_type
            )

            if not os.path.isdir(embedding_model_path):
                continue

            effective_rank, span_rank, stable_rank = get_matrix_ranks(
                embedding_model_path
            )

            row = [
                embedding_type,
                model_type,
                effective_rank,
                stable_rank,
                span_rank
            ]

            results.append(row)

    write_to_csv_file(results, results_csv_path)


# -----------------------------------------------------
# CLI Entry Point
# -----------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Matrix Rank Analysis on Embeddings")

    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to embedding parent folder"
    )

    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Path to save results CSV"
    )

    args = parser.parse_args()

    matrix_rank_analysis(args.input_dir, args.output_csv)


if __name__ == "__main__":
    main()