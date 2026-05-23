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


def verify_folder_structure(embedding_fldr, verbose=False):
    """
    Verifies that the embedding folder structure is valid for difference vector construction.

    Expected structure:
    embedding_fldr/
        class1/
            file1.npy
            file2.npy
        class2/
            file1.npy
            file2.npy
    """

    if not os.path.isdir(embedding_fldr):
        raise ValueError(f"{embedding_fldr} is not a valid directory")

    class_names = sorted(os.listdir(embedding_fldr))
    class_paths = [
        os.path.join(embedding_fldr, c)
        for c in class_names
        if os.path.isdir(os.path.join(embedding_fldr, c))
    ]

    if len(class_paths) < 2:
        raise ValueError("Need at least 2 class folders for difference vectors")

    file_sets = []

    for class_path in class_paths:
        files = os.listdir(class_path)

        npy_files = set()
        for f in files:
            fpath = os.path.join(class_path, f)

            if os.path.isdir(fpath):
                if verbose:
                    print(f"⚠️ Warning: Nested directory found: {fpath}")

            elif f.endswith(".npy"):
                npy_files.add(f)

            else:
                if verbose:
                    print(f"⚠️ Ignoring non-npy file: {fpath}")

        if len(npy_files) == 0:
            raise ValueError(f"No .npy files found in {class_path}")

        file_sets.append(npy_files)

    # Check intersection
    common_files = set.intersection(*file_sets)

    if len(common_files) == 0:
        raise ValueError("No common .npy filenames across classes")

    # Check consistency
    for i, file_set in enumerate(file_sets):
        missing = common_files.symmetric_difference(file_set)
        if missing and verbose:
            print(f"⚠️ Class {class_names[i]} has mismatched files: {missing}")

    if verbose:
        print(f"✅ Found {len(class_paths)} classes")
        print(f"✅ {len(common_files)} common aligned .npy files")

    return True

# -----------------------------------------------------
# Conditioning Metrics
# -----------------------------------------------------

def get_singular_values(D):
    _, S, _ = np.linalg.svd(D, full_matrices=False)
    return S


def get_condition_number(D):
    S = get_singular_values(D)
    if len(S) == 0:
        return 0
    sigma_max = np.max(S)
    sigma_min = np.min(S)

    # avoid divide by zero
    if sigma_min < 1e-12:
        return np.inf
    return sigma_max / sigma_min


def get_min_singular_value(D):
    S = get_singular_values(D)
    return np.min(S) if len(S) > 0 else 0


def get_spectrum_decay_ratio(D, k=5):
    """
    Ratio of top-k singular values energy to total energy
    Helps show low-dimensional structure
    """
    S = get_singular_values(D)
    if len(S) == 0:
        return 0

    total_energy = np.sum(S)
    top_k_energy = np.sum(S[:min(k, len(S))])

    return top_k_energy / total_energy if total_energy != 0 else 0

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

# def construct_difference_vectors(embedding_fldr):
#     class_names = sorted(os.listdir(embedding_fldr))
#     class_paths = list(filter(os.path.isdir, [os.path.join(embedding_fldr, c) for c in class_names]))

#     # get common filenames across all classes
#     file_sets = [set(os.listdir(p)) for p in class_paths]
#     common_filenames = sorted(get_intersection(file_sets))

#     D_list = []

#     for fname in common_filenames:
#         embeddings = []

#         for path in class_paths:
#             fpath = os.path.join(path, fname)
#             emb = load_npy(fpath).flatten()
#             embeddings.append(emb)

#         # pairwise differences
#         for i in range(len(embeddings)):
#             for j in range(i + 1, len(embeddings)):
#                 diff = embeddings[i] - embeddings[j]
#                 D_list.append(diff)

#     if len(D_list) == 0:
#         raise ValueError("No difference vectors constructed")

#     D = np.stack(D_list, axis=0)
#     return D

def construct_difference_vectors(embedding_fldr):
    """
    Constructs aligned difference vectors between TWO classes only.
    (Same logic as span_analysis code)
    """

    class_names = sorted(os.listdir(embedding_fldr))
    class_paths = [
        os.path.join(embedding_fldr, c)
        for c in class_names
        if os.path.isdir(os.path.join(embedding_fldr, c))
    ]

    if len(class_paths) != 2:
        raise ValueError(
            f"Expected exactly 2 classes for aligned differences, got {len(class_paths)}"
        )

    path_A, path_B = class_paths

    files_A = {f for f in os.listdir(path_A) if f.endswith(".npy")}
    files_B = {f for f in os.listdir(path_B) if f.endswith(".npy")}

    common_files = sorted(files_A.intersection(files_B))

    if len(common_files) == 0:
        raise ValueError("No matching filenames between classes")

    A = []
    B = []

    for fname in common_files:
        emb_A = np.load(os.path.join(path_A, fname)).flatten()
        emb_B = np.load(os.path.join(path_B, fname)).flatten()

        A.append(emb_A)
        B.append(emb_B)

    A = np.stack(A, axis=0)
    B = np.stack(B, axis=0)

    D = B - A

    return D

def get_matrix_stats(embedding_fldr):

    verify_folder_structure(embedding_fldr)
    D = construct_difference_vectors(embedding_fldr)

    effective_rank = get_effective_rank(D)
    span_rank = get_span_rank(D)
    stable_rank = get_stable_rank(D)

    condition_number = get_condition_number(D)
    min_singular = get_min_singular_value(D)
    spectrum_ratio = get_spectrum_decay_ratio(D, k=5)

    return (
        effective_rank,
        span_rank,
        stable_rank,
        condition_number,
        min_singular,
        spectrum_ratio
    )


# -----------------------------------------------------
# CSV Writer
# -----------------------------------------------------

def write_to_csv_file(results, results_csv_path):

    header = [
    "embedding_type",
    "model_type",
    "effective_rank",
    "stable_rank",
    "span_rank",
    "condition_number",
    "min_singular_value",
    "top5_spectrum_ratio"]

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

            (
                effective_rank,
                span_rank,
                stable_rank,
                condition_number,
                min_singular,
                spectrum_ratio
            ) = get_matrix_stats(embedding_model_path)

            row = [
                embedding_type,
                model_type,
                effective_rank,
                stable_rank,
                span_rank,
                condition_number,
                min_singular,
                spectrum_ratio
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