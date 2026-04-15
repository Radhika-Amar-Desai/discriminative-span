import os
import csv
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt

# -----------------------------------------------------
# Dataset
# -----------------------------------------------------
class ImageFolderDataset(torch.utils.data.Dataset):
    def __init__(self, folder, transform):
        self.paths = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


# -----------------------------------------------------
# Model Loader
# -----------------------------------------------------
def load_model(model_name, checkpoint_path, device):
    if model_name == "resnet18":
        model = models.resnet18(pretrained=False)
        model.fc = nn.Identity()

    elif model_name == "mobilenetv2":
        model = models.mobilenet_v2(pretrained=False)
        model.classifier = nn.Identity()

    elif model_name == "efficientnetb0":
        model = models.efficientnet_b0(pretrained=False)
        model.classifier = nn.Identity()

    else:
        raise ValueError("Unsupported model")

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)

    model.to(device)
    model.eval()

    return model


# -----------------------------------------------------
# Embedding Extraction
# -----------------------------------------------------
def extract_embeddings(model, dataloader, device):
    embeddings = []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            feats = model(batch)
            embeddings.append(feats.cpu().numpy())

    return np.concatenate(embeddings, axis=0)


# -----------------------------------------------------
# Difference Matrix
# -----------------------------------------------------
def compute_difference_matrix(A, B):
    assert A.shape == B.shape
    return B - A


# -----------------------------------------------------
# Subspace Functions (your original code)
# -----------------------------------------------------
def principal_angles(U1, U2):
    M = U1.T @ U2
    _, s, _ = np.linalg.svd(M)
    s = np.clip(s, -1.0, 1.0)
    angles = np.arccos(s)
    return angles, s


def subspace_alignment_score(D1, D2, k=None):
    U1, S1, _ = np.linalg.svd(D1, full_matrices=False)
    U2, S2, _ = np.linalg.svd(D2, full_matrices=False)

    if k is None:
        k = min(
            np.linalg.matrix_rank(D1),
            np.linalg.matrix_rank(D2)
        )

    k = min(k, U1.shape[1], U2.shape[1])

    U1_k = U1[:, :k]
    U2_k = U2[:, :k]

    angles, cosines = principal_angles(U1_k, U2_k)

    return {
        "k": k,
        "mean_cosine": float(np.mean(cosines)),
        "min_cosine": float(np.min(cosines)),
        "cosines": cosines,
        "angles": angles
    }


# -----------------------------------------------------
# Plotting
# -----------------------------------------------------
def plot_cosines(cosines, save_path):
    plt.figure()
    plt.plot(sorted(cosines, reverse=True))
    plt.title("Principal Cosines Spectrum")
    plt.xlabel("Index")
    plt.ylabel("Cosine")
    plt.savefig(save_path)
    plt.close()


# -----------------------------------------------------
# Main Pipeline
# -----------------------------------------------------
def run_analysis(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transform
    transform = transforms.Compose([
        transforms.Resize((config["data"]["image_size"], config["data"]["image_size"])),
        transforms.ToTensor()
    ])

    # Datasets
    dataset_A = ImageFolderDataset(config["data"]["class_A_folder"], transform)
    dataset_B = ImageFolderDataset(config["data"]["class_B_folder"], transform)

    loader_A = DataLoader(dataset_A, batch_size=config["data"]["batch_size"], shuffle=False)
    loader_B = DataLoader(dataset_B, batch_size=config["data"]["batch_size"], shuffle=False)

    # Model
    model = load_model(
        config["model_name"],
        config["checkpoint_path"],
        device
    )

    # Embeddings
    emb_A = extract_embeddings(model, loader_A, device)
    emb_B = extract_embeddings(model, loader_B, device)

    # Difference matrix
    D = compute_difference_matrix(emb_A, emb_B)

    # For now compare with itself (placeholder if you add foundation model later)
    results = subspace_alignment_score(D, D, k=config["analysis"]["k"])

    # Save CSV
    os.makedirs(os.path.dirname(config["output"]["csv_path"]), exist_ok=True)

    file_exists = os.path.isfile(config["output"]["csv_path"])

    with open(config["output"]["csv_path"], "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "k", "mean_cosine", "min_cosine"])

        if not file_exists:
            writer.writeheader()

        writer.writerow({
            "model": config["model_name"],
            "k": results["k"],
            "mean_cosine": results["mean_cosine"],
            "min_cosine": results["min_cosine"]
        })

    # Plot
    os.makedirs(config["output"]["plot_dir"], exist_ok=True)

    plot_path = os.path.join(
        config["output"]["plot_dir"],
        f"{config['model_name']}_cosines.png"
    )

    plot_cosines(results["cosines"], plot_path)

    print("✅ Analysis complete")
    print(results)


# -----------------------------------------------------
# Entry Point
# -----------------------------------------------------
if __name__ == "__main__":
    import importlib.util
    import sys

    config_path = sys.argv[1]

    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    CONFIG = config_module.CONFIG

    run_analysis(CONFIG)