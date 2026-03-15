import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

sys.path.append(PROJECT_ROOT)

import argparse
import logging
import importlib.util
import shutil

import torch
import numpy as np

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

from tqdm import tqdm

import clip

from utils.auto_logger import BASE_DIR, log_run, register_dataset


device = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------------------------------
# Logger
# -----------------------------------------------------

def setup_logger(log_file):

    logger = logging.getLogger("embedding_logger")
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )

    file_handler = logging.FileHandler(log_file)
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
# Model loaders
# -----------------------------------------------------

def load_dinov2():

    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    model.eval()

    return model.to(device)


def load_resnet18():

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    model.fc = torch.nn.Identity()

    model.eval()

    return model.to(device)


def load_clip():

    model, preprocess = clip.load("ViT-B/32", device=device)

    model.eval()

    return model, preprocess


# -----------------------------------------------------
# Embedding extraction and saving
# -----------------------------------------------------

def extract_and_save_embeddings(model, loader, dataset, output_dir, model_name, logger):

    model_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    with torch.no_grad():

        idx = 0

        for images, labels in tqdm(loader, desc=f"{model_name} embedding"):

            images = images.to(device)

            if model_name == "clip":
                features = model.encode_image(images)
            else:
                features = model(images)

            features = features.cpu().numpy()

            batch_size = features.shape[0]

            for i in range(batch_size):

                img_path, class_idx = dataset.samples[idx]

                class_name = dataset.classes[class_idx]

                class_dir = os.path.join(model_dir, class_name)
                os.makedirs(class_dir, exist_ok=True)

                filename = os.path.splitext(os.path.basename(img_path))[0] + ".npy"

                save_path = os.path.join(class_dir, filename)

                np.save(save_path, features[i])

                idx += 1

    logger.info(f"{model_name} embeddings saved in {model_dir}")


# -----------------------------------------------------
# Main
# -----------------------------------------------------

# def main():

#     parser = argparse.ArgumentParser()

#     parser.add_argument("--config", type=str, required=True)

#     args = parser.parse_args()

#     config = load_config(args.config)

#     DATASET_NAME = getattr(config, "DATASET_NAME", "dataset")
#     DATASET_PATH = config.DATASET_PATH
#     BATCH_SIZE = getattr(config, "BATCH_SIZE", 32)
#     NUM_WORKERS = getattr(config, "NUM_WORKERS", 4)

#     OUTPUT_DIR = getattr(config, "OUTPUT_DIR", os.path.join(BASE_DIR, "output"))

#     run_id, output_dir, log_file = log_run(
#         code_filepath=__file__,
#         dataset_name=DATASET_NAME,
#         base_output_dir=OUTPUT_DIR
#     )

#     logger = setup_logger(log_file)

#     shutil.copy(args.config, os.path.join(output_dir, "config.py"))

#     logger.info("====================================")
#     logger.info("Embedding Generation")
#     logger.info(f"Run ID: {run_id}")
#     logger.info(f"Dataset: {DATASET_NAME}")
#     logger.info(f"Device: {device}")
#     logger.info("====================================")

#     register_dataset(DATASET_NAME, DATASET_PATH, DATASET_PATH, DATASET_PATH)

#     # ------------------------------------------------
#     # Dataset (standard transform)
#     # ------------------------------------------------

#     base_transform = transforms.Compose([
#         transforms.Resize((224,224)),
#         transforms.ToTensor()
#     ])

#     dataset = datasets.ImageFolder(
#         DATASET_PATH,
#         transform=base_transform
#     )

#     loader = DataLoader(
#         dataset,
#         batch_size=BATCH_SIZE,
#         shuffle=False,
#         num_workers=NUM_WORKERS
#     )

#     logger.info(f"Images: {len(dataset)}")
#     logger.info(f"Classes: {dataset.classes}")

#     # ------------------------------------------------
#     # DINOv2
#     # ------------------------------------------------

#     logger.info("Loading DINOv2")

#     dinov2 = load_dinov2()

#     extract_and_save_embeddings(
#         dinov2,
#         loader,
#         dataset,
#         output_dir,
#         "dinov2",
#         logger
#     )

#     # ------------------------------------------------
#     # ResNet18
#     # ------------------------------------------------

#     logger.info("Loading ResNet18")

#     resnet = load_resnet18()

#     extract_and_save_embeddings(
#         resnet,
#         loader,
#         dataset,
#         output_dir,
#         "resnet18",
#         logger
#     )

#     # ------------------------------------------------
#     # CLIP
#     # ------------------------------------------------

#     logger.info("Loading CLIP")

#     clip_model, preprocess = load_clip()

#     clip_dataset = datasets.ImageFolder(
#         DATASET_PATH,
#         transform=preprocess
#     )

#     clip_loader = DataLoader(
#         clip_dataset,
#         batch_size=BATCH_SIZE,
#         shuffle=False,
#         num_workers=NUM_WORKERS
#     )

#     extract_and_save_embeddings(
#         clip_model,
#         clip_loader,
#         clip_dataset,
#         output_dir,
#         "clip",
#         logger
#     )

#     logger.info("All embeddings generated successfully")

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)

    args = parser.parse_args()

    config = load_config(args.config)

    DATASET_NAME = getattr(config, "DATASET_NAME", "dataset")
    DATASET_PATH = config.DATASET_PATH
    BATCH_SIZE = getattr(config, "BATCH_SIZE", 32)
    NUM_WORKERS = getattr(config, "NUM_WORKERS", 4)

    OUTPUT_DIR = getattr(config, "OUTPUT_DIR", os.path.join(BASE_DIR, "output"))

    # ------------------------------------------------
    # Output directory (NO run_xxxx folder)
    # ------------------------------------------------

    output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    log_file = os.path.join(output_dir, "embedding_log.txt")

    logger = setup_logger(log_file)

    logger.info("====================================")
    logger.info("Embedding Generation")
    logger.info(f"Dataset: {DATASET_NAME}")
    logger.info(f"Device: {device}")
    logger.info(f"Output Dir: {output_dir}")
    logger.info("====================================")

    register_dataset(DATASET_NAME, DATASET_PATH, DATASET_PATH, DATASET_PATH)

    # ------------------------------------------------
    # Dataset (standard transform)
    # ------------------------------------------------

    base_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(
        DATASET_PATH,
        transform=base_transform
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    logger.info(f"Images: {len(dataset)}")
    logger.info(f"Classes: {dataset.classes}")

    # ------------------------------------------------
    # DINOv2
    # ------------------------------------------------

    logger.info("Loading DINOv2")

    dinov2 = load_dinov2()

    extract_and_save_embeddings(
        dinov2,
        loader,
        dataset,
        output_dir,
        "dinov2",
        logger
    )

    # ------------------------------------------------
    # ResNet18
    # ------------------------------------------------

    logger.info("Loading ResNet18")

    resnet = load_resnet18()

    extract_and_save_embeddings(
        resnet,
        loader,
        dataset,
        output_dir,
        "resnet18",
        logger
    )

    # ------------------------------------------------
    # CLIP
    # ------------------------------------------------

    logger.info("Loading CLIP")

    clip_model, preprocess = load_clip()

    clip_dataset = datasets.ImageFolder(
        DATASET_PATH,
        transform=preprocess
    )

    clip_loader = DataLoader(
        clip_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    extract_and_save_embeddings(
        clip_model,
        clip_loader,
        clip_dataset,
        output_dir,
        "clip",
        logger
    )

    logger.info("All embeddings generated successfully")


if __name__ == "__main__":
    main()