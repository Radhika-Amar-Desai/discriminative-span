import os
import sys

# --------------------------------------------------
# Make project root importable
# --------------------------------------------------

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

sys.path.append(PROJECT_ROOT)

import argparse
import logging
import shutil
import importlib.util

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from utils.auto_logger import BASE_DIR, log_run, register_dataset


device = "cuda" if torch.cuda.is_available() else "cpu"

DESCRIPTION = """
Flexible binary classifier training script with config-driven experiments.
"""


# -----------------------------------------------------
# Logging setup
# -----------------------------------------------------

def setup_logger(log_file):

    logger = logging.getLogger("experiment_logger")
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
# Utility for safe config access
# -----------------------------------------------------

def get_config_value(config, name, default):
    return getattr(config, name, default)


# -----------------------------------------------------
# Evaluation
# -----------------------------------------------------

def evaluate(model, loader, logger):

    model.eval()

    preds = []
    labels = []

    with torch.no_grad():

        for x, y in loader:

            x = x.to(device)
            y = y.to(device)

            out = model(x)
            pred = torch.argmax(out, dim=1)

            preds.extend(pred.cpu().numpy())
            labels.extend(y.cpu().numpy())

    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)

    logger.info(f"Accuracy  : {acc:.4f}")
    logger.info(f"Precision : {precision:.4f}")
    logger.info(f"Recall    : {recall:.4f}")
    logger.info(f"F1 Score  : {f1:.4f}")

    return acc


# -----------------------------------------------------
# Training
# -----------------------------------------------------

def train(model, train_loader, optimizer, criterion, epochs, logger):

    model.train()

    for epoch in range(epochs):

        total_loss = 0

        for x, y in train_loader:

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            out = model(x)

            loss = criterion(out, y)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")


# -----------------------------------------------------
# Model Builder
# -----------------------------------------------------

def build_model(model_name, num_classes, pretrained, checkpoint_path, logger):

    model_name = model_name.lower()

    if model_name == "resnet18":

        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "efficientnet_b0":

        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif model_name == "mobilenet_v2":

        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    if checkpoint_path:

        logger.info(f"Loading checkpoint from {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state)

    return model.to(device)


# -----------------------------------------------------
# Load config
# -----------------------------------------------------

def load_config(path):

    spec = importlib.util.spec_from_file_location("config", path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    return config


# -----------------------------------------------------
# Main
# -----------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        required=True
    )

    args = parser.parse_args()

    config = load_config(args.config)

    # Defaults
    DATASET_NAME = get_config_value(config, "DATASET_NAME", "dataset")
    TRAIN_PATH = get_config_value(config, "TRAIN_PATH", None)
    VAL_PATH = get_config_value(config, "VAL_PATH", None)
    TEST_PATH = get_config_value(config, "TEST_PATH", None)

    MODEL_NAME = get_config_value(config, "MODEL_NAME", "resnet18")
    PRETRAINED = get_config_value(config, "PRETRAINED", True)
    CHECKPOINT = get_config_value(config, "CHECKPOINT_PATH", None)

    BATCH_SIZE = get_config_value(config, "BATCH_SIZE", 32)
    LR = get_config_value(config, "LR", 1e-4)
    EPOCHS = get_config_value(config, "EPOCHS", 10)

    # --------------------------------------------
    # Run logging
    # --------------------------------------------

    run_id, output_dir, log_file = log_run(
        code_filepath=__file__,
        dataset_name=DATASET_NAME,
        base_output_dir=os.path.join(BASE_DIR, "output")
    )

    logger = setup_logger(log_file)

    shutil.copy(args.config, os.path.join(output_dir, "config.py"))

    register_dataset(
        DATASET_NAME,
        TRAIN_PATH,
        TEST_PATH,
        VAL_PATH
    )

    logger.info(DESCRIPTION)

    logger.info("====================================")
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Dataset: {DATASET_NAME}")
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"Pretrained: {PRETRAINED}")
    logger.info(f"Checkpoint: {CHECKPOINT}")
    logger.info(f"Device: {device}")
    logger.info("====================================")

    # --------------------------------------------
    # Data
    # --------------------------------------------

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    train_dataset = datasets.ImageFolder(TRAIN_PATH, transform=transform)
    val_dataset = datasets.ImageFolder(VAL_PATH, transform=transform)
    test_dataset = datasets.ImageFolder(TEST_PATH, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")

    # --------------------------------------------
    # Model
    # --------------------------------------------

    model = build_model(
        MODEL_NAME,
        num_classes=2,
        pretrained=PRETRAINED,
        checkpoint_path=CHECKPOINT,
        logger=logger
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    logger.info(f"Epochs: {EPOCHS}")
    logger.info(f"Batch size: {BATCH_SIZE}")
    logger.info(f"Learning rate: {LR}")

    # --------------------------------------------
    # Train
    # --------------------------------------------

    train(
        model,
        train_loader,
        optimizer,
        criterion,
        EPOCHS,
        logger
    )

    # --------------------------------------------
    # Evaluate
    # --------------------------------------------

    logger.info("Test Evaluation")

    evaluate(model, test_loader, logger)

    # --------------------------------------------
    # Save model
    # --------------------------------------------

    model_path = os.path.join(output_dir, "model.pt")

    torch.save(model.state_dict(), model_path)

    logger.info(f"Model saved to {model_path}")
    logger.info("Experiment finished")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise
    except Exception:
        logging.exception("Unhandled exception occurred")
        raise