import os

DATASET_NAME = "toy_dataset"

OUTPUT_LOG_DIR = r"C:\Users\97433\Knowing_the_difference\output"

BASE_DIR = r"C:\Users\97433\Knowing_the_difference\data\embeddings\toy_dataset\synthetic_data"
DINOV2_EMBEDDING_FOLDER = os.path.join(BASE_DIR, "raw", r"dinov2")
RESNET18_EMBEDDING_FOLDER = os.path.join(BASE_DIR, "raw", r"resnet18")
CLIP_EMBEDDING_FOLDER = os.path.join(BASE_DIR, "raw", r"clip")


C_VALUE = 1.0


L1_SCALED_DINOV2_EMBEDDING_FOLDER = os.path.join(BASE_DIR, "l1_scaled", r"dinov2")
L1_SCALED_RESNET18_EMBEDDING_FOLDER = os.path.join(BASE_DIR, "l1_scaled", r"resnet18")
L1_SCALED_CLIP_EMBEDDING_FOLDER = os.path.join(BASE_DIR, "l1_scaled", r"clip")

L2_SCALED_DINOV2_EMBEDDING_FOLDER = os.path.join(BASE_DIR, "l2_scaled", r"dinov2")
L2_SCALED_RESNET18_EMBEDDING_FOLDER = os.path.join(BASE_DIR, "l2_scaled", r"resnet18")
L2_SCALED_CLIP_EMBEDDING_FOLDER = os.path.join(BASE_DIR, "l2_scaled", r"clip")

SCALED_DINOV2_EMBEDDING_FOLDER = os.path.join(BASE_DIR, "scaled", r"dinov2")
SCALED_RESNET18_EMBEDDING_FOLDER = os.path.join(BASE_DIR, "scaled", r"resnet18")
SCALED_CLIP_EMBEDDING_FOLDER = os.path.join(BASE_DIR, "scaled", r"clip")
