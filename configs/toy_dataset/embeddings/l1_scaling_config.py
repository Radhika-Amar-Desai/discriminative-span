import os

DATASET_NAME = "skin_lesion_dinov2"

BASE_DIR = r"C:\Users\97433\Knowing_the_difference\data\embeddings\toy_dataset\synthetic_data"
DINOV2_EMBEDDING_FOLDER = os.path.join(BASE_DIR, r"dinov2")
RESNET18_EMBEDDING_FOLDER = os.path.join(BASE_DIR, r"resnet18")
CLIP_EMBEDDING_FOLDER = os.path.join(BASE_DIR, r"clip")


C_VALUE = 1.0


L1_SCALED_DINOV2_EMBEDDING_FOLDER = os.path.join(BASE_DIR, r"l1_scaled_dinov2")
L1_SCALED_RESNET18_EMBEDDING_FOLDER = os.path.join(BASE_DIR, r"l1_scaled_resnet18")
L1_SCALED_CLIP_EMBEDDING_FOLDER = os.path.join(BASE_DIR, r"l1_scaled_clip")
