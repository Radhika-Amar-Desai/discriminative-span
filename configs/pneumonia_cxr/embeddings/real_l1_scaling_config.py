import os

DATASET_NAME = "pneumonia_cxr"

BASE_DIR = r"C:\Users\97433\Knowing_the_difference\data\embeddings\pneumonia_cxr\real_data"
DINOV2_EMBEDDING_FOLDER = os.path.join(BASE_DIR, "raw", r"dinov2")
RESNET18_EMBEDDING_FOLDER = os.path.join(BASE_DIR, "raw", r"resnet18")
CLIP_EMBEDDING_FOLDER = os.path.join(BASE_DIR, "raw", r"clip")


C_VALUE = 1.0


L1_SCALED_DINOV2_EMBEDDING_FOLDER =     os.path.join(BASE_DIR, "l1_scaled", r"halved_l1_scaled_dinov2")
L1_SCALED_RESNET18_EMBEDDING_FOLDER =   os.path.join(BASE_DIR, "l1_scaled", r"halved_l1_scaled_resnet18")
L1_SCALED_CLIP_EMBEDDING_FOLDER =       os.path.join(BASE_DIR, "l1_scaled", r"halved_l1_scaled_clip")
