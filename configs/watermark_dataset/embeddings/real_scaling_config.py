import os

DATASET_NAME = "watermark_dataset"


BASE_DIR = r"C:\Users\97433\Knowing_the_difference\data\embeddings\watermark_dataset"

REAL_DATA_DIR = os.path.join(BASE_DIR, "real_data")

DINOV2_EMBEDDING_FOLDER = os.path.join(REAL_DATA_DIR, "raw", r"dinov2")
RESNET18_EMBEDDING_FOLDER = os.path.join(REAL_DATA_DIR, "raw", r"resnet18")
CLIP_EMBEDDING_FOLDER = os.path.join(REAL_DATA_DIR, "raw", r"clip")

C_VALUE = 1.0

L1_SCALED_DINOV2_EMBEDDING_FOLDER =     os.path.join(REAL_DATA_DIR,
                                                                    "l1_scaled", r"dinov2")
L1_SCALED_RESNET18_EMBEDDING_FOLDER =   os.path.join(REAL_DATA_DIR,
                                                                    "l1_scaled", r"resnet18")
L1_SCALED_CLIP_EMBEDDING_FOLDER =       os.path.join(REAL_DATA_DIR,
                                                                    "l1_scaled", r"clip")

L2_SCALED_DINOV2_EMBEDDING_FOLDER =     os.path.join(REAL_DATA_DIR,
                                                                    "l2_scaled", r"halved_l2_scaled_dinov2")
L2_SCALED_RESNET18_EMBEDDING_FOLDER =   os.path.join(REAL_DATA_DIR,
                                                                    "l2_scaled", r"halved_l2_scaled_resnet18")
L2_SCALED_CLIP_EMBEDDING_FOLDER =       os.path.join(REAL_DATA_DIR,
                                                                    "l2_scaled", r"halved_l2_scaled_clip")

SCALED_DINOV2_EMBEDDING_FOLDER = os.path.join(REAL_DATA_DIR, "scaled", r"dinov2")
SCALED_RESNET18_EMBEDDING_FOLDER = os.path.join(REAL_DATA_DIR, "scaled", r"resnet18")
SCALED_CLIP_EMBEDDING_FOLDER = os.path.join(REAL_DATA_DIR, "scaled", r"clip")
