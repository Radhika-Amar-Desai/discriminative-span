import os

DATASET_NAME = "skin_lesion"


BASE_DIR = r"C:\Users\97433\Knowing_the_difference\data\embeddings\watermark_dataset"

SYNTHETIC_DATA_DIR = os.path.join(BASE_DIR, "synthetic_data")

DINOV2_EMBEDDING_FOLDER = os.path.join(SYNTHETIC_DATA_DIR, "raw", r"dinov2")
RESNET18_EMBEDDING_FOLDER = os.path.join(SYNTHETIC_DATA_DIR, "raw", r"resnet18")
CLIP_EMBEDDING_FOLDER = os.path.join(SYNTHETIC_DATA_DIR, "raw", r"clip")


C_VALUE = 1.0


# MODELS_BASE_DIR = r"C:\Users\97433\Knowing_the_difference\data\embeddings\skin_lesion\real_data\l1_scaled"

# DINOV2_joblib_model = os.path.join(MODELS_BASE_DIR,
#                                                 "halved_l1_scaled_dinov2",
#                                                 "classifier.joblib")
# DINOV2_weights_file = os.path.join(MODELS_BASE_DIR,
#                                                 "halved_l1_scaled_dinov2",
#                                                 "weights.npy")


# RESNET18_joblib_model = os.path.join(MODELS_BASE_DIR,
#                                                     "halved_l1_scaled_resnet18",
#                                                     "classifier.joblib")
# RESNET18_weights_file = os.path.join(MODELS_BASE_DIR,
#                                                     "halved_l1_scaled_resnet18",
#                                                     "weights.npy")


# CLIP_joblib_model = os.path.join(MODELS_BASE_DIR,
#                                                     "halved_l1_scaled_clip",
#                                                     "classifier.joblib")
# CLIP_weights_file = os.path.join(MODELS_BASE_DIR,
#                                                     "halved_l1_scaled_clip",
#                                                     "weights.npy")



L1_SCALED_DINOV2_EMBEDDING_FOLDER =     os.path.join(SYNTHETIC_DATA_DIR, "l1_scaled", r"dinov2")
L1_SCALED_RESNET18_EMBEDDING_FOLDER =   os.path.join(SYNTHETIC_DATA_DIR, "l1_scaled", r"resnet18")
L1_SCALED_CLIP_EMBEDDING_FOLDER =       os.path.join(SYNTHETIC_DATA_DIR, "l1_scaled", r"clip")


L2_SCALED_DINOV2_EMBEDDING_FOLDER =     os.path.join(SYNTHETIC_DATA_DIR, "l2_scaled", r"dinov2")
L2_SCALED_RESNET18_EMBEDDING_FOLDER =   os.path.join(SYNTHETIC_DATA_DIR, "l2_scaled", r"resnet18")
L2_SCALED_CLIP_EMBEDDING_FOLDER =       os.path.join(SYNTHETIC_DATA_DIR, "l2_scaled", r"clip")

SCALED_DINOV2_EMBEDDING_FOLDER = os.path.join(SYNTHETIC_DATA_DIR, "scaled", r"dinov2")
SCALED_RESNET18_EMBEDDING_FOLDER = os.path.join(SYNTHETIC_DATA_DIR, "scaled", r"resnet18")
SCALED_CLIP_EMBEDDING_FOLDER = os.path.join(SYNTHETIC_DATA_DIR, "scaled", r"clip")
