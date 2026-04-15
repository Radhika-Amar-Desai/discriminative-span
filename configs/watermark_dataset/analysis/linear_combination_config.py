import os

DATASET_NAME = "skin_lesion_experiment"

OUTPUT_DIR = r"C:\Users\97433\Knowing_the_difference\output"

ridge_lambda = 0.01

BASE_DATA_DIR = r'C:\Users\97433\Knowing_the_difference\data\embeddings\watermark_dataset'
BASE_SYNTHETIC_DATA_DIR = os.path.join(BASE_DATA_DIR, 'synthetic_data')
BASE_REAL_DATA_DIR = os.path.join(BASE_DATA_DIR, 'real_data')

L1_SCALED_SYNTHETIC_DATA_DIR = os.path.join(BASE_SYNTHETIC_DATA_DIR, 'l1_scaled')

L2_SCALED_SYNTHETIC_DATA_DIR = os.path.join(BASE_SYNTHETIC_DATA_DIR, 'l2_scaled')

SCALED_SYNTHETIC_DATA_DIR = os.path.join(BASE_SYNTHETIC_DATA_DIR, 'scaled')

RAW_SYNTHETIC_DATA_DIR = os.path.join(BASE_SYNTHETIC_DATA_DIR, 'raw')

CLASS_A = 'original'
CLASS_B = 'watermarked'


EMBEDDINGS = {

    # "l1": {

    #     "resnet18": {
    #         "real_A": os.path.join(BASE_REAL_DATA_DIR, 'resnet18', CLASS_A),
    #         "synthetic_B": os.path.join(L1_SCALED_SYNTHETIC_DATA_DIR, 'resnet18', CLASS_B),
    #         "real_B": os.path.join(BASE_REAL_DATA_DIR, 'resnet18', CLASS_B),
    #         "classifier_weights": os.path.join(L1_SCALED_SYNTHETIC_DATA_DIR, 'resnet18', 'weights.npy')
    #     },

    #     "clip": {
    #         "real_A": os.path.join(BASE_REAL_DATA_DIR, 'clip', CLASS_A),
    #         "synthetic_B": os.path.join(L1_SCALED_SYNTHETIC_DATA_DIR, 'clip', CLASS_B),
    #         "real_B": os.path.join(BASE_REAL_DATA_DIR, 'clip', CLASS_B),
    #         "classifier_weights": os.path.join(L1_SCALED_SYNTHETIC_DATA_DIR, 'clip', 'weights.npy')
    #     },

    #     "dinov2": {
    #         "real_A": os.path.join(BASE_REAL_DATA_DIR, 'dinov2', CLASS_A),
    #         "synthetic_B": os.path.join(L1_SCALED_SYNTHETIC_DATA_DIR, 'dinov2', CLASS_B),
    #         "real_B": os.path.join(BASE_REAL_DATA_DIR, 'dinov2', CLASS_B),
    #         "classifier_weights": os.path.join(L1_SCALED_SYNTHETIC_DATA_DIR, 'dinov2', 'weights.npy')
    #     }

    # },

    # "l2": {
    #     "resnet18": {
    #         "real_A": os.path.join(BASE_REAL_DATA_DIR, 'resnet18', CLASS_A),
    #         "synthetic_B": os.path.join(L2_SCALED_SYNTHETIC_DATA_DIR, 'resnet18', CLASS_B),
    #         "real_B": os.path.join(BASE_REAL_DATA_DIR, 'resnet18', CLASS_B),
    #         "classifier_weights": os.path.join(L2_SCALED_SYNTHETIC_DATA_DIR, 'resnet18', 'weights.npy')
    #     },

    #     "clip": {
    #         "real_A": os.path.join(BASE_REAL_DATA_DIR, 'clip', CLASS_A),
    #         "synthetic_B": os.path.join(L2_SCALED_SYNTHETIC_DATA_DIR, 'clip', CLASS_B),
    #         "real_B": os.path.join(BASE_REAL_DATA_DIR, 'clip', CLASS_B),
    #         "classifier_weights": os.path.join(L2_SCALED_SYNTHETIC_DATA_DIR, 'clip', 'weights.npy')
    #     },

    #     "dinov2": {
    #         "real_A": os.path.join(BASE_REAL_DATA_DIR, 'dinov2', CLASS_A),
    #         "synthetic_B": os.path.join(L2_SCALED_SYNTHETIC_DATA_DIR, 'dinov2', CLASS_B),
    #         "real_B": os.path.join(BASE_REAL_DATA_DIR, 'dinov2', CLASS_B),
    #         "classifier_weights": os.path.join(L2_SCALED_SYNTHETIC_DATA_DIR, 'dinov2', 'weights.npy')
    #     }
    # },

    # "scaled": {
    #     "resnet18": {
    #         "real_A": os.path.join(BASE_REAL_DATA_DIR, 'resnet18', CLASS_A),
    #         "synthetic_B": os.path.join(SCALED_SYNTHETIC_DATA_DIR, 'resnet18', CLASS_B),
    #         "real_B": os.path.join(BASE_REAL_DATA_DIR, 'resnet18', CLASS_B),
    #         "classifier_weights": os.path.join(SCALED_SYNTHETIC_DATA_DIR, 'resnet18', 'weights.npy')
    #     },

    #     "clip": {
    #         "real_A": os.path.join(BASE_REAL_DATA_DIR, 'clip', CLASS_A),
    #         "synthetic_B": os.path.join(SCALED_SYNTHETIC_DATA_DIR, 'clip', CLASS_B),
    #         "real_B": os.path.join(BASE_REAL_DATA_DIR, 'clip', CLASS_B),
    #         "classifier_weights": os.path.join(SCALED_SYNTHETIC_DATA_DIR, 'clip', 'weights.npy')
    #     },

    #     "dinov2": {
    #         "real_A": os.path.join(BASE_REAL_DATA_DIR, 'dinov2', CLASS_A),
    #         "synthetic_B": os.path.join(SCALED_SYNTHETIC_DATA_DIR, 'dinov2', CLASS_B),
    #         "real_B": os.path.join(BASE_REAL_DATA_DIR, 'dinov2', CLASS_B),
    #         "classifier_weights": os.path.join(SCALED_SYNTHETIC_DATA_DIR, 'dinov2', 'weights.npy')
    #     }
    # },

    "raw": {
        "resnet18": {
            "real_A": os.path.join(BASE_REAL_DATA_DIR, 'raw', 'resnet18', CLASS_A),
            "synthetic_A": os.path.join(RAW_SYNTHETIC_DATA_DIR, 'resnet18', CLASS_A),
            "synthetic_B": os.path.join(RAW_SYNTHETIC_DATA_DIR, 'resnet18', CLASS_B),
            "real_B": os.path.join(BASE_REAL_DATA_DIR, 'raw', 'resnet18', CLASS_B)
        },

        "clip": {
            "real_A": os.path.join(BASE_REAL_DATA_DIR, 'raw', 'clip', CLASS_A),
            "synthetic_A": os.path.join(RAW_SYNTHETIC_DATA_DIR, 'clip', CLASS_A),
            "synthetic_B": os.path.join(RAW_SYNTHETIC_DATA_DIR, 'clip', CLASS_B),
            "real_B": os.path.join(BASE_REAL_DATA_DIR, 'raw', 'clip', CLASS_B)
        },

        "dinov2": {
            "real_A": os.path.join(BASE_REAL_DATA_DIR, 'raw', 'dinov2', CLASS_A),
            "synthetic_A": os.path.join(RAW_SYNTHETIC_DATA_DIR, 'dinov2', CLASS_A),
            "synthetic_B": os.path.join(RAW_SYNTHETIC_DATA_DIR, 'dinov2', CLASS_B),
            "real_B": os.path.join(BASE_REAL_DATA_DIR, 'raw', 'dinov2', CLASS_B)
        }
    }
}