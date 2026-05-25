import os

DATASET_NAME = "pneumonia_cxr"

OUTPUT_DIR = r"C:\Users\97433\Knowing_the_difference\output"

ridge_lambda = 0.01

BASE_DATA_DIR = r'C:\Users\97433\Knowing_the_difference\data\embeddings\pneumonia_cxr'
BASE_SYNTHETIC_DATA_DIR = os.path.join(BASE_DATA_DIR, 'augmented_mirror')
RAW_REAL_DATA_DIR = os.path.join(BASE_DATA_DIR, 'real_data', 'raw')

L1_SCALED_SYNTHETIC_DATA_DIR = os.path.join(BASE_SYNTHETIC_DATA_DIR, 'l1_scaled')

L2_SCALED_SYNTHETIC_DATA_DIR = os.path.join(BASE_SYNTHETIC_DATA_DIR, 'l2_scaled')

SCALED_SYNTHETIC_DATA_DIR = os.path.join(BASE_SYNTHETIC_DATA_DIR, 'scaled')

RAW_SYNTHETIC_DATA_DIR = os.path.join(BASE_SYNTHETIC_DATA_DIR, 'raw')


EMBEDDINGS = {

    # "l1": {

    #     "resnet18": {
    #         "real_A": os.path.join(RAW_REAL_DATA_DIR, 'resnet18', 'NORMAL'),
    #         "synthetic_A": os.path.join(L1_SCALED_SYNTHETIC_DATA_DIR, 'resnet18', 'NORMAL'),
    #         "synthetic_B": os.path.join(L1_SCALED_SYNTHETIC_DATA_DIR, 'resnet18', 'PNEUMONIA'),
    #         "real_B": os.path.join(RAW_REAL_DATA_DIR, 'resnet18', 'PNEUMONIA')
    #     },

    #     "clip": {
    #         "real_A": os.path.join(RAW_REAL_DATA_DIR, 'clip', 'NORMAL'),
    #         "synthetic_A": os.path.join(L1_SCALED_SYNTHETIC_DATA_DIR, 'clip', 'NORMAL'),
    #         "synthetic_B": os.path.join(L1_SCALED_SYNTHETIC_DATA_DIR, 'clip', 'PNEUMONIA'),
    #         "real_B": os.path.join(RAW_REAL_DATA_DIR, 'clip', 'PNEUMONIA')
    #     },

    #     "dinov2": {
    #         "real_A": os.path.join(RAW_REAL_DATA_DIR, 'dinov2', 'NORMAL'),
    #         "synthetic_A": os.path.join(L1_SCALED_SYNTHETIC_DATA_DIR, 'dinov2', 'NORMAL'),
    #         "synthetic_B": os.path.join(L1_SCALED_SYNTHETIC_DATA_DIR, 'dinov2', 'PNEUMONIA'),
    #         "real_B": os.path.join(RAW_REAL_DATA_DIR, 'dinov2', 'PNEUMONIA')
    #     }

    # },

    # "l2": {
    #     "resnet18": {
    #         "real_A": os.path.join(RAW_REAL_DATA_DIR, 'resnet18', 'NORMAL'),
    #         "synthetic_A": os.path.join(L2_SCALED_SYNTHETIC_DATA_DIR, 'resnet18', 'NORMAL'),
    #         "synthetic_B": os.path.join(L2_SCALED_SYNTHETIC_DATA_DIR, 'resnet18', 'PNEUMONIA'),
    #         "real_B": os.path.join(RAW_REAL_DATA_DIR, 'resnet18', 'PNEUMONIA')
    #     },

    #     "clip": {
    #         "real_A": os.path.join(RAW_REAL_DATA_DIR, 'clip', 'NORMAL'),
    #         "synthetic_A": os.path.join(L2_SCALED_SYNTHETIC_DATA_DIR, 'clip', 'NORMAL'),
    #         "synthetic_B": os.path.join(L2_SCALED_SYNTHETIC_DATA_DIR, 'clip', 'PNEUMONIA'),
    #         "real_B": os.path.join(RAW_REAL_DATA_DIR, 'clip', 'PNEUMONIA')
    #     },

    #     "dinov2": {
    #         "real_A": os.path.join(RAW_REAL_DATA_DIR, 'dinov2', 'NORMAL'),
    #         "synthetic_A": os.path.join(L2_SCALED_SYNTHETIC_DATA_DIR, 'dinov2', 'NORMAL'),
    #         "synthetic_B": os.path.join(L2_SCALED_SYNTHETIC_DATA_DIR, 'dinov2', 'PNEUMONIA'),
    #         "real_B": os.path.join(RAW_REAL_DATA_DIR, 'dinov2', 'PNEUMONIA')
    #     }
    # },

    # "scaled": {
    #     "resnet18": {
    #         "real_A": os.path.join(RAW_REAL_DATA_DIR, 'resnet18', 'NORMAL'),
    #         "synthetic_A": os.path.join(SCALED_SYNTHETIC_DATA_DIR, 'resnet18', 'NORMAL'),
    #         "synthetic_B": os.path.join(SCALED_SYNTHETIC_DATA_DIR, 'resnet18', 'PNEUMONIA'),
    #         "real_B": os.path.join(RAW_REAL_DATA_DIR, 'resnet18', 'PNEUMONIA')
    #     },

    #     "clip": {
    #         "real_A": os.path.join(RAW_REAL_DATA_DIR, 'clip', 'NORMAL'),
    #         "synthetic_A": os.path.join(SCALED_SYNTHETIC_DATA_DIR, 'clip', 'NORMAL'),
    #         "synthetic_B": os.path.join(SCALED_SYNTHETIC_DATA_DIR, 'clip', 'PNEUMONIA'),
    #         "real_B": os.path.join(RAW_REAL_DATA_DIR, 'clip', 'PNEUMONIA')
    #     },

    #     "dinov2": {
    #         "real_A": os.path.join(RAW_REAL_DATA_DIR, 'dinov2', 'NORMAL'),
    #         "synthetic_A": os.path.join(SCALED_SYNTHETIC_DATA_DIR, 'dinov2', 'NORMAL'),
    #         "synthetic_B": os.path.join(SCALED_SYNTHETIC_DATA_DIR, 'dinov2', 'PNEUMONIA'),
    #         "real_B": os.path.join(RAW_REAL_DATA_DIR, 'dinov2', 'PNEUMONIA')
    #     }
    # },

    "raw": {
        "resnet18": {
            "real_A": os.path.join(RAW_REAL_DATA_DIR, 'resnet18', 'NORMAL'),
            "synthetic_A": os.path.join(RAW_SYNTHETIC_DATA_DIR, 'resnet18', 'NORMAL'),
            "synthetic_B": os.path.join(RAW_SYNTHETIC_DATA_DIR, 'resnet18', 'PNEUMONIA'),
            "real_B": os.path.join(RAW_REAL_DATA_DIR, 'resnet18', 'PNEUMONIA')
        },

        "clip": {
            "real_A": os.path.join(RAW_REAL_DATA_DIR, 'clip', 'NORMAL'),
            "synthetic_A": os.path.join(RAW_SYNTHETIC_DATA_DIR, 'clip', 'NORMAL'),
            "synthetic_B": os.path.join(RAW_SYNTHETIC_DATA_DIR, 'clip', 'PNEUMONIA'),
            "real_B": os.path.join(RAW_REAL_DATA_DIR, 'clip', 'PNEUMONIA')
        },

        "dinov2": {
            "real_A": os.path.join(RAW_REAL_DATA_DIR, 'dinov2', 'NORMAL'),
            "synthetic_A": os.path.join(RAW_SYNTHETIC_DATA_DIR, 'dinov2', 'NORMAL'),
            "synthetic_B": os.path.join(RAW_SYNTHETIC_DATA_DIR, 'dinov2', 'PNEUMONIA'),
            "real_B": os.path.join(RAW_REAL_DATA_DIR, 'dinov2', 'PNEUMONIA')
        }
    }
}