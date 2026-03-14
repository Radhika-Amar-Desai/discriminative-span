from asyncio import base_subprocess
import os

BASE_DIR = r"C:\Users\97433\Knowing_the_difference\data\datasets\skin_lesion"

# Dataset
DATASET_NAME = "skin_lesion"

TRAIN_PATH = os.path.join(BASE_DIR, 'final_real_data')
VAL_PATH = os.path.join(BASE_DIR, 'val_data')
TEST_PATH = os.path.join(BASE_DIR, 'test_data')

# Model
MODEL_NAME = "mobilenetv2"        # resnet18 | efficientnet_b0 | mobilenet_v2
PRETRAINED = True

# Optional custom checkpoint
CHECKPOINT_PATH = None

# Training
EPOCHS = 10
BATCH_SIZE = 32
LR = 1e-4