import os

DATASET_NAME = "pneumonia_cxr"
BASE_DIR = r"C:\Users\97433\Knowing_the_difference\data\datasets\pneumonia_cxr"
DATASET_PATH = os.path.join(BASE_DIR, "real_data")

OUTPUT_BASE_DIR = r"C:\Users\97433\Knowing_the_difference\data\embeddings\pneumonia_cxr"

OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, "real_data", "raw")
BATCH_SIZE = 32
NUM_WORKERS = 4