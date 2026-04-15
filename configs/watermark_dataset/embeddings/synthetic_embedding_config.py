import os

DATASET_NAME = "skin_lesion"
BASE_DIR = r"C:\Users\97433\Knowing_the_difference\data\datasets\watermark_dataset"
DATASET_PATH = os.path.join(BASE_DIR, "synthetic_data")

OUTPUT_BASE_DIR = r"C:\Users\97433\Knowing_the_difference\data\embeddings\watermark_dataset"

OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, "synthetic_data", "raw")
BATCH_SIZE = 32
NUM_WORKERS = 4