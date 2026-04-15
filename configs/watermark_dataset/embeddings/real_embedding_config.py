import os

DATASET_NAME = "watermark_dataset"
BASE_DIR = r"C:\Users\97433\Knowing_the_difference\data\datasets\watermark_dataset"
DATASET_PATH = os.path.join(BASE_DIR, "real_data")

OUTPUT_BASE_DIR = r"C:\Users\97433\Knowing_the_difference\data\embeddings\watermark_dataset"

OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, "real_data", "raw")
BATCH_SIZE = 32
NUM_WORKERS = 4