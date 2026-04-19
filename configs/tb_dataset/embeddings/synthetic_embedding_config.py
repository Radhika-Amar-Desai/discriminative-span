import os

DATASET_NAME = "tb_dataset"
BASE_DIR = r"C:\Users\97433\Knowing_the_difference\data\datasets\tb_dataset"
DATASET_PATH = os.path.join(BASE_DIR, "synthetic_data")

OUTPUT_BASE_DIR = r"C:\Users\97433\Knowing_the_difference\data\embeddings\tb_dataset"

OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, "synthetic_data", "raw")
BATCH_SIZE = 32
NUM_WORKERS = 4