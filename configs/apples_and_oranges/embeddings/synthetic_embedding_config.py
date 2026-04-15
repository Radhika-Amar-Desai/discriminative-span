import os

DATASET_NAME = "skin_lesion"
BASE_DIR = r"C:\Users\97433\Knowing_the_difference\data\datasets\apples_and_oranges"
DATASET_PATH = os.path.join(BASE_DIR, "synthetic_corrupted_data")

OUTPUT_BASE_DIR = r"C:\Users\97433\Knowing_the_difference\data\embeddings\apples_and_oranges"

OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, "synthetic_corrupted_data", "raw")
BATCH_SIZE = 32
NUM_WORKERS = 4