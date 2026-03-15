import os

DATASET_NAME = "skin_lesion"
BASE_DIR = r"C:\Users\97433\Knowing_the_difference\data\datasets\skin_lesion"
DATASET_PATH = os.path.join(BASE_DIR, "final_real_data")

OUTPUT_BASE_DIR = r"C:\Users\97433\Knowing_the_difference\data\embeddings\skin_lesion"

OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, "real_data", "raw")
BATCH_SIZE = 32
NUM_WORKERS = 4