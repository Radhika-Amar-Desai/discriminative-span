import os

DATASET_NAME = "horses_and_zebra"
BASE_DIR = r"C:\Users\97433\Knowing_the_difference\data\datasets\horses_and_zebra"
DATASET_PATH = os.path.join(BASE_DIR, "real_data")

OUTPUT_BASE_DIR = r"C:\Users\97433\Knowing_the_difference\data\embeddings\horses_and_zebra"

OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, "real_data", "raw")
BATCH_SIZE = 32
NUM_WORKERS = 4