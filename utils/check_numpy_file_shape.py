import numpy as np
import sys

def inspect_npy(file_path):
    try:
        data = np.load(file_path)
        print(f"Shape: {data.shape}")
        print(f"Dtype: {data.dtype}")
    except Exception as e:
        print(f"Error loading file: {e}")

if __name__ == "__main__":
    dir = r"C:\Users\97433\Knowing_the_difference\data\embeddings\pneumonia_cxr\synthetic_data\l2_scaled\resnet18\weights.npy"
    inspect_npy(dir)