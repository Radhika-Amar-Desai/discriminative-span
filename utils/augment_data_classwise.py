import os
import cv2
import numpy as np
import argparse


# ----------------------------
# Augmentations
# ----------------------------

def get_hflip_img(img):
    return cv2.flip(img, 1), "_hflip"


def get_vflip_img(img):
    return cv2.flip(img, 0), "_vflip"


def get_rot45_img(img):
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, 45, 1.0)

    cos = abs(M[0, 0])
    sin = abs(M[0, 1])

    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    rotated = cv2.warpAffine(img, M, (new_w, new_h))
    return rotated, "_rot45"


# ----------------------------
# Core logic
# ----------------------------

def obtain_aug_img(filepath):
    img = cv2.imread(filepath)
    if img is None:
        return [], []

    base_name, ext = os.path.splitext(os.path.basename(filepath))
    ext = ext.lstrip('.')  # remove dot

    aug_functions = [get_hflip_img, get_vflip_img, get_rot45_img]

    images, filenames = [], []

    for aug in aug_functions:
        aug_img, tag = aug(img)
        filename = f"{base_name}{tag}.{ext}"
        images.append(aug_img)
        filenames.append(filename)

    return images, filenames


def augment_fldr_data(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)

    for filename in os.listdir(src_dir):
        filepath = os.path.join(src_dir, filename)

        if not os.path.isfile(filepath):
            continue

        aug_images, aug_filenames = obtain_aug_img(filepath)

        for img, name in zip(aug_images, aug_filenames):
            out_path = os.path.join(dst_dir, name)
            cv2.imwrite(out_path, img)
            print(f"Saved: {out_path}")


def save_augmented_data(src_root, dst_root):
    os.makedirs(dst_root, exist_ok=True)
    for class_name in os.listdir(src_root):
        src_path = os.path.join(src_root, class_name)
        dst_path = os.path.join(dst_root, class_name)

        if not os.path.isdir(src_path):
            continue

        augment_fldr_data(src_path, dst_path)


# ----------------------------
# CLI
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="Image Augmentation CLI")
    parser.add_argument("--src", required=True, help="Source dataset folder")
    parser.add_argument("--dst", required=True, help="Destination folder")

    args = parser.parse_args()

    save_augmented_data(args.src, args.dst)


if __name__ == "__main__":
    main()