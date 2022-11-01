from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import models, transforms


def load_image(num):
    # image_name = "data/input_images/datasets/training/images/image_000032.jpg"
    # mask_name = "data/input_images/datasets/training/masks/image_000032.png"
    base_path = Path("data/freihand/evaluation/")
    image_path = base_path / "rgb" / f"{num:08d}.jpg"
    mask_path = base_path / "segmap" / f"{num:08d}.png"
    print(image_path)
    print(mask_path)
    image_name = str(image_path)
    mask_name = str(mask_path)
    mask = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)
    mask = np.where(mask == 0, 0, 1)
    # print(mask[0, 0])

    orig_image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
    image = 255 - orig_image

    if image.shape != (224, 224):
        image_ref = cv2.resize(image, (224, 224))
        image_ref = cv2.threshold(image_ref, 127, 1, cv2.THRESH_BINARY)[1]
    else:
        image_ref = cv2.threshold(image, 127, 1, cv2.THRESH_BINARY)[1]

    # Extract contour and compute distance transform
    contour = cv2.Laplacian(image_ref, -1)
    contour = cv2.threshold(contour, 0, 1, cv2.THRESH_BINARY_INV)[1]
    dist_map = cv2.distanceTransform(contour, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    dist_map = torch.tensor(dist_map)

    # image_ref = torch.tensor(image_ref, dtype=torch.int).unsqueeze(0)

    im = cv2.imread(image_name)
    im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im_rgb = torch.from_numpy(im_rgb.transpose(2, 0, 1)).clone()
    im_rgb = im_rgb.unsqueeze(0) / 255
    return im_rgb, torch.from_numpy(mask).unsqueeze(0), dist_map


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ds = FreiHAND(Path("data/freihand"), device)
    d = ds[46]
    vertices = d["vertices"] * RAW_IMG_SIZE
    # print(vertices)
    keypoints = d["keypoints"] * RAW_IMG_SIZE
    show_data(d["image_raw"], vertices)
