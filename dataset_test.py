import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.utils.dataset_util import RAW_IMG_SIZE, FreiHAND


def show_data(image_raw, keypoints):
    """
    Function to visualize data
    Input: torch.utils.data.Dataset
    """
    # n_cols = 4
    # n_rows = int(np.ceil(n_samples / n_cols))
    # plt.figure(figsize=[15, n_rows * 4])

    image = image_raw.numpy()
    image = np.moveaxis(image, 0, -1)
    # keypoints = sample["keypoints"].numpy()
    # keypoints = keypoints * RAW_IMG_SIZE

    # plt.subplot(n_rows, n_cols, i)
    plt.imshow(image)
    plt.scatter(keypoints[:, 0], keypoints[:, 1], c="k", alpha=0.5)

    #     plt.plot(
    #         keypoints[params["ids"], 0],
    #         keypoints[params["ids"], 1],
    #         params["color"],
    #     )
    plt.tight_layout()
    plt.show()


def main(data_path):
    print(data_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ds = FreiHAND(data_path, device)
    d = ds[46]
    vertices = d["vertices"] * RAW_IMG_SIZE
    # print(vertices)
    keypoints = d["keypoints"] * RAW_IMG_SIZE
    show_data(d["image_raw"], vertices)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=Path, default="./data/freihand/")
    args = parser.parse_args()
    main(args.data_path)
