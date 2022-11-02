import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.utils.dataset_util import RAW_IMG_SIZE, FreiHAND, show_data


def main(data_path):
    d = FreiHAND(data_path)[46]
    vertices = d["vertices"] * RAW_IMG_SIZE
    show_data(d["image_raw"], vertices)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=Path, default="./data/freihand/")
    args = parser.parse_args()
    main(args.data_path)
