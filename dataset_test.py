import argparse
from pathlib import Path

from src.utils.dataset_util import RAW_IMG_SIZE, FreiHAND, show_data


def main(data_path):
    d = FreiHAND(data_path)[46]
    vertices2d = d["vertices2d"] * RAW_IMG_SIZE
    keypoints2d = d["keypoints2d"] * RAW_IMG_SIZE
    show_data(d["image_raw"], vertices=vertices2d, keypoints=keypoints2d)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=Path, default="./data/freihand/")
    args = parser.parse_args()
    main(args.data_path)
