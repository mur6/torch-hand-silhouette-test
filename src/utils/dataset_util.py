import itertools
import json

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

RAW_IMG_SIZE = 224

MODEL_IMG_SIZE = 224
DATASET_MEANS = [0.3950, 0.4323, 0.2954]
DATASET_STDS = [0.1966, 0.1734, 0.1836]


def projectPoints(xyz, K):
    """
    Projects 3D coordinates into image space.
    Function taken from https://github.com/lmb-freiburg/freihand
    """
    xyz = np.array(xyz)
    K = np.array(K)
    uv = np.matmul(K, xyz.T).T
    return uv[:, :2] / uv[:, -1:]


class FreiHAND(Dataset):
    """
    Class to load FreiHAND dataset. Only training part is used here.
    Augmented images are not used, only raw - first 32,560 images
    Link to dataset:
    https://lmb.informatik.uni-freiburg.de/resources/datasets/FreihandDataset.en.html
    """

    def __init__(self, base_path, range_from=None, range_to=None):
        self.image_dir = base_path / "evaluation" / "rgb"
        self.mask_dir = base_path / "evaluation" / "segmap"
        # self.image_names = np.sort(os.listdir(self.image_dir))
        fn_K_matrix = base_path / "evaluation_K.json"
        with open(fn_K_matrix, "r") as f:
            self.K_matrix = np.array(json.load(f))[range_from:range_to].tolist()

        fn_anno = base_path / "evaluation_xyz.json"
        with open(fn_anno, "r") as f:
            self.anno = np.array(json.load(f))[range_from:range_to]

        with (base_path / "evaluation_verts.json").open() as fh:
            self.vertices = np.array(json.load(fh))[range_from:range_to]
            # self.vertices = (np.array(self.vertices)[split_ids] * 1000).tolist()

        self.image_names = list(sorted(self.image_dir.glob("*.jpg")))[range_from:range_to]
        # self.image_names = self.image_names[n_start:n_end]
        # print(self.image_names[:10])
        # self.K_matrix = self.K_matrix
        # self.anno = self.anno
        self.mask_names = list(sorted(self.mask_dir.glob("*.png")))[range_from:range_to]
        # print(self.mask_names[:10])
        self.image_raw_transform = transforms.ToTensor()
        self.image_transform = transforms.Compose(
            [
                transforms.Resize(MODEL_IMG_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=DATASET_MEANS, std=DATASET_STDS),
            ]
        )

    def __len__(self):
        return len(self.anno)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_raw = Image.open(str(image_name))
        image = self.image_transform(image_raw)
        image_raw = self.image_raw_transform(image_raw)

        mask_name = self.mask_names[idx]
        mask = cv2.imread(str(mask_name), cv2.IMREAD_GRAYSCALE)
        mask = np.where(mask == 0, 0, 1)

        # keypoints = torch.from_numpy(self.anno[idx])
        keypoints = torch.tensor(self.anno[idx], dtype=torch.float32)
        keypoints2d = projectPoints(self.anno[idx], self.K_matrix[idx])
        keypoints2d = torch.FloatTensor(keypoints2d / RAW_IMG_SIZE)
        # heatmaps = torch.from_numpy(np.float32(heatmaps))
        # print("center:", np.mean(self.vertices[idx], 0))
        # vertices = torch.from_numpy(self.vertices[idx])
        vertices = torch.tensor(self.vertices[idx], dtype=torch.float32)
        vertices2d = projectPoints(self.vertices[idx], self.K_matrix[idx])
        vertices2d = torch.from_numpy(vertices2d / RAW_IMG_SIZE)
        # heatmaps = torch.from_numpy(np.float32(heatmaps))
        focal_len = torch.tensor([self.K_matrix[idx][0][0], self.K_matrix[idx][1][1]])
        # print("K_matrix:", self.K_matrix[idx])
        # return {
        #     "image": image,
        #     "keypoints": keypoints,
        #     "keypoints2d": keypoints2d,
        #     "vertices": vertices,
        #     "vertices2d": vertices2d,
        #     "image_name": image_name,
        #     "image_raw": image_raw,
        #     "mask": torch.from_numpy(mask),
        #     "focal_len": focal_len,
        #     "K_matrix": torch.tensor(self.K_matrix[idx]),
        # }
        return image, image_raw, mask, vertices, keypoints, keypoints2d


def show_data(image_raw, *, vertices=None, keypoints=None):
    nrows, ncols = 1, 2
    fig, axs = plt.subplots(nrows, ncols)
    axs = axs.flatten()
    label_and_points = (
        ("vertices", vertices),
        ("keypoints", keypoints),
    )
    for index, (label, points) in zip(range(nrows * ncols), label_and_points):
        axs[index].set_title(label)
        axs[index].imshow(image_raw.permute(1, 2, 0).detach().numpy())
        # axs[index].imshow(image.points().numpy())
        axs[index].scatter(points[:, 0], points[:, 1], c="red", alpha=0.2)

    # image = image_raw.numpy()
    # image = np.moveaxis(image, 0, -1)
    # plt.imshow(image)
    # plt.scatter(keypoints[:, 0], keypoints[:, 1], c="k", alpha=0.5)

    plt.tight_layout()
    plt.show()
