import json

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

    def __init__(self, base_path):
        self.image_dir = base_path / "evaluation" / "rgb"
        # self.image_names = np.sort(os.listdir(self.image_dir))
        fn_K_matrix = base_path / "evaluation_K.json"
        with open(fn_K_matrix, "r") as f:
            self.K_matrix = np.array(json.load(f))

        fn_anno = base_path / "evaluation_xyz.json"
        with open(fn_anno, "r") as f:
            self.anno = np.array(json.load(f))

        with (base_path / "evaluation_verts.json").open() as fh:
            self.vertices = np.array(json.load(fh))
            # self.vertices = (np.array(self.vertices)[split_ids] * 1000).tolist()

        self.image_names = list(sorted(self.image_dir.glob("*.jpg")))
        # self.image_names = self.image_names[n_start:n_end]
        # print(self.image_names[:10])
        # self.K_matrix = self.K_matrix
        # self.anno = self.anno

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

        keypoints = projectPoints(self.anno[idx], self.K_matrix[idx])
        keypoints = keypoints / RAW_IMG_SIZE
        keypoints = torch.from_numpy(keypoints)
        # heatmaps = torch.from_numpy(np.float32(heatmaps))
        print("center:", np.mean(self.vertices[idx], 0))
        vertices = projectPoints(self.vertices[idx], self.K_matrix[idx])
        vertices = vertices / RAW_IMG_SIZE
        vertices = torch.from_numpy(vertices)
        # heatmaps = torch.from_numpy(np.float32(heatmaps))

        return {
            "image": image,
            "keypoints": keypoints,
            "vertices": vertices,
            "image_name": image_name,
            "image_raw": image_raw,
        }


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
