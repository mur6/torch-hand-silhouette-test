import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from src.loss import vertices_criterion
from src.model import Model
from src.utils.data import get_dataset
from src.utils.dataset_util import RAW_IMG_SIZE, FreiHAND
from src.utils.image import load_image
from src.utils.render import make_silhouette_phong_renderer


def show_images(image_raw, image, mask, vertices, pred_vertices):
    print("vertices: ", vertices.shape)
    print("pred_vertices: ", pred_vertices.shape)
    # image = image_raw.numpy()
    # image = np.moveaxis(image, 0, -1)

    # plt.imshow(image)
    # plt.scatter(vertices[:, 0], vertices[:, 1], c="k", alpha=0.5)
    nrows, ncols = 2, 2
    fig, axs = plt.subplots(nrows, ncols)
    axs = axs.flatten()
    label_and_images = (
        ("image raw", image_raw),
        ("image", image),
        ("mask", mask),
    )
    for index, (label, image) in zip(range(nrows * ncols), label_and_images):
        axs[index].set_title(label)
        print(label, image.shape)
        if image.shape[0] == 3:
            axs[index].imshow(image.permute(1, 2, 0).detach().numpy())
        else:
            axs[index].imshow(image.detach().numpy())
    axs[0].scatter(vertices[:, 0], vertices[:, 1], c="k", alpha=0.1)
    plt.tight_layout()
    plt.show()


def main_2(args):
    d = FreiHAND(args.data_path)[46]
    vertices = d["vertices"] * RAW_IMG_SIZE
    # show_data(d["image_raw"], vertices)
    orig_image, image, focal_len, image_ref, label, dist_map, mesh = get_dataset(args.data_path)[46]
    # print(focal_len)
    show_images(d["image_raw"], d["image"], d["mask"], vertices=vertices)
    # print(d["focal_len"])


def main(args):
    data = FreiHAND(args.data_path)[46]
    vertices = data["vertices"]

    print("vertices: ", vertices.shape, vertices.dtype, vertices[0])
    #

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    silhouette_renderer, phong_renderer = make_silhouette_phong_renderer(device)
    model = Model(device, renderer=silhouette_renderer)
    model.train()

    focal_lens = data["focal_len"].unsqueeze(0)
    pred = model(focal_lens)
    pred_vertices = pred["vertices"]
    print("pred vertices: ", pred_vertices.shape, pred_vertices.dtype, pred_vertices[0][0])
    optimizer = optim.Adam(model.parameters(), lr=0.4)

    loop = tqdm(range(50))
    for epoch in loop:
        optimizer.zero_grad()
        pred = model(focal_lens)
        pred_vertices = pred["vertices"] / 100.0
        loss = vertices_criterion(vertices.unsqueeze(0), pred_vertices)
        loss.backward()
        optimizer.step()
        tqdm.write(f"[Epoch {epoch}] Training Loss: {loss}")
    print("vertices: ", vertices.shape, vertices.dtype, vertices[0])
    print("pred vertices: ", pred_vertices.shape, pred_vertices.dtype, pred_vertices[0][0])
    show_images(
        data["image_raw"],
        data["image"],
        data["mask"],
        vertices=data["vertices2d"] * RAW_IMG_SIZE,
        pred_vertices=pred_vertices.squeeze(0),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=Path, default="./data/freihand/")
    parser.add_argument("--checkpoint_path", type=str, default="./checkpoint")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=150)
    parser.add_argument("--init_lr", type=float, default=1e-4)
    parser.add_argument("--num_pcs", type=int, default=45, help="number of pose PCs (ex: 6, 45)")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    main(args)
