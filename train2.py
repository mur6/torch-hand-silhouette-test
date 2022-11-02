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
from src.utils.dataset_util import RAW_IMG_SIZE, FreiHAND, projectPoints
from src.utils.mano_util import make_random_mano_model
from src.utils.render import make_silhouette_phong_renderer


def show_images(image_raw, image, mask, vertices, pred_vertices):
    print("vertices: ", vertices.shape)
    if pred_vertices is not None:
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
        ("result", image_raw),
    )
    for index, (label, image) in zip(range(nrows * ncols), label_and_images):
        axs[index].set_title(label)
        print(label, image.shape)
        if image.shape[0] == 3:
            axs[index].imshow(image.permute(1, 2, 0).detach().numpy())
        else:
            axs[index].imshow(image.detach().numpy())
    axs[0].scatter(vertices[:, 0], vertices[:, 1], c="k", alpha=0.1)
    if pred_vertices is not None:
        axs[3].scatter(pred_vertices[:, 0], pred_vertices[:, 1], c="red", alpha=0.3)
    plt.tight_layout()
    plt.show()


def main(args):
    data = FreiHAND(args.data_path)[args.data_number]
    vertices = data["vertices"]
    k_matrix = data["K_matrix"]

    print("vertices: ", vertices.shape, vertices.dtype, vertices[0])
    print("vertices: ", vertices.mean())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    silhouette_renderer, phong_renderer = make_silhouette_phong_renderer(device)
    model = Model(device, renderer=silhouette_renderer)
    model.train()

    focal_lens = data["focal_len"].unsqueeze(0)
    pred = model(focal_lens)
    pred_vertices = pred["vertices"]
    print("pred vertices: ", pred_vertices.shape, pred_vertices.dtype, pred_vertices[0][0])
    print("pred vertices: ", pred_vertices.mean())
    optimizer = optim.Adam(model.parameters(), lr=0.4)

    loop = tqdm(range(args.num_epochs))
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

    # pred_v2d = projectPoints(pred_vertices.squeeze(0).detach().numpy(), k_matrix.numpy())
    # print("pred_v2d: ", pred_v2d.shape)
    # print(f"pred_v2d: min={pred_v2d.min()}, max={pred_v2d.max()}, mean={pred_v2d.mean()}")
    show_images(
        data["image_raw"],
        data["image"],
        data["mask"],
        vertices=data["vertices2d"] * RAW_IMG_SIZE,
        pred_vertices=None,
    )
    pred_v3d = pred_vertices.squeeze(0).detach().numpy()
    print(pred_v3d.shape, pred_v3d)
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(pred_v3d[:, 0], pred_v3d[:, 1], pred_v3d[:, 2], marker="o")
    plt.show()


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=Path, default="./data/freihand/")
    parser.add_argument("--data_number", type=int)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--init_lr", type=float, default=1e-4)
    parser.add_argument("--num_pcs", type=int, default=45, help="number of pose PCs (ex: 6, 45)")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    main(args)
