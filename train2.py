import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from src.loss import ContourLoss, criterion
from src.model import Model
from src.utils.data import get_dataset
from src.utils.dataset_util import RAW_IMG_SIZE, FreiHAND, show_data
from src.utils.image import load_image
from src.utils.render import make_silhouette_phong_renderer


def show_images(image_raw, image, mask, orig_grey_image, vertices):
    # image = image_raw.numpy()
    # image = np.moveaxis(image, 0, -1)

    # plt.imshow(image)
    # plt.scatter(vertices[:, 0], vertices[:, 1], c="k", alpha=0.5)
    # plt.tight_layout()

    nrows, ncols = 2, 3
    fig, axs = plt.subplots(nrows, ncols)
    axs = axs.flatten()
    label_and_images = (
        ("image raw", image_raw),
        ("image", image),
        # ("orig grey image", orig_grey_image),
        ("mask", mask),
    )
    for index, (label, image) in zip(range(nrows * ncols), label_and_images):
        axs[index].set_title(label)
        print(label, image.shape)
        if image.shape[0] == 3:
            axs[index].imshow(image.permute(1, 2, 0).detach().numpy())
        else:
            axs[index].imshow(image.detach().numpy())
    plt.show()


def main_2(args):
    d = FreiHAND(args.data_path)[46]
    vertices = d["vertices"] * RAW_IMG_SIZE
    # show_data(d["image_raw"], vertices)

    show_images(d["image_raw"], d["image"], d["mask"], vertices=vertices)


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    silhouette_renderer, phong_renderer = make_silhouette_phong_renderer(device)
    # distance = 0.55  # distance from camera to the object
    # elevation = 120.0  # angle of elevation in degrees
    # azimuth = 120.0  # No rotation so the camera is positioned on the +Z axis.
    # R, T = look_at_view_transform(distance, elevation, azimuth, device=device)
    # silhouette = silhouette_renderer(meshes_world=mesh, R=R, T=T)
    # image_ref = phong_renderer(meshes_world=mesh, R=R, T=T)
    # silhouette = silhouette.cpu().numpy()
    meshes = mesh.unsqueeze(0)
    print(f"meshes: {meshes.shape}")
    model = Model(
        device, renderer=silhouette_renderer
    )  # Model(device, meshes=meshes, renderer=silhouette_renderer, mask=mask)

    # optimizer = optim.Adam(model.parameters(), lr=0.05)
    # # optimizer = torch.optim.SGD(model.parameters(), lr=0.75, momentum=0.9)
    # criterion = IoULoss()
    # # criterion = torchvision.ops.distance_box_iou_loss
    # train_loss = 0.0
    # # Train Phase
    model.train()
    focal_lens = focal_len.unsqueeze(0)
    d = model(focal_lens)
    pred_silhouettes = d["silhouettes"]
    pred_vertices = d["vertices"]

    fig, axs = plt.subplots(2, 2)
    axs = axs.flatten()
    axs[0].set_title("im_rgb")
    axs[0].imshow(im_rgb[0].permute(1, 2, 0).detach().numpy())
    axs[1].set_title("mask")
    axs[1].imshow(mask[0].detach().numpy())
    axs[2].set_title("grey image")
    axs[2].imshow(orig_image)
    axs[3].set_title("pred silhouettes")
    axs[3].imshow(pred_silhouettes[0].detach().numpy())
    # axs[3].set_title("dist_map")
    # axs[3].imshow(dist_map.detach().numpy())
    # plt.show()
    print(pred_vertices.shape)
    contour_loss = ContourLoss(device=device)
    optimizer = optim.Adam(model.parameters(), lr=0.4)
    loop = tqdm(range(50))
    for epoch in loop:
        optimizer.zero_grad()
        d = model(focal_lens)
        pred_silhouettes = d["silhouettes"]
        pred_vertices = d["vertices"]
        loss = criterion(contour_loss, mask, mesh.unsqueeze(0), pred_silhouettes, pred_vertices)
        loss.backward()
        optimizer.step()
        print(f"[Epoch {epoch}] Training Loss: {loss}")


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
    main_2(args)
