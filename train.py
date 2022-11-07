import argparse
import pickle
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from pytorch3d.io import load_obj, load_objs_as_meshes
from pytorch3d.renderer import (
    DirectionalLights,
    FoVPerspectiveCameras,
    Materials,
    MeshRasterizer,
    MeshRenderer,
    PointLights,
    RasterizationSettings,
    SoftPhongShader,
    TexturesUV,
    TexturesVertex,
    look_at_view_transform,
)
from pytorch3d.structures import Meshes
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from tqdm import tqdm

from src.loss import keypoints_criterion, vertices_criterion
from src.model import HandModel, SimpleSilhouetteModel
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


def show_3d_plot(points3d):
    # print(pred_v3d.shape, pred_v3d)
    points3d /= 164.0
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    X, Y, Z = points3d[:, 0], points3d[:, 1], points3d[:, 2]
    ax.scatter(X, Y, Z, marker="o")
    max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max() * 0.5
    mid_x = (X.max() + X.min()) * 0.5
    mid_y = (Y.max() + Y.min()) * 0.5
    mid_z = (Z.max() + Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    plt.show()


def main(args):
    data = FreiHAND(args.data_path)[args.data_number]
    vertices = data["vertices"]
    keypoints = data["keypoints"]
    keypoints2d = data["keypoints2d"]
    camera_params = data["K_matrix"].unsqueeze(0)
    # print(camera_params)
    # mask = torch.tensor(data["mask"], dtype=torch.float32).unsqueeze(0)

    print("vertices: ", vertices.shape, vertices.dtype, vertices[0])
    print("keypoints: ", keypoints.shape, keypoints.dtype)
    print("keypoints2d: ", keypoints2d.shape, keypoints2d.dtype)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # silhouette_renderer, phong_renderer = make_silhouette_phong_renderer(device)
    hand_model = HandModel(device)
    hand_model.train()

    # focal_lens = data["focal_len"].unsqueeze(0)
    hand_pred_data = hand_model(camera_params)
    print("######################################")
    pred_vertices = hand_pred_data["vertices"]
    pred_joints = hand_pred_data["joints"]
    pred_2d_joints = hand_pred_data["joints2d"]
    print("pred vertices: ", pred_vertices.shape, pred_vertices.dtype, pred_vertices[0][0])
    print("pred joints: ", pred_joints.shape, pred_joints.dtype, pred_joints[0][0])
    print("pred 2d joint2: ", pred_2d_joints.shape, pred_2d_joints.dtype)
    print("######################################")

    optimizer = optim.Adam(hand_model.parameters(), lr=0.4)
    loop = tqdm(range(args.num_epochs))
    for epoch in loop:
        optimizer.zero_grad()
        hand_pred_data = hand_model(camera_params)
        pred_vertices = hand_pred_data["vertices"]
        pred_joints = hand_pred_data["joints"]
        pred_2d_joints = hand_pred_data["joints2d"]
        loss1 = vertices_criterion(vertices.unsqueeze(0), pred_vertices)
        loss2 = keypoints_criterion(labels=keypoints.unsqueeze(0), pred_joints=pred_joints)
        # loss3 = torch.sum((keypoints2d - pred_2d_joints) ** 2) * 1e-7
        loss = loss1 + loss2
        loss.backward()
        optimizer.step()
        tqdm.write(f"[Epoch {epoch}] Training Loss: {loss}")
    print("vertices: ", vertices.shape, vertices.dtype, vertices[0])
    print("pred vertices: ", pred_vertices.shape, pred_vertices.dtype, pred_vertices[0][0])

    # pred_v2d = projectPoints(pred_vertices.squeeze(0).detach().numpy(), k_matrix.numpy())
    # print("pred_v2d: ", pred_v2d.shape)
    # print(f"pred_v2d: min={pred_v2d.min()}, max={pred_v2d.max()}, mean={pred_v2d.mean()}")
    if args.visualize:
        show_images(
            data["image_raw"],
            data["image"],
            data["mask"],
            vertices=data["vertices2d"] * RAW_IMG_SIZE,
            pred_vertices=pred_vertices.squeeze(0).detach().numpy(),
        )
        pred_v3d = pred_vertices.squeeze(0).detach().numpy()
        show_3d_plot(pred_v3d)
        pred_joints = pred_joints.squeeze(0).detach().numpy()
        show_3d_plot(pred_joints)

    pred_meshes = hand_pred_data["torch3d_meshes"]
    pred_meshes = pred_meshes.detach()
    if args.save_mesh:
        print(pred_meshes)
        with open("torch3d_pred_meshes.pickle", "wb") as fh:
            pickle.dump(pred_meshes, fh)
        print("saved.")
        # plt.figure(figsize=(7, 7))
        # texture_image = mesh.textures.maps_padded()
        # plt.imshow(texture_image.squeeze().cpu().numpy())
        # plt.axis("off")
        # plt.figure(figsize=(7, 7))
        # texturesuv_image_matplotlib(mesh.textures, subsample=None)
        # plt.axis("off")
        # plt.show()


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
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--save_mesh", action="store_true")
    args = parser.parse_args()
    main(args)
