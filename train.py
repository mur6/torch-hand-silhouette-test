import argparse
import pickle
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from pytorch3d.io import load_obj, load_objs_as_meshes
from tqdm import tqdm

from src.loss import keypoints_2d_criterion, keypoints_criterion, vertices_criterion
from src.model import HandModel, SimpleSilhouetteModel
from src.utils.data import get_dataset
from src.utils.dataset_util import RAW_IMG_SIZE, FreiHAND, projectPoints
from src.utils.mano_util import make_random_mano_model, show_3d_plot_list
from src.utils.render import make_silhouette_phong_renderer


def show_images(image_raw, image, mask, vertices, pred_vertices):
    # print("vertices: ", vertices.shape)
    # if pred_vertices is not None:
    #     print("pred_vertices: ", pred_vertices.shape)

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
    keypoints = data["keypoints"]
    keypoints2d = data["keypoints2d"]
    camera_params = data["K_matrix"]
    print(keypoints2d)
    # mask = torch.tensor(data["mask"], dtype=torch.float32).unsqueeze(0)

    print("vertices: ", vertices.shape, vertices.dtype, vertices[0])
    print("keypoints: ", keypoints.shape, keypoints.dtype)
    print("keypoints2d: ", keypoints2d.shape, keypoints2d.dtype)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # silhouette_renderer, phong_renderer = make_silhouette_phong_renderer(device)
    hand_model = HandModel(device)
    hand_model.train()

    # focal_lens = data["focal_len"].unsqueeze(0)
    hand_pred_data = hand_model()
    print("######################################")
    pred_vertices = hand_pred_data["vertices"]
    pred_joints = hand_pred_data["joints"]
    print("pred vertices: ", pred_vertices.shape, pred_vertices.dtype, pred_vertices[0][0])
    print("pred joints: ", pred_joints.shape, pred_joints.dtype, pred_joints[0][0])
    print("######################################")

    optimizer = optim.Adam(hand_model.parameters(), lr=0.4)
    loop = tqdm(range(args.num_epochs))
    for epoch in loop:
        optimizer.zero_grad()
        hand_pred_data = hand_model()
        pred_vertices = hand_pred_data["vertices"]
        pred_joints = hand_pred_data["joints"]
        loss1 = vertices_criterion(vertices.unsqueeze(0), pred_vertices)
        loss2 = keypoints_criterion(labels=keypoints.unsqueeze(0), pred_joints=pred_joints)
        loss = loss1 + loss2
        loss.backward()
        optimizer.step()
        tqdm.write(f"[Epoch {epoch}] Training Loss: {loss}")
    # print("vertices: ", vertices.shape, vertices.dtype, vertices[0])
    print("pred_vertices: ", pred_vertices.shape, pred_vertices.dtype, pred_vertices[0][0])

    # pred_v2d = projectPoints(pred_vertices.squeeze(0).detach().numpy(), k_matrix.numpy())
    # print("pred_v2d: ", pred_v2d.shape)
    # print(f"pred_v2d: min={pred_v2d.min()}, max={pred_v2d.max()}, mean={pred_v2d.mean()}")
    if args.visualize:
        show_images(
            data["image_raw"],
            data["image"],
            data["mask"],
            vertices=data["vertices2d"] * RAW_IMG_SIZE,
            pred_vertices=None,
        )
        pred_v3d = pred_vertices.squeeze(0).detach().numpy()
        pred_joints = pred_joints.squeeze(0).detach().numpy()

        show_3d_plot_list((pred_v3d, pred_joints), ncols=2)

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
