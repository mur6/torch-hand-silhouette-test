# from model import HandSilhouetteNet3

import argparse
import os
import random
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from pytorch3d.io import load_obj, load_objs_as_meshes
from pytorch3d.renderer import (
    BlendParams,
    DirectionalLights,
    FoVPerspectiveCameras,
    HardPhongShader,
    Materials,
    MeshRasterizer,
    MeshRenderer,
    PerspectiveCameras,
    PointLights,
    RasterizationSettings,
    SoftPhongShader,
    SoftSilhouetteShader,
    TexturesUV,
    TexturesVertex,
    look_at_rotation,
    look_at_view_transform,
)
from pytorch3d.structures import Meshes

# from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from torch.utils.data import DataLoader
from torchvision import models, transforms
from tqdm import tqdm

import mano
from dataset import FreiHandDataset_Estimated as FreiHandDataset

# from loss import FocalLoss, IoULoss
from src.loss import criterion


def make_silhouette_phong_renderer(device, image_size=224):
    # Initialize a perspective camera.
    cameras = FoVPerspectiveCameras(device=device)

    # To blend the 100 faces we set a few parameters which control the opacity and the sharpness of
    # edges. Refer to blending.py for more details.
    blend_params = BlendParams(sigma=1e-4, gamma=1e-4)

    # Define the settings for rasterization and shading. Here we set the output image to be of size
    # 256x256. To form the blended image we use 100 faces for each pixel. We also set bin_size and max_faces_per_bin to None which ensure that
    # the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for
    # explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of
    # the difference between naive and coarse-to-fine rasterization.
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=np.log(1.0 / 1e-4 - 1.0) * blend_params.sigma,
        faces_per_pixel=100,
    )

    # Create a silhouette mesh renderer by composing a rasterizer and a shader.
    silhouette_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftSilhouetteShader(blend_params=blend_params),
    )

    # We will also create a Phong renderer. This is simpler and only needs to render one face per pixel.
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    # We can add a point light in front of the object.
    lights = PointLights(device=device, location=((2.0, 2.0, -2.0),))
    phong_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=HardPhongShader(device=device, cameras=cameras, lights=lights),
    )
    return silhouette_renderer, phong_renderer


def main(args):
    # joints_anno_file = "training_xyz.json"
    # camera_Ks_file = "training_K.json"
    # data_split_file = "FreiHand_split_ids.json"
    # vertices_anno_file = "training_verts.json"
    joints_anno_file = "evaluation_xyz.json"
    camera_Ks_file = "evaluation_K.json"
    data_split_file = "FreiHand_split_ids.json"
    vertices_anno_file = "evaluation_verts.json"
    print("Checkpoint Path: {}\n".format(args.checkpoint_path))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # start_epoch = 0
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
            transforms.Normalize(mean=[0.8705], std=[0.3358]),
        ]
    )
    dataset_train = FreiHandDataset(
        args.data_path,
        joints_anno_file,
        camera_Ks_file,
        data_split_file,
        vertices_anno_file,
        split="train",
        transform=transform,
        augment=True,
    )
    dataloader_train = DataLoader(
        dataset=dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=16, pin_memory=True
    )
    print(f"count: {len(dataset_train)}")
    # model = HandSilhouetteNet3(mano_model_path="./models/MANO_RIGHT.pkl", num_pca_comps=args.num_pcs, device=device)
    # model.to(device)
    # optimizer = optim.Adam(model.parameters(), lr=args.init_lr)
    # a = dataset_train[0]
    orig_image, image, focal_len, image_ref, label, dist_map, mesh = dataset_train[46]
    print(f"im_rgb: {orig_image.shape}")
    print(f"focal_len: {focal_len}")
    print(f"mesh: {mesh.shape} {mesh.dtype}")
    im_rgb, mask, dist_map = load_image(46)
    print(f"im_rgb: {im_rgb.shape} mask: {mask.shape}, dist_map: {dist_map.shape}")
    # def visualize(image, silhouettes, mask):

    # joints = label
    # vertices = mesh

    # print(joints.shape)
    # print(vertices.shape)
    # print(vertices[0])
    # vertices2 = get_mano_verts()
    # print(vertices2.shape)
    # print(vertices2[0])

    #   def train():
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
    if False:
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
        plt.show()
        return
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
    if True:
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
        plt.show()
    # for epoch in range(12):
    #     optimizer.zero_grad()
    #     outputs = model()
    #     silhouettes = outputs["silhouettes"]
    #     # print(f"mask_float: dtype={mask_float.dtype} shape={mask_float.shape}")
    #     # print(f"silhouettes: dtype={silhouettes.dtype} shape={silhouettes.shape}")

    #     loss = criterion(silhouettes, mask_float)
    #     train_loss = loss.item()
    #     loss.backward()
    #     optimizer.step()


def make_random_mano_model():
    mano_model_path = "./models/MANO_RIGHT.pkl"
    n_comps = 45
    batch_size = 1

    rh_model = mano.load(
        model_path=mano_model_path, is_right=True, num_pca_comps=n_comps, batch_size=batch_size, flat_hand_mean=False
    )

    betas = torch.rand(batch_size, 10) * 0.1
    pose = torch.rand(batch_size, n_comps) * 0.1
    global_orient = torch.rand(batch_size, 3)
    transl = torch.rand(batch_size, 3)

    output = rh_model(
        betas=betas, global_orient=global_orient, hand_pose=pose, transl=transl, return_verts=True, return_tips=True
    )
    return rh_model, output


def get_mano_verts():
    rh_model, output = make_random_mano_model()
    coordinate_transform = torch.tensor([[-1, -1, 1]])
    # mesh_faces = torch.tensor(rh_model.faces.astype(int))
    verts = output.vertices[0] * coordinate_transform
    # faces = [mesh_faces for i in range(batch_size)]
    return verts


if __name__ == "__main__":
    # Set the seed for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    # args
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data/freihand/")
    parser.add_argument("--checkpoint_path", type=str, default="./checkpoint")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=150)
    parser.add_argument("--init_lr", type=float, default=1e-4)
    parser.add_argument("--num_pcs", type=int, default=45, help="number of pose PCs (ex: 6, 45)")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    main(args)
