# from model import HandSilhouetteNet3

import argparse
import os
import random
import sys
from pathlib import Path

import cv2
import mano
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from dataset import FreiHandDataset_Estimated as FreiHandDataset
from loss import FocalLoss, IoULoss
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


class Model(nn.Module):
    mano_model_path = "./models/MANO_RIGHT.pkl"
    n_comps = 45
    batch_size = 1

    def __init__(self, device, renderer):
        super().__init__()
        self.device = device
        # self.meshes = meshes
        self.renderer = renderer
        self.rh_model = mano.load(
            model_path=self.mano_model_path,
            is_right=True,
            num_pca_comps=self.n_comps,
            batch_size=self.batch_size,
            flat_hand_mean=False,
        )

        # Get the silhouette of the reference RGB image by finding all non-white pixel values.
        # image_ref = torch.from_numpy((image_ref[..., :3].max(-1) != 1).astype(np.float32))
        # self.register_buffer("image_ref", image_ref)
        # self.register_buffer("mask", mask)

        # Create an optimizable parameter for the x, y, z position of the camera.
        self.camera_position = nn.Parameter(
            torch.from_numpy(np.array([3.0, 6.9, +2.5], dtype=np.float32)).to(self.device)
        )
        betas = torch.rand(self.batch_size, 10) * 0.1
        pose = torch.rand(self.batch_size, self.n_comps) * 0.1
        global_orient = torch.rand(self.batch_size, 3)
        transl = torch.rand(self.batch_size, 3)

        # output = rh_model(
        #     betas=betas, global_orient=global_orient, hand_pose=pose, transl=transl, return_verts=True, return_tips=True
        # )
        self.betas = nn.Parameter(betas.to(self.device))
        self.pose = nn.Parameter(pose.to(self.device))
        self.global_orient = nn.Parameter(global_orient.to(self.device))
        self.transl = nn.Parameter(transl.to(self.device))

        self.blend_params = BlendParams(sigma=1e-4, gamma=1e-4, background_color=(1.0, 1.0, 1.0))
        # Define the settings for rasterization and shading. Here we set the output image to be of size 256x256
        # To form the blended image we use 100 faces for each pixel. We also set bin_size and max_faces_per_bin to None which ensure that
        # the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for
        # explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of
        # the difference between naive and coarse-to-fine rasterization.
        self.raster_settings = RasterizationSettings(
            image_size=224,
            blur_radius=0.0,
            faces_per_pixel=100,
        )
        self.camera_position = nn.Parameter(
            # 3.0, 6.9, +2.5
            # -0.21, -0.5, +0.19
            # -0.15, -0.30, 0.39
            # -0.03, -0.30, 0.39
            # 0.0, 0.5, 0.5
            # 0.2590, -0.3444, 0.4757
            # 0.77, -0.4, 0.98
            torch.from_numpy(np.array([3.0, 6.9, +2.5], dtype=np.float32)).to(self.device)
        )

    def forward2(self):
        batch_size = 1
        rh_output = self.rh_model(
            betas=self.betas,
            global_orient=self.global_orient,
            hand_pose=self.pose,
            transl=self.transl,
            return_verts=True,
            return_tips=True,
        )
        verts_rgb = torch.ones_like(rh_output.vertices)  # (B, V, 3)
        textures = TexturesVertex(verts_features=verts_rgb.to(self.device))
        # Coordinate transformation from FreiHand to PyTorch3D for rendering
        # [FreiHand] +X: right, +Y: down, +Z: in
        # [PyTorch3D] +X: left, +Y: up, +Z: in
        coordinate_transform = torch.tensor([[-1, -1, 1]]).to(self.device)
        mesh_faces = torch.tensor(self.rh_model.faces.astype(int)).to(self.device)
        hand_meshes = Meshes(
            verts=[rh_output.vertices[i] * coordinate_transform for i in range(batch_size)],
            faces=[mesh_faces for i in range(batch_size)],
            textures=textures,
        )

        # output = self.rh_model(
        #     betas=self.betas,
        #     global_orient=self.global_orient,
        #     hand_pose=self.pose,
        #     transl=self.transl,
        #     return_verts=True,
        #     return_tips=True,
        # )
        # coordinate_transform = torch.tensor([[-1, -1, 1]])
        # mesh_faces = torch.tensor(self.rh_model.faces.astype(int))
        # # verts = output.vertices[0] * coordinate_transform
        # hand_meshes = Meshes(
        #     verts=[output.vertices[i] * coordinate_transform for i in range(batch_size)],
        #     faces=[mesh_faces for i in range(batch_size)],
        #     # verts=[rh_output.vertices[0]],
        #     # faces=[mesh_faces],
        #     # textures=textures,
        # )

        # Render the image using the updated camera position. Based on the new position of the
        # camera we calculate the rotation and translation matrices
        distance = 1.55  # distance from camera to the object
        elevation = 120.0  # angle of elevation in degrees
        azimuth = 120.0  # No rotation so the camera is positioned on the +Z axis.

        # Get the position of the camera based on the spherical angles
        R, T = look_at_view_transform(distance, elevation, azimuth, device=self.device)
        # R = look_at_rotation(self.camera_position[None, :], device=self.device)  # (1, 3, 3)
        # T = -torch.bmm(R.transpose(1, 2), self.camera_position[None, :, None])[:, :, 0]  # (1, 3)

        image = self.renderer(meshes_world=self.meshes.clone(), R=R, T=T)

        # Calculate the silhouette loss
        # loss = torch.sum((image[..., 3] - self.mask) ** 2)
        return hand_meshes, image

    def forward(self, focal_lens):
        # Initialize a perspective camera
        # fx = fx_screen * 2.0 / image_width
        # fy = fy_screen * 2.0 / image_height
        # px = - (px_screen - image_width / 2.0) * 2.0 / image_width
        # py = - (py_screen - image_height / 2.0) * 2.0 / image_height
        self.cameras = PerspectiveCameras(focal_length=focal_lens * 2.0 / 224, device=self.device)

        # Create a silhouette mesh renderer by composing a rasterizer and a shader
        self.silhouette_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=self.cameras, raster_settings=self.raster_settings),
            shader=SoftSilhouetteShader(blend_params=self.blend_params),
        )

        batch_size = 1

        # Global orient & pose PCAs to 3D hand joints & reconstructed silhouette
        rh_output = self.rh_model(
            betas=self.betas,
            global_orient=self.global_orient,
            hand_pose=self.pose,
            transl=self.transl,
            return_verts=True,
            return_tips=True,
        )

        # Initialize each vertex to be white in color
        verts_rgb = torch.ones_like(rh_output.vertices)  # (B, V, 3)
        textures = TexturesVertex(verts_features=verts_rgb.to(self.device))

        # Coordinate transformation from FreiHand to PyTorch3D for rendering
        # [FreiHand] +X: right, +Y: down, +Z: in
        # [PyTorch3D] +X: left, +Y: up, +Z: in
        coordinate_transform = torch.tensor([[-1, -1, 1]]).to(self.device)

        mesh_faces = torch.tensor(self.rh_model.faces.astype(int)).to(self.device)

        # Create a Meshes object
        hand_meshes = Meshes(
            verts=[rh_output.vertices[i] * coordinate_transform for i in range(batch_size)],
            faces=[mesh_faces for i in range(batch_size)],
            textures=textures,
        )

        distance = 1.55  # distance from camera to the object
        elevation = 0.0  # angle of elevation in degrees
        azimuth = 0.0  # No rotation so the camera is positioned on the +Z axis.
        # Get the position of the camera based on the spherical angles
        R, T = look_at_view_transform(distance, elevation, azimuth, device=self.device)
        # Render the meshes
        # R = look_at_rotation(self.camera_position[None, :], device=self.device)  # (1, 3, 3)
        # T = -torch.bmm(R.transpose(1, 2), self.camera_position[None, :, None])[:, :, 0]  # (1, 3)
        silhouettes = self.silhouette_renderer(meshes_world=hand_meshes, R=R, T=T)
        silhouettes = silhouettes[..., 3]

        # # Reorder the joints to match FreiHand annotations
        # reorder = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
        # output_joints = rh_output.joints[:, reorder, :]

        # #################### Refinement Start ####################
        # mask_gt = self.mask
        # diff_map = torch.cat((mask_gt.unsqueeze(1), silhouettes.unsqueeze(1)), dim=1)
        # vertices = rh_output.vertices
        # #################### Refinement End ######################

        result = {
            # "output_joints": output_joints,
            "silhouettes": silhouettes,
            "vertices": rh_output.vertices,
            # "refined_vertices": vertices,
        }
        return result


def load_image(num):
    # image_name = "data/input_images/datasets/training/images/image_000032.jpg"
    # mask_name = "data/input_images/datasets/training/masks/image_000032.png"
    base_path = Path("data/freihand/evaluation/")
    image_path = base_path / "rgb" / f"{num:08d}.jpg"
    mask_path = base_path / "segmap" / f"{num:08d}.png"
    print(image_path)
    print(mask_path)
    image_name = str(image_path)
    mask_name = str(mask_path)
    mask = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)
    mask = np.where(mask == 0, 0, 1)
    # print(mask[0, 0])

    orig_image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
    image = 255 - orig_image

    if image.shape != (224, 224):
        image_ref = cv2.resize(image, (224, 224))
        image_ref = cv2.threshold(image_ref, 127, 1, cv2.THRESH_BINARY)[1]
    else:
        image_ref = cv2.threshold(image, 127, 1, cv2.THRESH_BINARY)[1]

    # Extract contour and compute distance transform
    contour = cv2.Laplacian(image_ref, -1)
    contour = cv2.threshold(contour, 0, 1, cv2.THRESH_BINARY_INV)[1]
    dist_map = cv2.distanceTransform(contour, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    dist_map = torch.tensor(dist_map)

    # image_ref = torch.tensor(image_ref, dtype=torch.int).unsqueeze(0)

    im = cv2.imread(image_name)
    im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im_rgb = torch.from_numpy(im_rgb.transpose(2, 0, 1)).clone()
    im_rgb = im_rgb.unsqueeze(0) / 255
    return im_rgb, torch.from_numpy(mask).unsqueeze(0), dist_map


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


def batch_align_w_scale(mtx1, mtx2):
    """Align the predicted entity in some optimality sense with the ground truth."""
    # center
    t1 = torch.mean(mtx1, dim=1, keepdim=True)
    t2 = torch.mean(mtx2, dim=1, keepdim=True)
    mtx1_t = mtx1 - t1
    mtx2_t = mtx2 - t2

    # scale
    s1 = torch.norm(mtx1_t, dim=(1, 2), keepdim=True) + 1e-8
    mtx1_t = mtx1_t / s1
    s2 = torch.norm(mtx2_t, dim=(1, 2), keepdim=True) + 1e-8
    mtx2_t = mtx2_t / s2

    # orthogonal procrustes alignment
    # u, w, vt = torch.linalg.svd(torch.bmm(torch.transpose(mtx2_t, 1, 2), mtx1_t).transpose(1, 2))
    # R = torch.bmm(u, vt)
    u, w, v = torch.svd(torch.bmm(mtx2_t.transpose(1, 2), mtx1_t).transpose(1, 2))
    R = torch.bmm(u, v.transpose(1, 2))
    s = w.sum(dim=1, keepdim=True)

    # apply trafos to the second matrix
    mtx2_t = torch.bmm(mtx2_t, R.transpose(1, 2)) * s.unsqueeze(1)
    mtx2_t = mtx2_t * s1 + t1
    return mtx2_t


def aligned_meshes_loss(meshes_gt, meshes_pred):
    meshes_pred_aligned = batch_align_w_scale(meshes_gt, meshes_pred)
    return nn.L1Loss()(meshes_pred_aligned, meshes_gt)


class ContourLoss(nn.Module):
    def __init__(self, device):
        super(ContourLoss, self).__init__()
        self.device = device

    def forward(self, outputs, dist_maps):
        # Binarize outputs [0.0, 1.0] -> {0., 1.}
        # outputs = (outputs >= 0.5).float()   # Thresholding is NOT differentiable
        outputs = 1 / (1 + torch.exp(-100 * (outputs - 0.5)))  # Differentiable binarization (approximation)
        mask = outputs < 0.5
        outputs = outputs * mask  # Zero out values above the threshold 0.5

        # Convert from (B x H x W) to (B x C x H x W)
        outputs = torch.unsqueeze(outputs, 1)

        # Apply Laplacian operator to grayscale images to find contours
        kernel = torch.tensor([[[[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]]]).to(self.device)

        contours = F.conv2d(outputs, kernel, padding=1)
        contours = torch.clamp(contours, min=0, max=255)

        # Convert from (B x C x H x W) back to (B x H x W)
        contours = torch.squeeze(contours, 1)

        # Compute the Chamfer distance between two images
        # Selecting indices is NOT differentiable -> use tanh(x) or 2 / (1 + e^(-100(x))) - 1 for differentiable thresholding
        # -> apply element-wise product between contours and distance maps
        contours = torch.tanh(contours)
        dist = contours * dist_maps  # element-wise product

        dist = dist.sum() / contours.shape[0]
        assert dist >= 0

        return dist


def criterion(contour_loss, mask, vertices, pred_mask, pred_vertices):
    print(f"mask: {mask.shape}")
    print(torch.max(mask), torch.min(mask))
    # print(f"vertices: {vertices.shape}")
    print(f"pred_mask: {pred_mask.shape}")
    print(torch.max(pred_mask), torch.min(pred_mask))
    # print(f"pred_vertices: {pred_vertices.shape}")
    loss1 = aligned_meshes_loss(vertices, pred_vertices)
    loss2 = 0.0001 * torch.sum((mask - pred_mask) ** 2)
    # loss2 = contour_loss(mask, pred_mask)
    print(loss1, loss2)
    return loss1 + loss2


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
