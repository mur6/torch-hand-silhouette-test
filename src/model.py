from math import radians
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from pytorch3d.renderer import (
    BlendParams,
    MeshRasterizer,
    MeshRenderer,
    PerspectiveCameras,
    RasterizationSettings,
    SoftSilhouetteShader,
    TexturesVertex,
    look_at_rotation,
    look_at_view_transform,
    rotate_on_spot,
)
from pytorch3d.structures import Meshes
from pytorch3d.transforms import axis_angle_to_matrix
from torchvision import models

import mano


class HandModel(nn.Module):
    mano_model_path = "./models/MANO_RIGHT.pkl"
    n_comps = 45
    batch_size = 1

    def __init__(self, device):
        super().__init__()
        self.device = device
        self.rh_model = mano.load(
            model_path=self.mano_model_path,
            is_right=True,
            num_pca_comps=self.n_comps,
            batch_size=self.batch_size,
            flat_hand_mean=False,
        )

        betas = torch.rand(self.batch_size, 10) * 0.1
        pose = torch.rand(self.batch_size, self.n_comps) * 0.1
        global_orient = torch.rand(self.batch_size, 3)
        transl = torch.rand(self.batch_size, 3)
        self.betas = nn.Parameter(betas.to(self.device))
        self.pose = nn.Parameter(pose.to(self.device))
        self.global_orient = nn.Parameter(global_orient.to(self.device))
        self.transl = nn.Parameter(transl.to(self.device))
        camera_params = torch.rand(self.batch_size, 3)
        self.camera_params = nn.Parameter(camera_params.to(self.device))

    def forward(self):
        # Global orient & pose PCAs to 3D hand joints & reconstructed silhouette
        rh_output = self.rh_model(
            betas=self.betas,
            global_orient=self.global_orient,
            hand_pose=self.pose,
            transl=self.transl,
            return_verts=True,
            return_tips=True,
        )
        # alpha = 0.3050 / 9.8489

        ############################################
        verts_rgb = torch.ones_like(rh_output.vertices)  # (B, V, 3)
        textures = TexturesVertex(verts_features=verts_rgb.to(self.device))

        # Coordinate transformation from FreiHand to PyTorch3D for rendering
        # [FreiHand] +X: right, +Y: down, +Z: in
        # [PyTorch3D] +X: left, +Y: up, +Z: in
        coordinate_transform = torch.tensor([[-1, -1, 1]]).to(self.device)
        mesh_faces = torch.tensor(self.rh_model.faces.astype(int)).to(self.device)

        # Create a Meshes object
        batch_size = self.batch_size
        # print("rh_output.vertices: ", rh_output.vertices.shape, rh_output.vertices.dtype)
        centered_vertices = rh_output.vertices - rh_output.vertices.mean()
        abs_min = torch.abs(centered_vertices.min())
        max_val = torch.max(centered_vertices.max(), abs_min)
        centered_vertices = centered_vertices / max_val
        # print("補正されたvertices: ", centered_vertices)
        reorder = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
        rh_output_joints = rh_output.joints[:, reorder, :]

        torch3d_meshes = Meshes(
            verts=[centered_vertices[i] * coordinate_transform for i in range(batch_size)],
            faces=[mesh_faces for i in range(batch_size)],
            textures=textures,
        )
        # print(f"X: {rh_output_joints.shape}")
        # print(f"camera: {camera_params.shape}")
        # joints2d = projectPoints(rh_output_joints.squeeze(0), camera_params)
        joints2d = orthographic_projection(rh_output_joints, self.camera_params)
        return {
            "torch3d_meshes": torch3d_meshes,
            "vertices": rh_output.vertices,
            "joints": rh_output_joints,
            "joints2d": joints2d,
        }


def projectPoints(X, camera):
    """
    Projects 3D coordinates into image space.
    Function taken from https://github.com/lmb-freiburg/freihand
    """
    # print(f"x.shape: {X.shape}")
    X = X.permute(1, 0)  # torch.transpose(X, 0, 1)
    # print(f"x.shape: {X.shape}")
    uv = torch.matmul(camera, X).permute(1, 0)
    # print(f"uv.shape: {uv.shape}")
    ret = uv[:, :2] / uv[:, -1:]
    # print(f"ret.shape: {ret.shape}")
    # print(ret)
    return ret


def orthographic_projection(X, camera):
    """Perform orthographic projection of 3D points X using the camera parameters
    Args:
        X: size = [B, N, 3]
        camera: size = [B, 3]
    Returns:
        Projected 2D points -- size = [B, N, 2]
    """
    camera = camera.view(-1, 1, 3)
    X_trans = X[:, :, :2] + camera[:, :, 1:]
    shape = X_trans.shape
    X_2d = (camera[:, :, 0] * X_trans.view(shape[0], -1)).view(shape)
    return X_2d


class SimpleSilhouetteModel(nn.Module):
    def __init__(self, device, renderer, meshes):
        super().__init__()
        self.device = device
        self.renderer = renderer
        self.meshes = meshes
        # cam_pos = [1.25, 0.27, 0.89]
        # cam_pos = [-1.7861, -1.2037, -1.5210]
        cam_pos = [-1.4014, -0.8891, -1.6230]
        cam_pos = [-1.4870, -1.2655, -0.9307]

        self.camera_position = nn.Parameter(torch.from_numpy(np.array(cam_pos, dtype=np.float32)).to(device))

        # angles = torch.FloatTensor([0, 0, 1.7453])
        angles = torch.FloatTensor([0.1931, 0.0621, 2.2167])
        self.angles = nn.Parameter(angles)
        print("angles: ", self.angles)
        # torch.FloatTensor([0, 0, 1.7453])

    def forward(self):
        R = look_at_rotation(self.camera_position[None, :], device=self.device)  # (1, 3, 3)
        T = -torch.bmm(R.transpose(1, 2), self.camera_position[None, :, None])[:, :, 0]  # (1, 3)
        rotation = axis_angle_to_matrix(self.angles)
        R, T = rotate_on_spot(R, T, rotation)
        # Calculate the silhouette loss
        # loss = torch.sum((image[..., 3] - self.image_ref) ** 2)
        return self.renderer(meshes_world=self.meshes.clone(), R=R, T=T)


class Silhouette2Model(nn.Module):
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

        # Render the meshes
        R = look_at_rotation(self.camera_position[None, :], device=self.device)  # (1, 3, 3)
        T = -torch.bmm(R.transpose(1, 2), self.camera_position[None, :, None])[:, :, 0]
        image = self.renderer(meshes_world=self.meshes.clone(), R=R, T=T)

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
        alpha = 0.3050 / 9.8489
        result = {
            # "output_joints": output_joints,
            "silhouettes": silhouettes,
            "vertices": rh_output.vertices * alpha,
            # "refined_vertices": vertices,
        }
        return result


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ds = FreiHAND(Path("data/freihand"), device)
    d = ds[46]
    vertices = d["vertices"] * RAW_IMG_SIZE
    # print(vertices)
    keypoints = d["keypoints"] * RAW_IMG_SIZE
    show_data(d["image_raw"], vertices)
