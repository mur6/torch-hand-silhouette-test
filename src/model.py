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
    look_at_view_transform,
)
from pytorch3d.structures import Meshes
from torchvision import models

import mano


class Encoder_with_Shape(nn.Module):
    def __init__(self, num_pca_comps):
        super(Encoder_with_Shape, self).__init__()

        self.feature_extractor = models.resnet18(pretrained=True)
        self.feature_extractor.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        fc_in_features = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Sequential(nn.Linear(fc_in_features, fc_in_features), nn.ReLU())

        self.hand_pca_estimator = nn.Sequential(
            nn.Linear(fc_in_features, fc_in_features // 2),
            nn.ReLU(),
            nn.Linear(fc_in_features // 2, num_pca_comps),  # hand pose PCAs
        )
        # ###  pose = torch.rand(batch_size, n_comps) * 0.1

        self.rotation_estimator = nn.Sequential(
            nn.Linear(fc_in_features, fc_in_features // 2),
            nn.ReLU(),
            nn.Linear(fc_in_features // 2, fc_in_features // 4),
            nn.ReLU(),
            nn.Linear(fc_in_features // 4, 3),  # 3D global orientation
        )

        self.translation_estimator = nn.Sequential(
            nn.Linear(fc_in_features + 2, fc_in_features // 2),
            nn.ReLU(),
            nn.Linear(fc_in_features // 2, fc_in_features // 4),
            nn.ReLU(),
            nn.Linear(fc_in_features // 4, 3),  # 3D translation
        )

        self.hand_shape_estimator = nn.Sequential(
            nn.Linear(fc_in_features, fc_in_features // 2),
            nn.ReLU(),
            nn.Linear(fc_in_features // 2, 10),  # MANO shape parameters
        )
        # ### betas = torch.rand(batch_size, 10) * 0.1

    def forward(self, x, focal_lens):
        x = self.feature_extractor(x)
        hand_pca = self.hand_pca_estimator(x)
        global_orientation = self.rotation_estimator(x)
        translation = self.translation_estimator(torch.cat([x, focal_lens], -1))
        hand_shape = self.hand_shape_estimator(x)
        output = torch.cat([hand_pca, global_orientation, translation, hand_shape], -1)
        # output = torch.cat([global_orientation, translation], -1)
        return output


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


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ds = FreiHAND(Path("data/freihand"), device)
    d = ds[46]
    vertices = d["vertices"] * RAW_IMG_SIZE
    # print(vertices)
    keypoints = d["keypoints"] * RAW_IMG_SIZE
    show_data(d["image_raw"], vertices)
