from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# PyTorch3D rendering components
from pytorch3d.renderer import (
    BlendParams,
    MeshRasterizer,
    MeshRenderer,
    PerspectiveCameras,
    RasterizationSettings,
    SoftSilhouetteShader,
    TexturesVertex,
)

# PyTorch3D data structures
from pytorch3d.structures import Meshes
from torchvision import models

import mano
from mano.lbs import vertices2joints


class HandModel(nn.Module):
    mano_model_path = "./models/MANO_RIGHT.pkl"
    n_comps = 45

    def __init__(self, *, device, batch_size):
        super().__init__()
        self.device = device
        self.batch_size = batch_size
        self.rh_model = mano.load(
            model_path=self.mano_model_path,
            is_right=True,
            num_pca_comps=self.n_comps,
            batch_size=self.batch_size,
            flat_hand_mean=False,
        )

        betas = torch.rand(self.batch_size, 10) * 0.1
        pose = torch.rand(self.batch_size, self.n_comps) * 0.1
        self.betas = nn.Parameter(betas.to(self.device))
        self.pose = nn.Parameter(pose.to(self.device))

    def forward(self):
        angle = (3.14 / 6) * 3
        global_orient = torch.FloatTensor((angle, 0, 0)).expand(self.batch_size, -1)
        transl = torch.zeros((self.batch_size, 3))
        rh_output = self.rh_model(
            betas=self.betas,
            global_orient=global_orient,
            hand_pose=self.pose,
            transl=transl,
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
        return {
            "torch3d_meshes": torch3d_meshes,
            "vertices": rh_output.vertices,
            "joints": rh_output_joints,
        }


class RefineNet(nn.Module):
    def __init__(self, num_vertices):
        super(RefineNet, self).__init__()

        self.num_vertices = num_vertices

        self.net = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size=(7, 7)),
        )

        self.fc = nn.Linear(7 * 7 * 256, num_vertices * 3)

    def forward(self, x):
        x = self.net(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class HandModelWithResnet(nn.Module):
    mano_model_path = "./models/MANO_RIGHT.pkl"
    num_pca_comps = 45

    def __init__(self, *, device):
        super().__init__()
        self.device = device
        # MANO right hand template model
        self.rh_model = mano.load(
            model_path=self.mano_model_path,
            is_right=True,
            num_pca_comps=self.num_pca_comps,
            flat_hand_mean=True,
        )
        # RefineNet
        self.refine_net = RefineNet(num_vertices=778)

        self.feature_extractor = models.resnet34(pretrained=True)
        # self.feature_extractor.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        fc_in_features = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Sequential(nn.Linear(fc_in_features, fc_in_features), nn.ReLU())

        self.hand_pca_estimator = nn.Sequential(
            nn.Linear(fc_in_features, fc_in_features // 2),
            nn.ReLU(),
            nn.Linear(fc_in_features // 2, self.num_pca_comps),  # hand pose PCAs
        )

        self.hand_shape_estimator = nn.Sequential(
            nn.Linear(fc_in_features, fc_in_features // 2),
            nn.ReLU(),
            nn.Linear(fc_in_features // 2, 10),  # MANO shape parameters
        )
        # 3D global orientation
        self.global_orientation = nn.Sequential(
            nn.Linear(fc_in_features, fc_in_features // 2),
            nn.ReLU(),
            nn.Linear(fc_in_features // 2, fc_in_features // 4),
            nn.ReLU(),
            nn.Linear(fc_in_features // 4, 3),
        )
        # translation
        self.translation_estimator = nn.Sequential(
            nn.Linear(fc_in_features, fc_in_features // 2),
            nn.ReLU(),
            nn.Linear(fc_in_features // 2, fc_in_features // 4),
            nn.ReLU(),
            nn.Linear(fc_in_features // 4, 3),  # 3D translation
        )

        self.blend_params = BlendParams(sigma=1e-4, gamma=1e-4, background_color=(1.0, 1.0, 1.0))
        self.raster_settings = RasterizationSettings(
            image_size=224,
            blur_radius=0.0,
            faces_per_pixel=100,
        )

    def forward(self, image, focal_lens, mask_gt):
        x = self.feature_extractor(image)
        hand_pca_pose = self.hand_pca_estimator(x)
        hand_shape = self.hand_shape_estimator(x)
        global_orient = self.global_orientation(x)
        transl = self.translation_estimator(x)

        # angle = (3.14 / 6) * 3
        # global_orient = torch.FloatTensor((angle, 0, 0)).expand(self.batch_size, -1)
        # transl = torch.zeros((self.batch_size, 3))
        rh_output = self.rh_model(
            betas=hand_shape,
            global_orient=global_orient,
            hand_pose=hand_pca_pose,
            transl=transl,
            return_verts=True,
            return_tips=True,
        )

        # ################### pytorch3d: Start #############################
        # Initialize each vertex to be white in color
        verts_rgb = torch.ones_like(rh_output.vertices)  # (B, V, 3)
        textures = TexturesVertex(verts_features=verts_rgb.to(self.device))

        # Coordinate transformation from FreiHand to PyTorch3D for rendering
        coordinate_transform = torch.tensor([[-1, -1, 1]]).to(self.device)

        mesh_faces = torch.tensor(self.rh_model.faces.astype(int)).to(self.device)

        batch_size = image.shape[0]
        # Create a Meshes object
        hand_meshes = Meshes(
            verts=[rh_output.vertices[i] * coordinate_transform for i in range(batch_size)],
            faces=[mesh_faces for i in range(batch_size)],
            textures=textures,
        )

        # Render the meshes
        self.cameras = PerspectiveCameras(focal_length=focal_lens * 2.0 / 224, device=self.device)
        # Create a silhouette mesh renderer by composing a rasterizer and a shader
        self.silhouette_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=self.cameras, raster_settings=self.raster_settings),
            shader=SoftSilhouetteShader(blend_params=self.blend_params),
        )
        silhouettes = self.silhouette_renderer(meshes_world=hand_meshes)
        silhouettes = silhouettes[..., 3]
        # ################### pytorch3d: End #############################

        # ################### 村木が作成した独自の補正コード: Start ####################
        # centered_vertices = rh_output.vertices - rh_output.vertices.mean()
        # abs_min = torch.abs(centered_vertices.min())
        # max_val = torch.max(centered_vertices.max(), abs_min)
        # centered_vertices = centered_vertices / max_val
        # print("補正されたvertices: ", centered_vertices)
        # ################### 村木が作成した独自の補正コード: End ####################

        reorder = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
        output_joints = rh_output.joints[:, reorder, :]

        # ################### Refinement Start ####################
        diff_map = torch.cat((mask_gt.unsqueeze(1), silhouettes.unsqueeze(1)), dim=1)
        offset = self.refine_net(diff_map)
        offset = torch.clamp(offset, min=-50, max=50)
        offset = offset.view(-1, 778, 3)

        vertices = rh_output.vertices + offset

        refined_joints = vertices2joints(self.rh_model.J_regressor, vertices)
        refined_joints = self.rh_model.add_joints(vertices, refined_joints)[:, reorder, :]
        # ################### Refinement End ######################

        return {
            "vertices": rh_output.vertices,
            "refined_vertices": vertices,
            "joints": output_joints,
            "refined_joints": refined_joints,
            "code": torch.cat([hand_pca_pose, hand_shape], -1),
        }


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
