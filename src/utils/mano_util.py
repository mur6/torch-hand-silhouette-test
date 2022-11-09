from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch

import mano


def make_random_mano_model(*, global_orient=None, transl=None):
    mano_model_path = "./models/MANO_RIGHT.pkl"
    n_comps = 45
    batch_size = 1

    rh_model = mano.load(
        model_path=mano_model_path, is_right=True, num_pca_comps=n_comps, batch_size=batch_size, flat_hand_mean=False
    )

    betas = torch.rand(batch_size, 10) * 0.0
    pose = torch.rand(batch_size, n_comps) * 0.0
    if global_orient is None:
        global_orient = torch.zeros((batch_size, 3))
    if transl is None:
        transl = torch.zeros((batch_size, 3))

    output = rh_model(
        betas=betas, global_orient=global_orient, hand_pose=pose, transl=transl, return_verts=True, return_tips=True
    )
    return rh_model, output


def get_mano_verts(*, global_orient=None, transl=None):
    rh_model, output = make_random_mano_model(global_orient=global_orient, transl=transl)
    coordinate_transform = torch.tensor([[-1, -1, 1]])
    # mesh_faces = torch.tensor(rh_model.faces.astype(int))
    verts = output.vertices[0] * coordinate_transform
    # faces = [mesh_faces for i in range(batch_size)]
    return verts


def show_2d_vertices(vertices):
    # print("vertices: ", vertices.shape)
    # if pred_vertices is not None:
    #     print("pred_vertices: ", pred_vertices.shape)

    fig = plt.figure()
    axs = fig.add_subplot()
    axs.scatter(vertices[:, 0], vertices[:, 1], c="k", alpha=0.1)
    # if pred_vertices is not None:
    #     axs[3].scatter(pred_vertices[:, 0], pred_vertices[:, 1], c="red", alpha=0.3)
    plt.tight_layout()
    plt.show()


def show_3d_plot(axs, points3d):
    # print(pred_v3d.shape, pred_v3d)
    points3d /= 164.0
    X, Y, Z = points3d[:, 0], points3d[:, 1], points3d[:, 2]
    axs.scatter(X, Y, Z, marker="o")
    max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max() * 0.5
    mid_x = (X.max() + X.min()) * 0.5
    mid_y = (Y.max() + Y.min()) * 0.5
    mid_z = (Z.max() + Z.min()) * 0.5
    axs.set_xlim(mid_x - max_range, mid_x + max_range)
    axs.set_ylim(mid_y - max_range, mid_y + max_range)
    axs.set_zlim(mid_z - max_range, mid_z + max_range)


def show_3d_plot_list(points3d_list, *, ncols=4, nrows=None, figsize=(12, 8)):
    num = len(points3d_list)
    if nrows is None:
        nrows = num // ncols
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize, subplot_kw=dict(projection="3d"))
    if isinstance(axs, Iterable):
        axs = axs.flatten()
    else:
        axs = (axs,)
    for ax, points3d in zip(axs, points3d_list):
        show_3d_plot(ax, points3d)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    verts = (get_mano_verts(),)
    show_3d_plot_list(verts, ncols=1)
