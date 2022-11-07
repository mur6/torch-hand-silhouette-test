import torch

import mano


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
    output.joints
    # faces = [mesh_faces for i in range(batch_size)]
    return verts


if __name__ == "__main__":
    verts = get_mano_verts()
    print(verts)
