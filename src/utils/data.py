from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import FreiHandDataset_Estimated as FreiHandDataset


def get_dataset(data_path):
    # joints_anno_file = "training_xyz.json"
    # camera_Ks_file = "training_K.json"
    # data_split_file = "FreiHand_split_ids.json"
    # vertices_anno_file = "training_verts.json"
    joints_anno_file = "evaluation_xyz.json"
    camera_Ks_file = "evaluation_K.json"
    data_split_file = "FreiHand_split_ids.json"
    vertices_anno_file = "evaluation_verts.json"
    # print("Checkpoint Path: {}\n".format(args.checkpoint_path))

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
        data_path,
        joints_anno_file,
        camera_Ks_file,
        data_split_file,
        vertices_anno_file,
        split="train",
        transform=transform,
        augment=True,
    )
    return dataset_train


def get_dataloader_train(data_path, *, batch_size=1):
    dataset_train = get_dataset(data_path)
    dataloader_train = DataLoader(
        dataset=dataset_train, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True
    )
    orig_image, image, focal_len, image_ref, label, dist_map, mesh = dataset_train[46]
    print(f"im_rgb: {orig_image.shape}")
    print(f"focal_len: {focal_len}")
    print(f"mesh: {mesh.shape} {mesh.dtype}")
    return dataloader_train


if __name__ == "__main__":
    data_path = "../my-mask2hand/data/freihand/"
    dataset_train = get_dataset(data_path)
    print(f"count: {len(dataset_train)}")
