import argparse
import pickle
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.loss import keypoints_2d_criterion, keypoints_criterion, vertices_criterion
from src.model import HandModel, HandModelWithResnet
from src.utils.data import get_dataset
from src.utils.dataset_util import RAW_IMG_SIZE, FreiHAND, projectPoints
from src.utils.mano_util import make_random_mano_model, show_3d_plot_list


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


def train_model(
    model,
    dataloader_train,
    dataloader_val,
    optimizer,
    device,
    num_epochs,
    start_epoch=0,
    scheduler=None,
    train_loss_list=[],
    val_loss_list=[],
    last_lr_list=[],
    checkpoint_path="./checkpoint",
):
    if start_epoch == 0:
        min_val_loss = sys.maxsize
    else:
        min_val_loss = min(val_loss_list)

    for epoch in range(start_epoch, num_epochs):
        train_loss, val_loss = 0, 0

        # Train Phase
        model.train()
        for image, image_raw, mask, vertices, keypoints, _ in dataloader_train:
            image = image.to(device)
            image_raw = image_raw.to(device)
            mask = mask.to(device)
            vertices = vertices.to(device)
            keypoints = keypoints.to(device)

            optimizer.zero_grad()
            outputs = model(image)
            pred_vertices = outputs["vertices"]
            pred_joints = outputs["joints"]
            loss1 = vertices_criterion(vertices, pred_vertices)
            loss2 = keypoints_criterion(labels=keypoints, pred_joints=pred_joints)
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * image.size(0)
        train_loss /= len(dataloader_train.dataset)
        train_loss_list.append(train_loss)

        # Validation Phase
        model.eval()
        with torch.no_grad():
            for image, image_raw, mask, vertices, keypoints, _ in dataloader_train:
                image = image.to(device)
                image_raw = image_raw.to(device)
                mask = mask.to(device)
                vertices = vertices.to(device)
                keypoints = keypoints.to(device)

                loss1 = vertices_criterion(vertices, pred_vertices)
                loss2 = keypoints_criterion(labels=keypoints, pred_joints=pred_joints)
                loss = loss1 + loss2

                val_loss += loss.item() * image.size(0)
            val_loss /= len(dataloader_val.dataset)
            val_loss_list.append(val_loss)

        # Learning rate scheduling
        if scheduler is not None:
            scheduler.step(val_loss)
            last_lr_list.append(scheduler._last_lr)

        # Save the loss values
        df_loss = pd.DataFrame({"train_loss": train_loss_list, "val_loss": val_loss_list, "last_lr": last_lr_list})
        df_loss.to_csv(str(checkpoint_path / "loss.csv"), index=False)

        # Save checkpoints
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
            },
            str(checkpoint_path / "model.pth"),
        )

        if (epoch + 1) % 10 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                },
                str(checkpoint_path / f"model_epoch{epoch}.pth"),
            )

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                },
                str(checkpoint_path / "model_best.pth"),
            )

        print(
            f"[Epoch {epoch}] Training Loss: {train_loss}, Validation Loss: {val_loss}, Last Learning Rate: {scheduler._last_lr}"
        )

    # model.eval()
    # if args.visualize:
    #     pred_vertices = hand_pred_data["vertices"]
    #     pred_joints = hand_pred_data["joints"]
    #     for i, (a_image, a_image_raw, a_mask, a_keypoints2d) in enumerate(zip(image, image_raw, mask, keypoints2d)):
    #         show_images(
    #             a_image_raw,
    #             a_image,
    #             a_mask,
    #             vertices=a_keypoints2d * RAW_IMG_SIZE,
    #             pred_vertices=None,
    #         )
    #         pred_v3d = pred_vertices[i].detach().numpy()
    #         pred_joints = pred_joints[i].detach().numpy()
    #         show_3d_plot_list((pred_v3d, pred_joints), ncols=2)

    # pred_meshes = hand_pred_data["torch3d_meshes"]
    # pred_meshes = pred_meshes.detach()
    # if args.save_mesh:
    #     print(pred_meshes)
    #     with open("torch3d_pred_meshes.pickle", "wb") as fh:
    #         pickle.dump(pred_meshes, fh)
    #     print("saved.")


def main(args):
    print("Checkpoint Path: {}\n".format(args.checkpoint_path))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    start_epoch = 0

    dataset_train = FreiHAND(args.data_path, range_from=0, range_to=3000)
    # print(dataset_train[0][-1])
    dataset_val = FreiHAND(args.data_path, range_from=3000, range_to=4000)
    # print(dataset_val[0][-1])

    dataloader_train = DataLoader(
        dataset=dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=16, pin_memory=True
    )
    dataloader_val = DataLoader(
        dataset=dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True
    )

    print("Number of samples in training dataset: ", len(dataset_train))
    print("Number of samples in validation dataset: ", len(dataset_val))

    # Create model, optimizer, and learning rate scheduler
    model = HandModelWithResnet(device=device, batch_size=args.batch_size)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.init_lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6)

    # Train model
    train_loss_list = []
    val_loss_list = []
    last_lr_list = []

    if args.resume:
        checkpoint_file = "model.pth"
        checkpoint_file_path = str(args.checkpoint_path / checkpoint_file)
        checkpoint = torch.load(checkpoint_file_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print("Start Epoch: {}\n".format(start_epoch))

        df_loss = pd.read_csv(str(args.checkpoint_path / "loss.csv"))
        train_loss_list = df_loss["train_loss"].tolist()
        val_loss_list = df_loss["val_loss"].tolist()
        last_lr_list = df_loss["last_lr"].tolist()

    train_model(
        model,
        dataloader_train,
        dataloader_val,
        optimizer,
        device,
        args.num_epochs,
        start_epoch,
        scheduler,
        train_loss_list,
        val_loss_list,
        last_lr_list,
        args.checkpoint_path,
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=Path, default="./data/freihand/")
    parser.add_argument("--checkpoint_path", type=Path, default="./checkpoints")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=150)
    parser.add_argument("--init_lr", type=float, default=1e-4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    main(args)
