import argparse
import pickle
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torchvision
from PIL import Image
from torchvision import transforms as transforms

from src.model import HandModelWithResnet
from src.utils.mano_util import make_random_mano_model, show_3d_plot_list


def get_images(image_dir):
    def _iter_pil_images():
        sample_images = sorted(list(image_dir.glob("*.jpeg")))
        for p in sample_images:
            yield Image.open(p)

    return tuple(_iter_pil_images())


def images_to_tensors(images):
    transform = transforms.ToTensor()
    return torch.stack([transform(im) for im in images], dim=0)


def main(args):
    print(f"model path: {args.model_path}")
    images = get_images(args.image_dir)
    # image = torchvision.transforms.functional.pil_to_tensor(img) / 255.0
    x = images_to_tensors(images)
    print(x.shape)
    if False:
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.imshow(x.permute(1, 2, 0))
        plt.show()
    # x.unsqueeze_(0)
    print(x.shape)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = HandModelWithResnet(device=device, batch_size=args.batch_size)
    model.to(device)

    # optimizer = optim.Adam(model.parameters(), lr=args.init_lr)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6)

    # checkpoint_file = "model.pth"
    # checkpoint_file_path = str(args.checkpoint_path / checkpoint_file)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.no_grad():
        outputs = model(x)
        vertices = outputs["vertices"]
        joints = outputs["joints"]
        print(vertices.shape)
        show_3d_plot_list((vertices[-1],), ncols=1)
        # optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        # start_epoch = checkpoint["epoch"] + 1
        # print("Start Epoch: {}\n".format(start_epoch))


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=Path)
    parser.add_argument("--image_dir", type=Path)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    main(args)
