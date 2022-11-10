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

from src.model import HandModelWithResnet


def main(args):
    print(f"model path: {args.model_path}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = HandModelWithResnet(device=device, batch_size=args.batch_size)
    model.to(device)

    # optimizer = optim.Adam(model.parameters(), lr=args.init_lr)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6)

    # checkpoint_file = "model.pth"
    # checkpoint_file_path = str(args.checkpoint_path / checkpoint_file)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

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
    parser.add_argument("--checkpoint_path", type=Path, default="./checkpoints")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=150)
    parser.add_argument("--init_lr", type=float, default=1e-4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    main(args)
