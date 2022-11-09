import random

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.utils.mano_util import get_mano_verts, show_3d_plot


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    # global_orient = (torch.FloatTensor((3.14, 3.14, 3.14)) / 2.0).unsqueeze_(0)

    delta = 3.14 / 6.0
    # N = 12
    nrows, ncols = 3, 4
    fig, axs = plt.subplots(nrows, ncols, figsize=(12, 8), subplot_kw=dict(projection="3d"))
    axs = axs.flatten()

    for i, ax in enumerate(axs):
        angle = delta * i
        print(f"angle: {angle} delta: {delta} index: {i}")
        global_orient = torch.FloatTensor((angle, 0, 0)).unsqueeze_(0)
        verts = get_mano_verts(global_orient=global_orient)
        show_3d_plot(ax, verts)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
