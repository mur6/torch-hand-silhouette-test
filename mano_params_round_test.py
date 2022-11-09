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

    fig = plt.figure()
    delta = 3.14 / 6.0
    N = 12
    for i in range(N):
        angle = delta * i
        global_orient = torch.FloatTensor((0, angle, 0)).unsqueeze_(0)
        verts = get_mano_verts(global_orient=global_orient)
        # print(verts)
        ax = fig.add_subplot(N, 1, (i + 1), projection="3d")
        show_3d_plot(ax, verts)
    plt.show()


if __name__ == "__main__":
    main()
