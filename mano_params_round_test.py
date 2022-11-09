import random

import numpy as np
import torch

from src.utils.mano_util import get_mano_verts, show_3d_plot_list


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    # global_orient = (torch.FloatTensor((3.14, 3.14, 3.14)) / 2.0).unsqueeze_(0)

    def iter_verts():
        delta = 3.14 / 6.0
        for i in range(12):
            angle = delta * i
            print(f"angle: {angle} delta: {delta} index: {i}")
            global_orient = torch.FloatTensor((angle, 0, 0)).unsqueeze_(0)
            yield get_mano_verts(global_orient=global_orient)

    show_3d_plot_list(list(iter_verts()))


if __name__ == "__main__":
    main()
