import random

import matplotlib.pyplot as plt
import numpy as np
import torch

import mano
from src.utils.mano_util import show_3d_plot


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    verts = get_mano_verts()
    show_3d_plot(verts)

    # global_orient = (torch.FloatTensor((3.14, 3.14, 3.14)) / 2.0).unsqueeze_(0)
    global_orient = torch.FloatTensor((0, 3.14 / 6.0, 0)).unsqueeze_(0)
    # print(f"global_orient: {global_orient}")

    transl = torch.FloatTensor((0.0, 1.0, 2.0)).unsqueeze_(0)
    # print(f"transl: {transl}")


if __name__ == "__main__":
    main()
