# from model import HandSilhouetteNet3

import argparse
import os
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import models, transforms


def batch_align_w_scale(mtx1, mtx2):
    """Align the predicted entity in some optimality sense with the ground truth."""
    # center
    t1 = torch.mean(mtx1, dim=1, keepdim=True)
    t2 = torch.mean(mtx2, dim=1, keepdim=True)
    mtx1_t = mtx1 - t1
    mtx2_t = mtx2 - t2

    # scale
    s1 = torch.norm(mtx1_t, dim=(1, 2), keepdim=True) + 1e-8
    mtx1_t = mtx1_t / s1
    s2 = torch.norm(mtx2_t, dim=(1, 2), keepdim=True) + 1e-8
    mtx2_t = mtx2_t / s2

    # orthogonal procrustes alignment
    # u, w, vt = torch.linalg.svd(torch.bmm(torch.transpose(mtx2_t, 1, 2), mtx1_t).transpose(1, 2))
    # R = torch.bmm(u, vt)
    u, w, v = torch.svd(torch.bmm(mtx2_t.transpose(1, 2), mtx1_t).transpose(1, 2))
    R = torch.bmm(u, v.transpose(1, 2))
    s = w.sum(dim=1, keepdim=True)

    # apply trafos to the second matrix
    mtx2_t = torch.bmm(mtx2_t, R.transpose(1, 2)) * s.unsqueeze(1)
    mtx2_t = mtx2_t * s1 + t1
    return mtx2_t


def aligned_meshes_loss(meshes_gt, meshes_pred):
    meshes_pred_aligned = batch_align_w_scale(meshes_gt, meshes_pred)
    return nn.L1Loss()(meshes_pred_aligned, meshes_gt)


class ContourLoss(nn.Module):
    def __init__(self, device):
        super(ContourLoss, self).__init__()
        self.device = device

    def forward(self, outputs, dist_maps):
        # Binarize outputs [0.0, 1.0] -> {0., 1.}
        # outputs = (outputs >= 0.5).float()   # Thresholding is NOT differentiable
        outputs = 1 / (1 + torch.exp(-100 * (outputs - 0.5)))  # Differentiable binarization (approximation)
        mask = outputs < 0.5
        outputs = outputs * mask  # Zero out values above the threshold 0.5

        # Convert from (B x H x W) to (B x C x H x W)
        outputs = torch.unsqueeze(outputs, 1)

        # Apply Laplacian operator to grayscale images to find contours
        kernel = torch.tensor([[[[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]]]).to(self.device)

        contours = F.conv2d(outputs, kernel, padding=1)
        contours = torch.clamp(contours, min=0, max=255)

        # Convert from (B x C x H x W) back to (B x H x W)
        contours = torch.squeeze(contours, 1)

        # Compute the Chamfer distance between two images
        # Selecting indices is NOT differentiable -> use tanh(x) or 2 / (1 + e^(-100(x))) - 1 for differentiable thresholding
        # -> apply element-wise product between contours and distance maps
        contours = torch.tanh(contours)
        dist = contours * dist_maps  # element-wise product

        dist = dist.sum() / contours.shape[0]
        assert dist >= 0

        return dist


def criterion(contour_loss, mask, vertices, pred_mask, pred_vertices):
    print(f"mask: {mask.shape}")
    print(torch.max(mask), torch.min(mask))
    # print(f"vertices: {vertices.shape}")
    print(f"pred_mask: {pred_mask.shape}")
    print(torch.max(pred_mask), torch.min(pred_mask))
    # print(f"pred_vertices: {pred_vertices.shape}")
    loss1 = aligned_meshes_loss(vertices, pred_vertices)
    loss2 = 0.0001 * torch.sum((mask - pred_mask) ** 2)
    # loss2 = contour_loss(mask, pred_mask)
    print(loss1, loss2)
    return loss1 + loss2
