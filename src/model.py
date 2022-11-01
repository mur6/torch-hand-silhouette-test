# from model import HandSilhouetteNet3

import argparse
import os
import random
import sys
from pathlib import Path

import cv2
import mano
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from dataset import FreiHandDataset_Estimated as FreiHandDataset
from torchvision import models, transforms


class Encoder_with_Shape(nn.Module):
    def __init__(self, num_pca_comps):
        super(Encoder_with_Shape, self).__init__()

        self.feature_extractor = models.resnet18(pretrained=True)
        self.feature_extractor.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        fc_in_features = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Sequential(nn.Linear(fc_in_features, fc_in_features), nn.ReLU())

        self.hand_pca_estimator = nn.Sequential(
            nn.Linear(fc_in_features, fc_in_features // 2),
            nn.ReLU(),
            nn.Linear(fc_in_features // 2, num_pca_comps),  # hand pose PCAs
        )
        # ###  pose = torch.rand(batch_size, n_comps) * 0.1

        self.rotation_estimator = nn.Sequential(
            nn.Linear(fc_in_features, fc_in_features // 2),
            nn.ReLU(),
            nn.Linear(fc_in_features // 2, fc_in_features // 4),
            nn.ReLU(),
            nn.Linear(fc_in_features // 4, 3),  # 3D global orientation
        )

        self.translation_estimator = nn.Sequential(
            nn.Linear(fc_in_features + 2, fc_in_features // 2),
            nn.ReLU(),
            nn.Linear(fc_in_features // 2, fc_in_features // 4),
            nn.ReLU(),
            nn.Linear(fc_in_features // 4, 3),  # 3D translation
        )

        self.hand_shape_estimator = nn.Sequential(
            nn.Linear(fc_in_features, fc_in_features // 2),
            nn.ReLU(),
            nn.Linear(fc_in_features // 2, 10),  # MANO shape parameters
        )
        # ### betas = torch.rand(batch_size, 10) * 0.1

    def forward(self, x, focal_lens):
        x = self.feature_extractor(x)
        hand_pca = self.hand_pca_estimator(x)
        global_orientation = self.rotation_estimator(x)
        translation = self.translation_estimator(torch.cat([x, focal_lens], -1))
        hand_shape = self.hand_shape_estimator(x)
        output = torch.cat([hand_pca, global_orientation, translation, hand_shape], -1)
        # output = torch.cat([global_orientation, translation], -1)
        return output
