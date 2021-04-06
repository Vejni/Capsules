import matplotlib.pyplot as plt

from .imagewisemodels import ImageWiseModels

from .DynamicCaps.capsulelayers import DenseCapsule, PrimaryCapsule
from .DynamicCaps.capsulenet import caps_loss

import torch.nn.functional as F
from .VarCaps import layers
from .VarCaps import vb_routing 

from .EMCaps import capsules

from .datasets import MEANS, STD
from torch.utils.data import DataLoader, ConcatDataset
import torchvision.transforms as transforms
from tqdm import tqdm
import torchvision
import PIL

from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch
import copy
import time
import os

# For testing we want to get a whole image in patches
BATCH_SIZE = 12

class VariationalMixedCapsules(ImageWiseModels): 
    """
    Capsule Routing via Variational Bayes based on https://github.com/fabio-deep/Variational-Capsule-Routing
    """
 
    def __init__(self, input_size, classes, channels, output_size, patchwise_path, args):
        super(VariationalMixedCapsules, self).__init__(input_size, classes, channels, output_size, patchwise_path)
        print("Trained PatchWise Model ready to use:", self.patch_wise_model)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.output_size = output_size
        self.n_classes = args.classes
        self.routings = args.routings

        self.P = args.pose_dim
        self.PP = int(np.max([2, self.P*self.P]))
        self.A, self.B, self.C, self.D = args.arch

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels=input_size[0], out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            # Block 2
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Block 3
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=64, out_channels=channels, kernel_size=1, stride=1),
        )

        # Layer 1: Just a conventional Conv2D layer
        self.Conv_1 = nn.Conv2d(output_size[0], self.A, kernel_size=5, stride=2, bias=False)
        nn.init.kaiming_uniform_(self.Conv_1.weight)

        self.BN_1 = nn.BatchNorm2d(self.A)
        self.PrimaryCaps = layers.PrimaryCapsules2d(in_channels=self.A, out_caps=self.B,
            kernel_size=1, stride=1, pose_dim=self.P)

        self.ConvCaps_1 = layers.ConvCapsules2d(in_caps=self.B, out_caps=self.C,
            kernel_size=3, stride=2, pose_dim=self.P)

        self.ConvRouting_1 = vb_routing.VariationalBayesRouting2d(in_caps=self.B, out_caps=self.C,
            kernel_size=3, stride=2, pose_dim=self.P,
            cov='diag', iter=args.routings,
            alpha0=1., m0=torch.zeros(self.PP), kappa0=1.,
            Psi0=torch.eye(self.PP), nu0=self.PP+1)

        self.ConvCaps_2 = layers.ConvCapsules2d(in_caps=self.C, out_caps=self.D,
            kernel_size=3, stride=1, pose_dim=self.P)

        self.ConvRouting_2 = vb_routing.VariationalBayesRouting2d(in_caps=self.C, out_caps=self.D,
            kernel_size=3, stride=1, pose_dim=self.P,
            cov='diag', iter=args.routings,
            alpha0=1., m0=torch.zeros(self.PP), kappa0=1.,
            Psi0=torch.eye(self.PP), nu0=self.PP+1)

        self.ClassCaps = layers.ConvCapsules2d(in_caps=self.D, out_caps=self.n_classes,
            kernel_size=1, stride=1, pose_dim=self.P, share_W_ij=True, coor_add=True)

        self.ClassRouting = vb_routing.VariationalBayesRouting2d(in_caps=self.D, out_caps=self.n_classes,
            kernel_size=12, stride=1, pose_dim=self.P, # adjust final kernel_size K depending on input H/W, for H=W=32, K=4.
            cov='diag', iter=args.routings,
            alpha0=1., m0=torch.zeros(self.PP), kappa0=1.,
            Psi0=torch.eye(self.PP), nu0=self.PP+1, class_caps=True)
        self.to(self.device)

    def forward(self, x):
        x = self.features(x)
        # Out ← [?, A, F, F]
        x = F.relu(self.BN_1(self.Conv_1(x)))
        # Out ← a [?, B, F, F], v [?, B, P, P, F, F]
        a,v = self.PrimaryCaps(x)
        # Out ← a [?, B, 1, 1, 1, F, F, K, K], v [?, B, C, P*P, 1, F, F, K, K]
        a,v = self.ConvCaps_1(a, v, self.device)
        # Out ← a [?, C, F, F], v [?, C, P, P, F, F]
        a,v = self.ConvRouting_1(a, v, self.device)
        # Out ← a [?, C, 1, 1, 1, F, F, K, K], v [?, C, D, P*P, 1, F, F, K, K]
        a,v = self.ConvCaps_2(a, v, self.device)
        # Out ← a [?, D, F, F], v [?, D, P, P, F, F]
        a,v = self.ConvRouting_2(a, v, self.device)
        # Out ← a [?, D, 1, 1, 1, F, F, K, K], v [?, D, n_classes, P*P, 1, F, F, K, K]
        a,v = self.ClassCaps(a, v, self.device)
        # Out ← yhat [?, n_classes], v [?, n_classes, P, P]
        yhat, v = self.ClassRouting(a, v, self.device)
        return yhat
