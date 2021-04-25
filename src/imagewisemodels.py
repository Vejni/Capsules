# Base Models
from .patchwisemodel import PatchWiseModel
from .model import Model

# Dynamic
from .DynamicCaps.capsulelayers import DenseCapsule, PrimaryCapsule
from .DynamicCaps.capsulenet import caps_loss

# Varcaps
from .VarCaps import layers
from .VarCaps import vb_routing 

# SR Capsules
from .SRCaps.modules import SelfRouting2d

# Data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from .datasets import MEANS, STD
import torchvision

# Training
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch


class ImageWiseModels(Model):
    """
    Base Image-Wise model, variants inherit from this class
    it assigns a hopefully trained patchwise net to the model to use for forward passes
    """
    def __init__(self, args, patchwise, original_architecture):
        super(ImageWiseModels, self).__init__(args)

        if patchwise is not None:
            self.patch_wise_model = patchwise
        else:
            print("Creating untrained patchwise model")
            self.patch_wise_model = PatchWiseModel(args, original_architecture=original_architecture)

    def propagate(self, inputs, labels, criterion=None):
        inputs = self.patch_wise_model.features(inputs)
        outputs = self(inputs)
        if criterion:
            loss = criterion(outputs, labels)
        else:
            loss = 0
        return loss, outputs

class BaseCNN(ImageWiseModels):
    """ Simpler CNN baseline than Nazeri """
    def __init__(self, args, patchwise=None, original_architecture=False):
        super(BaseCNN, self).__init__(args, patchwise, original_architecture)

        self.cnn_layers = nn.Sequential(
            # Convolutional Layer 1
            nn.Conv2d(args.input_size[0], 32, kernel_size=5, stride=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 1),

            # Convolutional Layer 2
            nn.Conv2d(32, 32, kernel_size=5, stride=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 1),

            # Convolutional Layer 3
            nn.Conv2d(32, 16, kernel_size=5, stride=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 1),
        )

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        self.set_linear_layer(args)

        # This will send net to device, so only call here
        self.init_device()

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x
    
    def set_linear_layer(self, args):
        """ Pytorch has no nice way of connecting CNNs with the denselayers, this code sets the dimensions correctly on the fly """
        train_data = torchvision.datasets.ImageFolder(root=args.data_path + "/train", transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=MEANS, std=STD)
        ]))
        train_data_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,  num_workers=args.workers)
        super(BaseCNN, self).train()  # Set model to training mode
        data, _ = next(iter(train_data_loader))
        #data = data[0]
        x = data.to(self.device)
        x = self.patch_wise_model.features(x)
        x = self.cnn_layers(x)

        print("Setting Linear Layer with shape:", x.shape)
        self.linear_layers = nn.Sequential(
            nn.Linear(x.size(1)*x.size(2)*x.size(3), 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            #nn.Dropout(0.5),
            nn.Linear(32, args.classes) # NO softmax, bc it is in crossentropy loss
        )
        self.to(self.device)

class NazeriCNN(ImageWiseModels):
    """ CNN imagewise network as in https://arxiv.org/abs/1803.04054 """
    def __init__(self, args, patchwise=None, original_architecture=False):
        super(NazeriCNN, self).__init__(args, patchwise, original_architecture)

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels=args.input_size[0], out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(1 * 16 * 16, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(64, args.classes),
        )

        # This will send net to device, so only call here
        self.init_device()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = F.log_softmax(x, dim=1)
        return x

    def initialize_weights(self):
        """ As in https://arxiv.org/abs/1803.04054 """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

class DynamicCapsules(ImageWiseModels): 
    """
    A Capsule Network adapted from https://github.com/XifengGuo/CapsNet-Pytorch
    """
    def __init__(self, args, patchwise=None, original_architecture=False):
        super(DynamicCapsules, self).__init__(args, patchwise, original_architecture)
        self.output_size = args.output_size
        self.classes = args.classes
        self.routings = args.routings
        self.lam_recon = args.lam_recon
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Layer 1: Just a conventional Conv2D layer
        self.conv1 = nn.Conv2d(args.output_size[0], 64, kernel_size=9, stride=1, padding=0)

        # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_caps, dim_caps]
        self.primarycaps = PrimaryCapsule(64, 64, 8, kernel_size=9, stride=2, padding=0)

        # Layer 3: Capsule layer. Routing algorithm works here.
        self.digitcaps = DenseCapsule(in_num_caps=4608, in_dim_caps=8,
                                      out_num_caps=args.classes, out_dim_caps=16, routings=self.routings, device=self.device)

        # Decoder network.
        self.decoder = nn.Sequential(
            nn.Linear(16*args.classes, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, args.output_size[0] * args.output_size[1] * args.output_size[2]),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU()

        # This will send net to device, so only call here
        self.init_device()

    def forward(self, x, y=None):
        x = self.relu(self.conv1(x))
        x = self.primarycaps(x)
        x = self.digitcaps(x)
        length = x.norm(dim=-1)
        if y is None:  # during testing, no label given. create one-hot coding using `length`
            index = length.max(dim=1)[1]
            y = Variable(torch.zeros(length.size()).scatter_(1, index.view(-1, 1).cpu().data, 1.).to(self.device))
        reconstruction = self.decoder((x * y[:, :, None]).view(x.size(0), -1))
        return length, reconstruction.view(-1, *self.output_size)

    def propagate(self, inputs, labels, criterion=None):
        labels = torch.zeros(labels.size(0), self.classes).scatter_(1, labels.view(-1, 1), 1.)  # change to one-hot coding
        inputs = self.patch_wise_model.features(inputs)
        if criterion:
            y_pred, x_recon = self(inputs, labels)
            loss = caps_loss(labels, y_pred, inputs, x_recon, self.lam_recon)  # compute loss
        else:
            y_pred, x_recon = self(inputs) # No y in testing
            loss = 0
        return loss, y_pred

class VariationalCapsules(ImageWiseModels): 
    """
    Capsule Routing via Variational Bayes based on https://github.com/fabio-deep/Variational-Capsule-Routing
    """
    def __init__(self, args, patchwise=None, original_architecture=False):
        super(VariationalCapsules, self).__init__(args, patchwise, original_architecture)

        self.output_size = args.output_size
        self.n_classes = args.classes
        self.routings = args.routings

        self.P = args.pose_dim
        self.PP = int(np.max([2, self.P*self.P]))
        self.A, self.B, self.C, self.D = args.arch

        # Layer 1: Just a conventional Conv2D layer
        self.Conv_1 = nn.Conv2d(self.output_size[0], self.A, kernel_size=5, stride=2, bias=False)
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

        # This will send net to device, so only call here
        self.init_device()

    def forward(self, x):
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

class SRCapsules(ImageWiseModels):
    """
    Self Routing Capsules based on https://github.com/coder3000/SR-CapsNet
    """
    def __init__(self, args, patchwise=None, original_architecture=False):
        super(SRCapsules, self).__init__(args, patchwise, original_architecture)

        planes = 16
        last_size = 14
        self.num_caps = 16

        self.conv1 = nn.Conv2d(args.input_size[0], 256, kernel_size=7, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv_a = nn.Conv2d(256, self.num_caps, kernel_size=5, stride=1, padding=1, bias=False)
        self.conv_pose = nn.Conv2d(256, self.num_caps*planes, kernel_size=5, stride=1, padding=1, bias=False)
        self.bn_a = nn.BatchNorm2d(self.num_caps)
        self.bn_pose = nn.BatchNorm2d(self.num_caps*planes)

        self.conv_caps = SelfRouting2d(self.num_caps, self.num_caps, planes, planes, kernel_size=3, stride=2, padding=1, pose_out=True)
        self.bn_pose_conv_caps = nn.BatchNorm2d(self.num_caps*planes)

        self.fc_caps = SelfRouting2d(self.num_caps, args.classes, planes, 1, kernel_size=last_size, padding=0, pose_out=False)
    
        # This will send net to device, so only call here
        self.init_device()

        self.loss = nn.NLLLoss()
        self.loss.to(self.device)

    def forward(self, x):
        #x = torch.rand((8, 3, 32, 32))
        out = F.relu(self.bn1(self.conv1(x)))
        a, pose = self.conv_a(out), self.conv_pose(out)
        a, pose = torch.sigmoid(self.bn_a(a)), self.bn_pose(pose)

        a, pose = self.conv_caps(a, pose)
        pose = self.bn_pose_conv_caps(pose)

        a, _ = self.fc_caps(a, pose)

        out = a.view(a.size(0), -1)
        out = out.log()
        return out
    
    def propagate(self, inputs, labels, criterion=None):
        inputs = self.patch_wise_model.features(inputs)
        y_pred = self(inputs)
        if criterion:
            loss = self.loss(y_pred, labels)
        else:
            loss = 0
        return loss, y_pred