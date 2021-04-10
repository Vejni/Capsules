from .imagewisemodels import ImageWiseModels

import torch.nn.functional as F
from .VarCaps import layers
from .VarCaps import vb_routing 

from efficientnet_pytorch import EfficientNet

from .datasets import MEANS, STD
from torch.utils.data import DataLoader, ConcatDataset
import torchvision.transforms as transforms
from tqdm import tqdm
import torchvision
import PIL

import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch
import copy
import time

# For testing we want to get a whole image in patches
BATCH_SIZE = 12

class VariationalMixedCapsules(ImageWiseModels): 
    """
    Capsule Routing via Variational Bayes based on https://github.com/fabio-deep/Variational-Capsule-Routing
    """
 
    def __init__(self, input_size, classes, channels, output_size, args, features=True):
        super(VariationalMixedCapsules, self).__init__(input_size, classes, channels, output_size)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.output_size = output_size
        self.n_classes = args.classes
        self.routings = args.routings

        self.P = args.pose_dim
        self.PP = int(np.max([2, self.P*self.P]))
        self.A, self.B, self.C, self.D = args.arch
        K = 12

        if features:
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
        else:
            self.features = None


        # Layer 1: Just a conventional Conv2D layer
        self.Conv_1 = nn.Conv2d(input_size[0], self.A, kernel_size=5, stride=2, bias=False)
        nn.init.kaiming_uniform_(self.Conv_1.weight)

        self.Conv_2 = nn.Conv2d(self.A, self.A, kernel_size=5, stride=2, bias=False)
        nn.init.kaiming_uniform_(self.Conv_2.weight)

        self.Conv_3 = nn.Conv2d(self.A, self.A, kernel_size=3, stride=2, bias=False)
        nn.init.kaiming_uniform_(self.Conv_3.weight)

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
            kernel_size=K, stride=1, pose_dim=self.P, # adjust final kernel_size K depending on input H/W, for H=W=32, K=4.
            cov='diag', iter=args.routings,
            alpha0=1., m0=torch.zeros(self.PP), kappa0=1.,
            Psi0=torch.eye(self.PP), nu0=self.PP+1, class_caps=True)
        self.to(self.device)

    def forward(self, x):
        if self.features:
            x = self.features(x)
            # Out ← [?, A, F, F]
            x = F.relu(self.BN_1(self.Conv_1(x)))
        else:
            # Out ← [?, A, F, F]
            x = F.relu(self.BN_1(self.Conv_1(x)))
            x = F.relu(self.BN_1(self.Conv_2(x)))
            x = F.relu(self.BN_1(self.Conv_2(x)))
            x = F.relu(self.BN_1(self.Conv_3(x)))
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

    def train_model(self, args, path=None):
        """ Main Training loop with data augmentation, early stopping and scheduler """

        print('Start training patch-wise network: {}\n'.format(time.strftime('%Y/%m/%d %H:%M')))

        validation_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=MEANS, std=STD)
        ])

        if args.augment:
            """
            Create versions of the dataset for each augmentation as in https://arxiv.org/abs/1803.04054 and others
            """
            augmenting = [
                transforms.Compose([
                    transforms.RandomRotation(10, resample=PIL.Image.BILINEAR),
                    transforms.ColorJitter(hue=.05, saturation=.05),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=MEANS, std=STD)
                ]),
                transforms.Compose([
                    transforms.RandomHorizontalFlip(p=1.),
                    transforms.RandomRotation(10, resample=PIL.Image.BILINEAR),
                    transforms.ColorJitter(hue=.05, saturation=.05),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=MEANS, std=STD)
                ]),
                transforms.Compose([
                    transforms.RandomVerticalFlip(p=1.),
                    transforms.RandomRotation(10, resample=PIL.Image.BILINEAR),
                    transforms.ColorJitter(hue=.05, saturation=.05),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=MEANS, std=STD)
                ]),
                transforms.Compose([
                    transforms.RandomHorizontalFlip(p=1.),
                    transforms.RandomVerticalFlip(p=1.),
                    transforms.RandomRotation(10, resample=PIL.Image.BILINEAR),
                    transforms.ColorJitter(hue=.05, saturation=.05),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=MEANS, std=STD)
                ])
            ]

            train_data_loader = DataLoader(
                ConcatDataset([
                    torchvision.datasets.ImageFolder(root=args.data_path + "/train", transform=t) for t in augmenting
                ]), 
                batch_size=args.batch_size, shuffle=True,  num_workers=args.workers
            )
        
        else:
            training_transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(10, resample=PIL.Image.BILINEAR),
                transforms.ColorJitter(hue=.05, saturation=.05),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEANS, std=STD)
            ])
            train_data = torchvision.datasets.ImageFolder(root=args.data_path + "/train", transform=training_transforms)
            train_data_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,  num_workers=args.workers)
        
        val_data = torchvision.datasets.ImageFolder(root=args.data_path + "/validation", transform=validation_transforms)
        val_data_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=True,  num_workers=args.workers)

        optimizer = optim.Adam(self.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        criterion = nn.CrossEntropyLoss()
        start_epoch = 0

        # If checkpoint provided load states
        if path:
            optimizer, start_epoch = self.load_ckp(path, optimizer)
            print("Model loaded, trained for ", start_epoch, "epochs")

        # keeping track of losses 
        self.train_losses = []
        self.valid_losses = []
        self.train_acc = []
        self.val_acc = []
        since = time.time()

        # For "early stopping"
        best_model_wts = copy.deepcopy(self.state_dict())
        best_acc = 0.

        for epoch in range(start_epoch, args.epochs):
            print('Epoch {}/{}'.format(epoch+1, args.epochs))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    super(VariationalMixedCapsules, self).train()  # Set model to training mode
                    dataloader = train_data_loader
                else:
                    super(VariationalMixedCapsules, self).eval()   # Set model to evaluate mode
                    dataloader = val_data_loader

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in tqdm(dataloader):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # Training here
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                # Adjust learning rate
                if phase == 'train':
                    scheduler.step()

                # Data metrics
                epoch_loss = running_loss /len(dataloader.dataset)
                epoch_acc = running_corrects.double() / len(dataloader.dataset)

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))
                
                if phase == 'train':
                    self.train_losses.append(epoch_loss)
                    self.train_acc.append(epoch_acc)
                else:
                    self.valid_losses.append(epoch_loss)
                    self.val_acc.append(epoch_acc)

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.state_dict())
                    self.checkpoint = {
                        'epoch': epoch + 1,
                        'state_dict': best_model_wts,
                        'optimizer': optimizer.state_dict(),
                        'loss': criterion
                    }
                    file_name = "checkpoint_"+ str(epoch + 1) + "_patchwise_network_" + self.time + ".ckpt"
                    self.save_checkpoint(args.chechpoint_path + file_name)

        # Finished
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights and save checkpoint
        self.load_state_dict(best_model_wts)
    
    def test(self, args):
        test_data = torchvision.datasets.ImageFolder(root=args.data_path + "/test", transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=MEANS, std=STD)
        ]))
        test_data_loader = DataLoader(test_data, batch_size=BATCH_SIZE, num_workers=args.workers)

        super(VariationalMixedCapsules, self).eval()
        with torch.no_grad():
            patch_acc = 0
            image_acc = 0
            for inputs, labels in tqdm(test_data_loader):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                inputs  = self.patch_wise_model.features(inputs)

                y_pred = self(inputs) # No y in testing
                _, predicted = torch.max(y_pred, 1)
                patch_acc += (predicted == labels).sum().item()
                        
                # Majority voting
                predicted = predicted.cpu()
                maj_prob = 2 - np.argmax(np.sum(np.eye(args.classes)[np.array(predicted).reshape(-1)], axis=0)[::-1])
                confidence = np.sum(np.array(predicted) == maj_prob) / predicted.size(0)
                confidence = np.round(confidence * 100, 2)

                if labels.data[0].item()== maj_prob:
                    image_acc += 1

        patch_acc  /= len(test_data_loader.dataset)
        image_acc /=  (len(test_data_loader.dataset)/12)

        print('Test Accuracy of the model: {} %'.format(100 * patch_acc))
        print('Test Accuracy of the model on with majority voting: {} %'.format(100 * image_acc))


class EffNet(ImageWiseModels): 
    """
    TODO
    """
 
    def __init__(self, input_size, classes, channels, output_size, args):
        super(EffNet, self).__init__(input_size, classes, channels, output_size)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.output_size = output_size
        self.n_classes = args.classes

        self.model = EfficientNet.from_pretrained('efficientnet-b0')
        num_ftrs = self.model._fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 128),
            nn.ReLU(), 
            nn.Linear(128, self.n_classes)
        )

        print(self.model)
        print("Parameters:", sum(p.numel() for p in self.model.parameters()))
        print("Trainable parameters:", sum(p.numel() for p in self.model.parameters() if p.requires_grad))

    def train_model(self, args, path=None):
        """ Main Training loop with data augmentation, early stopping and scheduler """

        validation_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=MEANS, std=STD)
        ])

        if args.augment:
            """
            Create versions of the dataset for each augmentation as in https://arxiv.org/abs/1803.04054 and others
            """
            augmenting = [
                transforms.Compose([
                    transforms.RandomRotation(10, resample=PIL.Image.BILINEAR),
                    transforms.ColorJitter(hue=.05, saturation=.05),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=MEANS, std=STD)
                ]),
                transforms.Compose([
                    transforms.RandomHorizontalFlip(p=1.),
                    transforms.RandomRotation(10, resample=PIL.Image.BILINEAR),
                    transforms.ColorJitter(hue=.05, saturation=.05),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=MEANS, std=STD)
                ]),
                transforms.Compose([
                    transforms.RandomVerticalFlip(p=1.),
                    transforms.RandomRotation(10, resample=PIL.Image.BILINEAR),
                    transforms.ColorJitter(hue=.05, saturation=.05),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=MEANS, std=STD)
                ]),
                transforms.Compose([
                    transforms.RandomHorizontalFlip(p=1.),
                    transforms.RandomVerticalFlip(p=1.),
                    transforms.RandomRotation(10, resample=PIL.Image.BILINEAR),
                    transforms.ColorJitter(hue=.05, saturation=.05),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=MEANS, std=STD)
                ])
            ]

            train_data_loader = DataLoader(
                ConcatDataset([
                    torchvision.datasets.ImageFolder(root=args.data_path + "/train", transform=t) for t in augmenting
                ]), 
                batch_size=args.batch_size, shuffle=True,  num_workers=args.workers
            )
        
        else:
            training_transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(10, resample=PIL.Image.BILINEAR),
                transforms.ColorJitter(hue=.05, saturation=.05),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEANS, std=STD)
            ])
            train_data = torchvision.datasets.ImageFolder(root=args.data_path + "/train", transform=training_transforms)
            train_data_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,  num_workers=args.workers)
        
        val_data = torchvision.datasets.ImageFolder(root=args.data_path + "/validation", transform=validation_transforms)
        val_data_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=True,  num_workers=args.workers)

        optimizer = optim.Adam(self.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        criterion = nn.CrossEntropyLoss()
        start_epoch = 0

        # If checkpoint provided load states
        if path:
            optimizer, start_epoch = self.load_ckp(path, optimizer)
            print("Model loaded, trained for ", start_epoch, "epochs")

        # keeping track of losses 
        self.train_losses = []
        self.valid_losses = []
        self.train_acc = []
        self.val_acc = []
        since = time.time()

        # For "early stopping"
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.

        for epoch in range(start_epoch, args.epochs):
            print('Epoch {}/{}'.format(epoch+1, args.epochs))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    super(EffNet, self).train()  # Set model to training mode
                    dataloader = train_data_loader
                else:
                    super(EffNet, self).eval()   # Set model to evaluate mode
                    dataloader = val_data_loader

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in tqdm(dataloader):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # Training here
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                # Adjust learning rate
                if phase == 'train':
                    scheduler.step()

                # Data metrics
                epoch_loss = running_loss /len(dataloader.dataset)
                epoch_acc = running_corrects.double() / len(dataloader.dataset)

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))
                
                if phase == 'train':
                    self.train_losses.append(epoch_loss)
                    self.train_acc.append(epoch_acc)
                else:
                    self.valid_losses.append(epoch_loss)
                    self.val_acc.append(epoch_acc)

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.state_dict())
                    self.checkpoint = {
                        'epoch': epoch + 1,
                        'state_dict': best_model_wts,
                        'optimizer': optimizer.state_dict(),
                        'loss': criterion
                    }
                    file_name = "checkpoint_"+ str(epoch + 1) + "_patchwise_network_" + self.time + ".ckpt"
                    self.save_checkpoint(args.chechpoint_path + file_name)

        # Finished
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights and save checkpoint
        self.model.load_state_dict(best_model_wts)
    
    def test(self, args):
        test_data = torchvision.datasets.ImageFolder(root=args.data_path + "/test", transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=MEANS, std=STD)
        ]))
        test_data_loader = DataLoader(test_data, batch_size=BATCH_SIZE, num_workers=args.workers)

        super(EffNet, self).eval()
        with torch.no_grad():
            patch_acc = 0
            image_acc = 0
            for inputs, labels in tqdm(test_data_loader):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                inputs  = self.patch_wise_model.features(inputs)

                y_pred = self(inputs) # No y in testing
                _, predicted = torch.max(y_pred, 1)
                patch_acc += (predicted == labels).sum().item()
                        
                # Majority voting
                predicted = predicted.cpu()
                maj_prob = 2 - np.argmax(np.sum(np.eye(args.classes)[np.array(predicted).reshape(-1)], axis=0)[::-1])
                confidence = np.sum(np.array(predicted) == maj_prob) / predicted.size(0)
                confidence = np.round(confidence * 100, 2)

                if labels.data[0].item()== maj_prob:
                    image_acc += 1

        patch_acc  /= len(test_data_loader.dataset)
        image_acc /=  (len(test_data_loader.dataset)/12)

        print('Test Accuracy of the model: {} %'.format(100 * patch_acc))
        print('Test Accuracy of the model on with majority voting: {} %'.format(100 * image_acc))

