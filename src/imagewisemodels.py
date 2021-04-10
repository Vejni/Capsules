import matplotlib.pyplot as plt

from .patchwisemodel import PatchWiseModel

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

class ImageWiseModels(nn.Module):
    """
    Base Image-Wise model, variants inherit from this class
    it assigns a hopefully trained patchwise net to the model to use for forward passes
    """
    def __init__(self, input_size, classes, channels, output_size, patchwise_path=None):
        super(ImageWiseModels, self).__init__()

        self.patch_wise_model = PatchWiseModel(input_size, classes, channels, output_size)
        if patchwise_path is not None:
            self.patch_wise_model.load(patchwise_path)

    def plot_metrics(self):
        """ Plots accuracy and loss side-by-side """

        # Loss
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Training loss')
        plt.plot(self.valid_losses, label='Validation loss')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend(frameon=False)

        # Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(self.train_acc, label='Training Accuracy')
        plt.plot(self.val_acc, label='Validation Accuracy')
        plt.xlabel("Epochs")
        plt.ylabel("Acc")
        plt.legend(frameon=False)

    def train_model(self, args, path=None):
        """ Main Training loop with data augmentation, early stopping and scheduler """

        print('Start training image-wise network: {}\n'.format(time.strftime('%Y/%m/%d %H:%M')))

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
                    super(ImageWiseModels, self).train()  # Set model to training mode
                    dataloader = train_data_loader
                else:
                    super(ImageWiseModels, self).eval()   # Set model to evaluate mode
                    dataloader = val_data_loader

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in tqdm(dataloader):
                    inputs = inputs.to(self.device)
                    inputs = self.patch_wise_model.features(inputs)
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
                    best_optimizer_wts = copy.deepcopy(optimizer.state_dict())
                    best_epoch = epoch

        # Finished
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights and save checkpoint
        self.load_state_dict(best_model_wts)
        self.checkpoint = {
            'epoch': best_epoch + 1,
            'state_dict': best_model_wts,
            'optimizer': best_optimizer_wts
        }
    
    def test(self, args):
        test_data = torchvision.datasets.ImageFolder(root=args.data_path + "/test", transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=MEANS, std=STD)
        ]))
        test_data_loader = DataLoader(test_data, batch_size=BATCH_SIZE, num_workers=args.workers)

        super(ImageWiseModels, self).eval()
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

    def save_model(self, path, model_type):
        file_name = path + "imagewise_network_" + model_type + "_" +str(time.strftime('%Y-%m-%d_%H-%M')) + ".ckpt"
        torch.save(self.state_dict(), file_name)
        print(file_name)
        return file_name
    
    def load(self, path):
        try:
            if os.path.exists(path):
                print('Loading "imagewise" model...')
                self.load_state_dict(torch.load(path))
        except:
            print('Failed to load pre-trained network with path:', path)

    def load_ckp(self, checkpoint_fpath, optimizer):
        """ To continue training we need more than just saving the weights """
        checkpoint = torch.load(checkpoint_fpath)
        self.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        return optimizer, checkpoint['epoch']

    def save_checkpoint(self, path):
        torch.save(self.checkpoint, path)

class BaseCNN(ImageWiseModels):
    """ Simple CNN for baseline, inherits frmo ImageWiseModels """
    def __init__(self, input_size, classes, channels, output_size, patchwise_path, args):
        super(BaseCNN, self).__init__(input_size, classes, channels, output_size, patchwise_path)
        print("Trained PatchWise Model ready to use:", self.patch_wise_model)

        self.cnn_layers = nn.Sequential(
            # Convolutional Layer 1
            nn.Conv2d(input_size[0], 32, kernel_size=5, stride=1, bias=False),
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

            # And flatten
            # nn.Flatten()
        )

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        self.set_linear_layer(args)

        print(self)
        print("Parameters:", sum(p.numel() for p in super(BaseCNN, self).parameters()))
        print("Trainable parameters:", sum(p.numel() for p in super(BaseCNN, self).parameters() if p.requires_grad))
        print("Using:", self.device)

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
        train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,  num_workers=args.workers)
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

class DynamicCapsules(ImageWiseModels): 
    """
    A Capsule Network adapted from https://github.com/XifengGuo/CapsNet-Pytorch
    """
    def __init__(self, input_size, classes, channels, output_size, patchwise_path, args):
        super(DynamicCapsules, self).__init__(input_size, classes, channels, output_size, patchwise_path)
        print("Trained PatchWise Model ready to use:", self.patch_wise_model)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.output_size = output_size
        self.classes = classes
        self.routings = args.routings

        # Layer 1: Just a conventional Conv2D layer
        self.conv1 = nn.Conv2d(output_size[0], 64, kernel_size=9, stride=1, padding=0)

        # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_caps, dim_caps]
        self.primarycaps = PrimaryCapsule(64, 64, 8, kernel_size=9, stride=2, padding=0)

        # Layer 3: Capsule layer. Routing algorithm works here.
        self.digitcaps = DenseCapsule(in_num_caps=4608, in_dim_caps=8,
                                      out_num_caps=classes, out_dim_caps=16, routings=self.routings, device=self.device)

        # Decoder network.
        self.decoder = nn.Sequential(
            nn.Linear(16*classes, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, output_size[0] * output_size[1] * output_size[2]),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU()
        self.to(self.device)

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

    def train_model(self, args):
        """ Need to overwrite it for changes """
        print('Start training image-wise network: {}\n'.format(time.strftime('%Y/%m/%d %H:%M')))
        validation_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=MEANS, std=STD)
        ])

        if args.augment:
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

        optimizer = optim.Adam(self.parameters(), lr=args.lr) # betas=(self.args.beta1, self.args.beta2)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)

        # keeping-track-of-losses 
        self.train_losses = []
        self.valid_losses = []
        self.train_acc = []
        self.val_acc = []
        since = time.time()

        best_model_wts = copy.deepcopy(self.state_dict())
        best_acc = 0.0

        for epoch in range(args.epochs):
            print('Epoch {}/{}'.format(epoch+1, args.epochs))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    super(DynamicCapsules, self).train()  # Set model to training mode
                    dataloader = train_data_loader
                else:
                    super(DynamicCapsules, self).eval()   # Set model to evaluate mode
                    dataloader = val_data_loader

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in tqdm(dataloader):
                    labels = torch.zeros(labels.size(0), args.classes).scatter_(1, labels.view(-1, 1), 1.)  # change to one-hot coding
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    inputs = self.patch_wise_model.features(inputs)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        y_pred, x_recon = self(inputs, labels)
                        loss = caps_loss(labels, y_pred, inputs, x_recon, args.lam_recon)  # compute loss

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    _, predicted = torch.max(y_pred, 1)
                    _, temp = torch.max(labels, 1)
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += (predicted == temp).float().sum()
                if phase == 'train':
                    scheduler.step()

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

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        self.load_state_dict(best_model_wts)    
    
    def test(self, args):
        test_data = torchvision.datasets.ImageFolder(root=args.data_path + "/test", transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=MEANS, std=STD)
        ]))
        test_data_loader = DataLoader(test_data, batch_size=BATCH_SIZE, num_workers=args.workers)

        super(DynamicCapsules, self).eval()
        with torch.no_grad():
            patch_acc = 0
            patch_loss = 0
            image_acc = 0
            for inputs, labels in tqdm(test_data_loader):
                labels = torch.zeros(labels.size(0), args.classes).scatter_(1, labels.view(-1, 1), 1.)  # change to one-hot coding
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                inputs  = self.patch_wise_model.features(inputs)

                y_pred, x_recon = self(inputs) # No y in testing
                _, predicted = torch.max(y_pred, 1)
                _, temp = torch.max(labels, 1)

                patch_acc += (predicted == temp).sum().item()
                patch_loss += caps_loss(labels, y_pred, inputs, x_recon, args.lam_recon).item() * inputs.size(0)  # sum up batch loss
        
                # Majority voting
                predicted = predicted.cpu()
                maj_prob = 2 - np.argmax(np.sum(np.eye(args.classes)[np.array(predicted).reshape(-1)], axis=0)[::-1])
                confidence = np.sum(np.array(predicted) == maj_prob) / predicted.size(0)
                confidence = np.round(confidence * 100, 2)

                if temp.data[0].item()== maj_prob:
                    image_acc += 1

        patch_loss /= len(test_data_loader.dataset)
        patch_acc  /= len(test_data_loader.dataset)
        image_acc /=  (len(test_data_loader.dataset)/12)
        print('Test Loss of the model: {} %'.format(patch_loss))
        print('Test Accuracy of the model: {} %'.format(100 * patch_acc))
        print('Test Accuracy of the model on with majority voting: {} %'.format(100 * image_acc))

class VariationalCapsules(ImageWiseModels): 
    """
    Capsule Routing via Variational Bayes based on https://github.com/fabio-deep/Variational-Capsule-Routing
    """
 
    def __init__(self, input_size, classes, channels, output_size, patchwise_path, args):
        super(VariationalCapsules, self).__init__(input_size, classes, channels, output_size, patchwise_path)
        print("Trained PatchWise Model ready to use:", self.patch_wise_model)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.output_size = output_size
        self.n_classes = args.classes
        self.routings = args.routings

        self.P = args.pose_dim
        self.PP = int(np.max([2, self.P*self.P]))
        self.A, self.B, self.C, self.D = args.arch

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

class EMCapsules(ImageWiseModels):
    """A network with one ReLU convolutional layer followed by
    a primary convolutional capsule layer and two more convolutional capsule layers.

    Suppose image shape is 28x28x1, the feature maps change as follows:
    1. ReLU Conv1
        (_, 1, 28, 28) -> 5x5 filters, 32 out channels, stride 2 with padding
        x -> (_, 32, 14, 14)
    2. PrimaryCaps
        (_, 32, 14, 14) -> 1x1 filter, 32 out capsules, stride 1, no padding
        x -> pose: (_, 14, 14, 32x4x4), activation: (_, 14, 14, 32)
    3. ConvCaps1
        (_, 14, 14, 32x(4x4+1)) -> 3x3 filters, 32 out capsules, stride 2, no padding
        x -> pose: (_, 6, 6, 32x4x4), activation: (_, 6, 6, 32)
    4. ConvCaps2
        (_, 6, 6, 32x(4x4+1)) -> 3x3 filters, 32 out capsules, stride 1, no padding
        x -> pose: (_, 4, 4, 32x4x4), activation: (_, 4, 4, 32)
    5. ClassCaps
        (_, 4, 4, 32x(4x4+1)) -> 1x1 conv, 10 out capsules
        x -> pose: (_, 10x4x4), activation: (_, 10)

        Note that ClassCaps only outputs activation for each class

    Args:
        A: output channels of normal conv
        B: output channels of primary caps
        C: output channels of 1st conv caps
        D: output channels of 2nd conv caps
        E: output channels of class caps (i.e. number of classes)
        K: kernel of conv caps
        P: size of square pose matrix
        iters: number of EM iterations
        ...
    """
    def __init__(self, input_size, classes, channels, output_size, patchwise_path, args):
        super(EMCapsules, self).__init__(input_size, classes, channels, output_size, patchwise_path)

        A, B, C, D, K, P = args.EM_arch
        E = classes
        print("Trained PatchWise Model ready to use:", self.patch_wise_model)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.conv1 = nn.Conv2d(in_channels=input_size[0], out_channels=A,
                               kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(num_features=A, eps=0.001,
                                 momentum=0.1, affine=True)
        self.relu1 = nn.ReLU(inplace=False)
        self.primary_caps = capsules.PrimaryCaps(A, B, 1, P, stride=1)
        self.conv_caps1 = capsules.ConvCaps(self.device, B, C, K, P, stride=2, iters=args.routings)
        self.conv_caps2 = capsules.ConvCaps(self.device, C, D, K, P, stride=1, iters=args.routings)
        self.class_caps = capsules.ConvCaps(self.device, D, E, 1, P, stride=1, iters=args.routings,
                                        coor_add=True, w_shared=True)
        self.to(self.device)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.primary_caps(x)
        x = self.conv_caps1(x)
        x = self.conv_caps2(x)
        x = self.class_caps(x)
        return x
