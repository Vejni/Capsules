import matplotlib.pyplot as plt

from .patchwisemodel import PatchWiseModel

from .CapsNetPytorch.capsulelayers import DenseCapsule, PrimaryCapsule
from .CapsNetPytorch.capsulenet import caps_loss

from .datasets import MEANS, STD
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
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

BATCH_SIZE = 12

class ImageWiseModels(nn.Module):
    def __init__(self, input_size, classes, channels, output_size, patchwise_path):
        super(ImageWiseModels, self).__init__()

        self.patch_wise_model = PatchWiseModel(input_size, classes, channels, output_size)
        self.patch_wise_model.load(patchwise_path)

    def plot_metrics(self):
        # Loss
        plt.plot(self.train_losses, label='Training loss')
        plt.plot(self.valid_losses, label='Validation loss')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend(frameon=False)

        # Accuracy
        plt.plot(self.train_acc, label='Training Accuracy')
        plt.plot(self.val_acc, label='Validation Accuracy')
        plt.xlabel("Epochs")
        plt.ylabel("Acc")
        plt.legend(frameon=False)

    def save_model(self, path, model_type):
        file_name = self.state_dict(), path + "imagewise_network_" + model_type + "_" +str(time.strftime('%Y-%m-%d_%H-%M')) + ".ckpt"
        torch.save(file_name)
        print(file_name)
        return file_name
    
    def load(self, path):
        try:
            if os.path.exists(path):
                print('Loading "imagewise" model...')
                self.load_state_dict(torch.load(path))
        except:
            print('Failed to load pre-trained network with path:', path)

class BaseCNN(ImageWiseModels):
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

    def train_model(self, args):
        print('Start training image-wise network: {}\n'.format(time.strftime('%Y/%m/%d %H:%M')))
        training_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10, resample=PIL.Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEANS, std=STD)
        ])

        validation_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=MEANS, std=STD)
        ])
        train_data = torchvision.datasets.ImageFolder(root=args.data_path + "/train", transform=training_transforms)
        val_data = torchvision.datasets.ImageFolder(root=args.data_path + "/validation", transform=validation_transforms)

        train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,  num_workers=args.workers)
        val_data_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True,  num_workers=args.workers)

        optimizer = optim.Adam(self.parameters(), lr=args.lr) # betas=(self.args.beta1, self.args.beta2)
        criterion = nn.CrossEntropyLoss()

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
                    super(BaseCNN, self).train()  # Set model to training mode
                    dataloader = train_data_loader
                else:
                    super(BaseCNN, self).eval()   # Set model to evaluate mode
                    dataloader = val_data_loader

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in tqdm(dataloader):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    inputs = self.patch_wise_model.features(inputs)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
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

                epoch_loss = running_loss /len(dataloader)
                epoch_acc = running_corrects.double() / len(dataloader)

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

        super(BaseCNN, self).eval()
        with torch.no_grad():
            patch_acc = 0
            image_acc = 0
            for inputs, labels in test_data_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self(inputs)
                _, predicted = torch.max(outputs.data, 1)

                # Majority voting
                maj_prob = 2 - np.argmax(np.sum(np.eye(args.classes)[np.array(predicted).reshape(-1)], axis=0)[::-1])
                confidence = np.sum(np.array(predicted) == maj_prob) / predicted.size(0)
                confidence = np.round(confidence * 100, 2)

                if labels.data[0].item()== maj_prob:
                    image_acc += 1

        patch_acc  /= len(test_data_loader.dataset)
        image_acc /=  (len(test_data_loader.dataset)/12)
        print('Test Accuracy of the model: {} %'.format(100 * patch_acc))
        print('Test Accuracy of the model on with majority voting: {} %'.format(100 * image_acc))


class DynamicCapsules(ImageWiseModels): 
    """
    A Capsule Network 
    :param input_size: data size = [channels, width, height]
    :param classes: number of classes
    :param routings: number of routing iterations
    Shape:
        - Input: (batch, channels, width, height), optional (batch, classes) .
        - Output:((batch, classes), (batch, channels, width, height))
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
        print('Start training image-wise network: {}\n'.format(time.strftime('%Y/%m/%d %H:%M')))
        training_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10, resample=PIL.Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEANS, std=STD)
        ])

        validation_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=MEANS, std=STD)
        ])
        train_data = torchvision.datasets.ImageFolder(root=args.data_path + "/train", transform=training_transforms)
        val_data = torchvision.datasets.ImageFolder(root=args.data_path + "/validation", transform=validation_transforms)

        train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,  num_workers=args.workers)
        val_data_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True,  num_workers=args.workers)

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
