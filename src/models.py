from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import torchvision
import PIL

import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch
import copy
import time

MEANS = [0.4731, 0.3757, 0.4117]
STD = [0.3731, 0.3243, 0.3199]

class PatchWiseModel(nn.Module):
    """
    TODO
    """
    def __init__(self, input_size, classes, channels, output_size):
        super(PatchWiseModel, self).__init__()
        self.input_size = input_size
        self.classes = classes

        # Convolutional Layers
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

            # Block 4
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Block 5
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256, out_channels=channels, kernel_size=1, stride=1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(channels * output_size[1] * output_size[2], classes),
        )

        self.initialize_weights()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        super(PatchWiseModel, self).print()
        print("Parameters:", sum(p.numel() for p in super(PatchWiseModel, self).parameters()))
        print("Trainable parameters:", sum(p.numel() for p in super(PatchWiseModel, self).parameters() if p.requires_grad))

    def initialize_weights(self):
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

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = F.log_softmax(x, dim=1)
        return x

    def train(self, data_path, args):
        print('Start training patch-wise network: {}\n'.format(time.strftime('%Y/%m/%d %H:%M')))

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

        training_data = torchvision.datasets.ImageFolder(root=data_path + "/train", transform=training_transforms)
        train_data_loader = DataLoader(training_data, batch_size=args.batch_size, shuffle=True,  num_workers=2, drop_last=True)

        validation_data = torchvision.datasets.ImageFolder(root=data_path + "/validation", transform=validation_transforms)
        val_data_loader = DataLoader(validation_data, batch_size=args.batch_size, shuffle=True,  num_workers=2, drop_last=True)

        optimizer = optim.Adam(self.parameters(), lr=args.lr) # betas=(self.args.beta1, self.args.beta2)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
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
                    super(PatchWiseModel, self).train()  # Set model to training mode
                    dataloader = train_data_loader
                else:
                    super(PatchWiseModel, self).eval()   # Set model to evaluate mode
                    dataloader = val_data_loader

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloader:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

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
                if phase == 'train':
                    scheduler.step()

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