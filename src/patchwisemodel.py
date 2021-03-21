import matplotlib.pyplot as plt

import torchvision.transforms as transforms
from torch.utils.data import DataLoader, ConcatDataset
from .datasets import MEANS, STD
from tqdm import tqdm
import torchvision
import PIL

from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
import copy
import time
import os

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
            nn.Linear(channels * output_size[2] * output_size[1], classes),
        )

        self.initialize_weights()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        print(self)
        print("Parameters:", sum(p.numel() for p in super(PatchWiseModel, self).parameters()))
        print("Trainable parameters:", sum(p.numel() for p in super(PatchWiseModel, self).parameters() if p.requires_grad))
        print("Using:", self.device)

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

    def output(self, input_tensor):
        super(PatchWiseModel, self).eval()
        with torch.no_grad():
            res = self.features(Variable(input_tensor))
        return res.squeeze()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = F.log_softmax(x, dim=1)
        return x

    def train_model(self, args):
        print('Start training patch-wise network: {}\n'.format(time.strftime('%Y/%m/%d %H:%M')))

        training_transforms_1 = transforms.Compose([
            transforms.RandomRotation(10, resample=PIL.Image.BILINEAR),
            transforms.ColorJitter(hue=.05, saturation=.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEANS, std=STD)
        ])

        training_transforms_2 = transforms.Compose([
            transforms.RandomHorizontalFlip(p=1.),
            transforms.RandomRotation(10, resample=PIL.Image.BILINEAR),
            transforms.ColorJitter(hue=.05, saturation=.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEANS, std=STD)
        ])

        training_transforms_3 = transforms.Compose([
            transforms.RandomVerticalFlip(p=1.),
            transforms.RandomRotation(10, resample=PIL.Image.BILINEAR),
            transforms.ColorJitter(hue=.05, saturation=.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEANS, std=STD)
        ])

        validation_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=MEANS, std=STD)
        ])

        train_data_1 = torchvision.datasets.ImageFolder(root=args.data_path + "/train", transform=training_transforms_1)
        train_data_2 = torchvision.datasets.ImageFolder(root=args.data_path + "/train", transform=training_transforms_2)
        train_data_3 = torchvision.datasets.ImageFolder(root=args.data_path + "/train", transform=training_transforms_3)
        val_data = torchvision.datasets.ImageFolder(root=args.data_path + "/validation", transform=validation_transforms)

        train_data_loader = DataLoader(ConcatDataset([train_data_1, train_data_2, train_data_3]), batch_size=args.batch_size, shuffle=True,  num_workers=args.workers)
        val_data_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=True,  num_workers=args.workers)

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
        best_acc = 0.

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
                for inputs, labels in tqdm(dataloader):
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

    def test(self, args):
        test_data = torchvision.datasets.ImageFolder(root=args.data_path + "/test", transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=MEANS, std=STD)
        ]))
        test_data_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

        super(PatchWiseModel, self).eval()
        with torch.no_grad():
            correct = 0
            for images, labels in tqdm(test_data_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self(images)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                
        print('Test Accuracy of the model: {} %'.format(100 * correct / len(test_data_loader.dataset)))
    
    def test_separate_classes(self, args):
        test_data = torchvision.datasets.ImageFolder(root=args.data_path + "/test", transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=MEANS, std=STD)
        ]))
        test_data_loader = DataLoader(test_data, batch_size=args.batch_size, num_workers=args.workers)


        class_correct = list(0. for i in range(3))
        class_total = list(0. for i in range(3))

        super(PatchWiseModel, self).eval()
        with torch.no_grad():
            for images, labels in tqdm(test_data_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self(images)
                _, predicted = torch.max(outputs, 1)
                predicted = predicted.tolist()
                labels = labels.tolist()
                for i in range(len(predicted)):
                    if predicted[i] == labels[i]:
                        class_correct[predicted[i]] += 1
                    class_total[labels[i]] += 1


        for i in range(3):
            print('Accuracy of %5s : %2d %%' % (
                i, 100 * class_correct[i] / class_total[i]))
    
    def save_model(self, path):
        file_name = path + "patchwise_network_" + str(time.strftime('%Y-%m-%d_%H-%M')) + ".ckpt"
        torch.save(self.state_dict(), file_name)
        print("Model saved:", file_name)
        return file_name
    
    def load(self, path):
        try:
            if os.path.exists(path):
                print('Loading "patch-wise" model...')
                self.load_state_dict(torch.load(path))
        except:
            print('Failed to load pre-trained network with path:', path)