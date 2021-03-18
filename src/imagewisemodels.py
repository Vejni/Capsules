import matplotlib.pyplot as plt

from .patchwisemodel import PatchWiseModel

from .datasets import ImageWiseDataset, set_seed
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
import copy
import time
import os

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
        torch.save(self.state_dict(), path + "imagewise_network_" + model_type +str(time.strftime('%Y/%m/%d %H:%M')) + "ckpt")
        print("Model saved:", path + "imagewise_network_" + model_type + str(time.strftime('%Y/%m/%d %H:%M')) + "ckpt")
        return path + "imagewise_network_" + str(time.strftime('%Y/%m/%d %H:%M')) + "ckpt"
    
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
        x = self.patch_wise_model(x)
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x
    
    def set_linear_layer(self, args):
        train_data_loader = DataLoader(
            dataset=ImageWiseDataset(path=args.data_path + "/train"),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers
        )
        super(BaseCNN, self).train()  # Set model to training mode
        data, _ = next(iter(train_data_loader))
        x = data.to(self.device)
        x = self.patch_wise_model(x)
        x = self.cnn_layers(x)

        self.linear_layers = nn.Sequential(
            nn.Linear(x[1].item()*x[2].item()*x[3].item(), 64),
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

        train_data_loader = DataLoader(
            dataset=ImageWiseDataset(path=args.data_path + "/train", rotate=args.augment, flip=args.augment, enhance=args.augment),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers
        )

        val_data_loader = DataLoader(
            dataset=ImageWiseDataset(path=args.data_path + "/validation"),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers
        )

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
        test_data_loader = DataLoader(
            dataset=ImageWiseDataset(path=args.data_path + "/test"),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers
        )

        super(BaseCNN, self).eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_data_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        print('Test Accuracy of the model: {} %'.format(100 * correct / total))
