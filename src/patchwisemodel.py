import matplotlib.pyplot as plt

import torchvision.transforms as transforms
from torch.utils.data import DataLoader, ConcatDataset
from .datasets import MEANS, STD
from tqdm import tqdm
import torchvision
import PIL

import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
import copy
import time
import os

class PatchWiseModel(nn.Module):
    """
    A CNN classifier that is used by the image-wise networks to downscale the images
    by feeding them though the convolutional layers of the trained patchwise net.
    """
    def __init__(self, args, original_architecture=False):
        super(PatchWiseModel, self).__init__()
        self.name = args.name
        self.time = str(time.strftime('%Y-%m-%d_%H-%M'))

        if original_architecture:
            """
            This is the original architecture proposed in: https://arxiv.org/abs/1803.04054
            it is prone to overfit
            """
            self.features = nn.Sequential(
                # Block 1
                nn.Conv2d(in_channels=args.input_size[0], out_channels=16, kernel_size=3, stride=1, padding=1),
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

                nn.Conv2d(in_channels=256, out_channels=args.output_size[2], kernel_size=1, stride=1),
            )
        else:
            """
            Smaller version using 10 conv layers instead of 16
            """
            self.features = nn.Sequential(
                # Block 1
                nn.Conv2d(in_channels=args.input_size[0], out_channels=16, kernel_size=3, stride=1, padding=1),
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

                nn.Conv2d(in_channels=64, out_channels=args.output_size[0], kernel_size=1, stride=1),
            )

        # The classification layer
        self.classifier = nn.Sequential(
            nn.Linear(args.output_size[0] * args.output_size[2] * args.output_size[1], args.classes),
        )

        self.initialize_weights()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        print(self)
        print("Parameters:", sum(p.numel() for p in super(PatchWiseModel, self).parameters()))
        print("Trainable parameters:", sum(p.numel() for p in super(PatchWiseModel, self).parameters() if p.requires_grad))
        print("Using:", self.device)

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

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = F.log_softmax(x, dim=1)
        return x

    def train_model(self, args, path=None):
        """ Main Training loop with data augmentation, early stopping and scheduler """

        print('Start training network: {}\n'.format(time.strftime('%Y/%m/%d %H:%M')))

        if args.test_stat:
            MEANS = [0.5, 0.5, 0.5]
            STD = [0.5, 0.5, 0.5]

        validation_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=MEANS, std=STD)
        ])

        if args.augment:
            """
            Create versions of the dataset for each augmentation as in https://arxiv.org/abs/1803.04054 and others
            """
            augmenting = [
                # 0 degrees
                transforms.Compose([
                    transforms.ColorJitter(hue=.05, saturation=.05),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=MEANS, std=STD)
                ]),
                # 90 degrees
                transforms.Compose([
                    transforms.RandomRotation((90, 90), resample=PIL.Image.BILINEAR),
                    transforms.ColorJitter(hue=.05, saturation=.05),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=MEANS, std=STD)
                ]),
                # 180 degrees
                transforms.Compose([
                    transforms.RandomRotation((180, 180), resample=PIL.Image.BILINEAR),
                    transforms.ColorJitter(hue=.05, saturation=.05),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=MEANS, std=STD)
                ]),
                # 270 degrees + flip
                transforms.Compose([
                    transforms.RandomRotation((270, 270), resample=PIL.Image.BILINEAR),
                    transforms.ColorJitter(hue=.05, saturation=.05),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=MEANS, std=STD)
                ])
            ]

            if args.flip:
                augmenting += [
                    transforms.Compose([
                        transforms.RandomVerticalFlip(p=1.),
                        transforms.ColorJitter(hue=.05, saturation=.05),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=MEANS, std=STD)
                    ]),
                    transforms.Compose([
                        transforms.RandomVerticalFlip(p=1.),
                        transforms.RandomRotation((90, 90), resample=PIL.Image.BILINEAR),
                        transforms.ColorJitter(hue=.05, saturation=.05),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=MEANS, std=STD)
                    ]),
                    transforms.Compose([
                        transforms.RandomVerticalFlip(p=1.),
                        transforms.RandomRotation((180, 180), resample=PIL.Image.BILINEAR),
                        transforms.ColorJitter(hue=.05, saturation=.05),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=MEANS, std=STD)
                    ]),
                    transforms.Compose([
                        transforms.RandomVerticalFlip(p=1.),
                        transforms.RandomRotation((270, 270), resample=PIL.Image.BILINEAR),
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

        print("Using ", len(train_data_loader.dataset), "training samples")
        print("Using ", len(val_data_loader.dataset), "validation samples")

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
                    file_name = "checkpoint_"+ str(epoch + 1) + args.name + self.time + ".ckpt"
                    self.save_checkpoint(args.checkpoint_path + file_name)

        # Finished
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights and save checkpoint
        self.load_state_dict(best_model_wts)
    
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

    def test(self, args):
        """ Test on patched dataset """
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
        """ Tests the model on each class separately and reports the accuracies """
        test_data = torchvision.datasets.ImageFolder(root=args.data_path + "/test", transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=MEANS, std=STD)
        ]))
        test_data_loader = DataLoader(test_data, batch_size=args.batch_size, num_workers=args.workers)

        tp = [0] * args.classes
        tpfp = [0] * args.classes
        tpfn = [0] * args.classes
        precision = [0] * args.classes
        recall = [0] * args.classes
        f1 = [0] * args.classes

        class_correct = list(0. for i in range(args.classes))
        class_total = list(0. for i in range(args.classes))

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

                for label in range(args.classes):
                    t_labels = torch.sum(torch.tensor(labels) == label)
                    p_labels = torch.sum(torch.tensor(predicted) == label)
                    tp[label] += torch.sum(t_labels == (p_labels * 2 - 1))
                    tpfp[label] += torch.sum(p_labels)
                    tpfn[label] += torch.sum(t_labels)

        for label in range(args.classes):
            print('Accuracy of %5s : %2d %%' % (
                label, 100 * class_correct[label] / class_total[label]))

            precision[label] += (tp[label] / (tpfp[label] + 1e-8))
            recall[label] += (tp[label] / (tpfn[label] + 1e-8))
            f1[label] = 2 * precision[label] * recall[label] / (precision[label] + recall[label] + 1e-8)

            print('{}:  \t Precision: {:.2f},  Recall: {:.2f},  F1: {:.2f}'.format(
                    label,
                    precision[label],
                    recall[label],
                    f1[label]
                ))

            print('')

    def save_checkpoint(self, path):
        torch.save(self.checkpoint, path)

    def save_model(self, path):
        file_name = path + self.name + self.time + ".ckpt"
        torch.save(self.state_dict(), file_name)
        print("Model saved:", file_name)
        return file_name
    
    def load(self, path):
        try:
            if os.path.exists(path):
                print('Loading model...')
                self.load_state_dict(torch.load(path))
        except:
            print('Failed to load pre-trained network with path:', path)
    
    def load_ckp(self, checkpoint_fpath, optimizer):
        """ To continue training we need more than just saving the weights """
        checkpoint = torch.load(checkpoint_fpath)
        self.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        return optimizer, checkpoint['epoch']