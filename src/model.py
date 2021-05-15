# For the docs
import matplotlib.pyplot as plt

# Data
from torch.utils.data import DataLoader, ConcatDataset
import torchvision.transforms as transforms
from .datasets import MEANS, STD
from tqdm import tqdm
import torchvision
import PIL

# Training
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch
import copy
import time
import os

# For testing we want to get a whole image in patches
BATCH_SIZE = 12

class Model(nn.Module):
    """
    Basic module class, imagewise, patchwise and mixed models inherit from this one, overwriting the propagate method
    """
    def __init__(self, args):
        super(Model, self).__init__()
        self.name = args.name
        self.time = str(time.strftime('%Y-%m-%d_%H-%M'))

        if "breakhis" in args.data_path.lower():
            self.breakhis = True
        else:
            self.breakhis = False

    def init_device(self):
        """ Sends model to CPU / GPU """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        print(self)
        print("Parameters:", sum(p.numel() for p in super(Model, self).parameters()))
        print("Trainable parameters:", sum(p.numel() for p in super(Model, self).parameters() if p.requires_grad))
        print("Using:", self.device)

        # Additional Info when using cuda
        if self.device.type == 'cuda':
            print(torch.cuda.get_device_name(0))
            print('Memory Usage:')
            print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
            print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

    def train_model(self, args, path=None):
        """ Main Training loop with data augmentation, early stopping and scheduler """
        print('Start training network: {}\n'.format(time.strftime('%Y/%m/%d %H:%M')))

        if not args.predefined_stats:
            means = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
        else:
            means = MEANS
            std = STD

        validation_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=means, std=std)
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
                    transforms.Normalize(mean=means, std=std)
                ]),
                # 90 degrees
                transforms.Compose([
                    transforms.RandomRotation((90, 90), resample=PIL.Image.BILINEAR),
                    transforms.ColorJitter(hue=.05, saturation=.05),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=means, std=std)
                ]),
                # 180 degrees
                transforms.Compose([
                    transforms.RandomRotation((180, 180), resample=PIL.Image.BILINEAR),
                    transforms.ColorJitter(hue=.05, saturation=.05),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=means, std=std)
                ]),
                # 270 degrees + flip
                transforms.Compose([
                    transforms.RandomRotation((270, 270), resample=PIL.Image.BILINEAR),
                    transforms.ColorJitter(hue=.05, saturation=.05),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=means, std=std)
                ])
            ]

            if args.flip:
                augmenting += [
                    transforms.Compose([
                        transforms.RandomVerticalFlip(p=1.),
                        transforms.ColorJitter(hue=.05, saturation=.05),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=means, std=std)
                    ]),
                    transforms.Compose([
                        transforms.RandomVerticalFlip(p=1.),
                        transforms.RandomRotation((90, 90), resample=PIL.Image.BILINEAR),
                        transforms.ColorJitter(hue=.05, saturation=.05),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=means, std=std)
                    ]),
                    transforms.Compose([
                        transforms.RandomVerticalFlip(p=1.),
                        transforms.RandomRotation((180, 180), resample=PIL.Image.BILINEAR),
                        transforms.ColorJitter(hue=.05, saturation=.05),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=means, std=std)
                    ]),
                    transforms.Compose([
                        transforms.RandomVerticalFlip(p=1.),
                        transforms.RandomRotation((270, 270), resample=PIL.Image.BILINEAR),
                        transforms.ColorJitter(hue=.05, saturation=.05),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=means, std=std)
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
                transforms.Normalize(mean=means, std=std)
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
                    super(Model, self).train()  # Set model to training mode
                    dataloader = train_data_loader
                else:
                    super(Model, self).eval()   # Set model to evaluate mode
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
                        # Overwrite this for each model
                        loss, preds = self.propagate(inputs, labels, criterion)
                        _, preds = torch.max(preds, 1)

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
                    checkpoint = {
                        'epoch': epoch + 1,
                        'state_dict': best_model_wts,
                        'optimizer': optimizer.state_dict(),
                        'loss': criterion
                    }
                    file_name = "checkpoint_"+ str(epoch + 1) + args.name + self.time + ".ckpt"
                    torch.save(checkpoint, args.checkpoint_path + file_name)

        # Finished
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best Validation Accuracy: {:4f}'.format(best_acc))

        # load best model weights and save checkpoint
        self.load_state_dict(best_model_wts)
    
    def propagate(self, inputs, labels, criterion=None):
        """ Default Training step - some models use this """
        outputs = self(inputs)
        if criterion:
            loss = criterion(outputs, labels)
        else:
            loss = 0
        return loss, outputs

    def plot_metrics(self, path, pr=False, pl=True):
        """ Plots accuracy and loss side-by-side """

        # Plotting on HPC throws error
        if pl:
            # Loss
            plt.subplot(1, 2, 1)
            plt.plot(self.train_losses, label='Training loss')
            plt.plot(self.valid_losses, label='Validation loss')
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend(frameon=False)
            plt.savefig(path + "loss.png")

            # Accuracy
            plt.subplot(1, 2, 2)
            plt.plot(self.train_acc, label='Training Accuracy')
            plt.plot(self.val_acc, label='Validation Accuracy')
            plt.xlabel("Epochs")
            plt.ylabel("Acc")
            plt.legend(frameon=False)
            plt.savefig(path + "accuracy.png")

        if pr:
            print(self.train_losses, self.valid_losses, self.train_acc, self.val_acc, sep="\n")

    def test(self, args, voting=False):
        """ Test on patched dataset """

        if not args.predefined_stats:
            means = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
        else:
            means = MEANS
            std = STD

        test_data = torchvision.datasets.ImageFolder(root=args.data_path + "/test", transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=means, std=std)
        ]))
        test_data_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=args.workers)

        super(Model, self).eval()
        with torch.no_grad():
            patch_acc = 0
            image_acc_maj = 0
            image_acc_sum = 0
            image_acc_max = 0
            for images, labels in tqdm(test_data_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                _, preds = self.propagate(images, labels)
                _, predicted = torch.max(preds, 1)
                patch_acc += (predicted == labels).sum().item()

                if not self.breakhis and voting:
                    # Voting
                    preds = preds.cpu()

                    maj_prob = (args.classes - 1) - np.argmax(np.sum(np.eye(args.classes)[np.array(predicted).reshape(-1)], axis=0)[::-1])
                    sum_prob = (args.classes - 1) - np.argmax(np.sum(np.exp(preds.numpy()), axis=0)[::-1])
                    max_prob = (args.classes - 1) - np.argmax(np.max(np.exp(preds.numpy()), axis=0)[::-1])
        
                    confidence = np.sum(np.array(predicted) == maj_prob) / predicted.size(0)
                    confidence = np.round(confidence * 100, 2)

                    if labels.data[0].item()== maj_prob:
                        image_acc_maj += 1
                        
                    if labels.data[0].item()== sum_prob:
                        image_acc_sum += 1
                        
                    if labels.data[0].item()== max_prob:
                        image_acc_max += 1

        patch_acc  /= len(test_data_loader.dataset)
        print('Test Accuracy of the model: {:.2f} %'.format(patch_acc))

        if not self.breakhis and voting:
            image_acc_maj /=  (len(test_data_loader.dataset)/12)
            image_acc_sum /=  (len(test_data_loader.dataset)/12)
            image_acc_max /=  (len(test_data_loader.dataset)/12)

            print('Test Accuracy of the model on with majority voting: {:.2f} %'.format(image_acc_maj))
            print('Test Accuracy of the model on with sum voting: {:.2f} %'.format(image_acc_sum))
            print('Test Accuracy of the model on with max voting: {:.2f} %'.format(image_acc_max))
    
    def test_separate_classes(self, args):
        """ Tests the model on each class separately and reports classification metrics """
        if not args.predefined_stats:
            means = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
        else:
            means = MEANS
            std = STD

        test_data = torchvision.datasets.ImageFolder(root=args.data_path + "/test", transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=means, std=std)
        ]))
        test_data_loader = DataLoader(test_data, batch_size=args.batch_size, num_workers=args.workers)
        conf_matrix = torch.zeros(args.classes, args.classes)

        super(Model, self).eval()
        with torch.no_grad():
            for images, labels in tqdm(test_data_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                _, predicted = self.propagate(images, labels)
                _, predicted = torch.max(predicted, 1)
                predicted = predicted.tolist()
                labels = labels.tolist()

                for t, p in zip(labels, predicted):
                    conf_matrix[t, p] += 1

        print('Confusion matrix\n', conf_matrix)

        TP = conf_matrix.diag()
        for c in range(args.classes):
            idx = torch.ones(args.classes)
            idx = idx.type(torch.BoolTensor)
            idx[c] = 0
            # all non-class samples classified as non-class
            TN = conf_matrix[idx.nonzero(as_tuple=False)[:, None], idx.nonzero(as_tuple=False)].sum() #conf_matrix[idx[:, None], idx].sum() - conf_matrix[idx, c].sum()
            # all non-class samples classified as class
            FP = conf_matrix[idx, c].sum()
            # all class samples not classified as class
            FN = conf_matrix[c, idx].sum()
            
            print('Class {}\nTP {}, TN {}, FP {}, FN {}'.format(
                c, TP[c], TN, FP, FN))
            print('Sensitivity {:.2f}, Specificity {:.2f}, F1 {:.2f}, Accuracy {:.2f}'.format(
                TP[c] / (TP[c]+FN), TN / (TN + FP), 2*TP[c] / (2*TP[c] + FP + FN), ((TP[c] + TN) / (TP[c] + TN + FP + FN))))

    def test_training(self, args):
        """ Test on patched training dataset for debugging """

        if not args.predefined_stats:
            means = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
        else:
            means = MEANS
            std = STD

        train_data = torchvision.datasets.ImageFolder(root=args.data_path + "/train", transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=means, std=std)
        ]))
        train_data_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

        super(Model, self).eval()
        with torch.no_grad():
            correct = 0
            for images, labels in tqdm(train_data_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                _, predicted = self.propagate(images, labels)
                _, predicted = torch.max(predicted, 1)
                correct += (predicted == labels).sum().item()
                
        print('Training Accuracy of the model: {:.2f}'.format(correct / len(train_data_loader.dataset)))

    def save_model(self, path):
        """ Save model after training has finished """
        file_name = path + self.name + self.time + ".ckpt"
        torch.save(self.state_dict(), file_name)
        print("Model saved:", file_name)
        return file_name
    
    def load(self, path):
        """ Load pre-trained weights """
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