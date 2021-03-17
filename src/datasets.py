from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image, ImageEnhance
import numpy as np
import torchvision
import random
import torch
import torch
import glob
import os

from src.patching import PatchExtractor

VALIDATION_SET = 0.15
TRAINING_SET = 0.7
TEST_SET = 0.15

MEANS = [0.4731, 0.3757, 0.4117]
STD = [0.3731, 0.3243, 0.3199]

IMAGE_SIZE = (1536, 2048)
LABELS = ["Grade 1", "Grade 2", "Grade 3"]
SEED = 123

PATCH_SIZE = 512
STRIDE = 256

def set_seed(seed=SEED):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def split_test_train_val(root_dir, test_set=TEST_SET, training_set=TRAINING_SET, val_set=VALIDATION_SET):
    from tqdm import tqdm
    set_seed(SEED)
    assert test_set + training_set + val_set == 1, "Train/Test/Val Set sizes incorrect"
    BATCH_SIZE = 1

    t = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor()
    ])

    data = torchvision.datasets.ImageFolder(root=root_dir+"/Histopathological_Graded", transform=t)

    train_size = int(training_set*len(data))
    val_size = int(val_set*len(data))
    test_size = len(data) - (train_size + val_size)
    train_data, test_data, val_data = torch.utils.data.random_split(data, [train_size, test_size, val_size])

    train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE,  num_workers=0)
    test_data_loader = DataLoader(test_data, batch_size=BATCH_SIZE, num_workers=0)
    val_data_loader = DataLoader(val_data, batch_size=BATCH_SIZE, num_workers=0)

    print("Number of training images:", len(train_data), "Number of test images:", len(test_data), "Number of validation images:", len(val_data))

    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    for t in ["train", "test", "validation"]:
        for i in range(len(LABELS)):
            if not os.path.exists(root_dir + "/" + t + "/" + LABELS[i]):
                os.makedirs(root_dir + "/" + t + "/" + LABELS[i])

    # Compute normalization metrics
    mean = 0.
    std = 0.

    # Training images
    i = 0
    for inputs, labels in tqdm(train_data_loader):
        input = inputs[0]

        temp = input.view(3, -1)
        mean += temp.mean(1)
        std += temp.std(1)

        torchvision.utils.save_image(input, root_dir + "/train/" + LABELS[labels[0].item()] + "/" + str(i) + ".JPG")
        i += 1
    
    # Validation images
    i = 0
    for inputs, labels in tqdm(val_data_loader):
        input = inputs[0]

        temp = input.view(3, -1)
        mean += temp.mean(1)
        std += temp.std(1)

        torchvision.utils.save_image(input, root_dir + "/validation/" + LABELS[labels[0].item()] + "/" + str(i) + ".JPG")
        i += 1

    # Test images
    i = 0
    for inputs, labels in tqdm(test_data_loader):
        input = inputs[0]
        torchvision.utils.save_image(input, root_dir + "/test/" + LABELS[labels[0].item()] + "/" + str(i) + ".JPG")
        i += 1
    
    print("Printing Normalization Metrics")
    mean /= (train_size + val_size)
    std /= (train_size + val_size)
    print("Means:", mean)
    print("Std:", std)

class PatchWiseDataset(Dataset):
    def __init__(self, path, stride=STRIDE, rotate=False, flip=False, enhance=False):
        super().__init__()

        wp = int((IMAGE_SIZE[0] - PATCH_SIZE) / stride + 1)
        hp = int((IMAGE_SIZE[1] - PATCH_SIZE) / stride + 1)
        labels = {name: index for index in range(len(LABELS)) for name in glob.glob(path + '/' + LABELS[index] + '/*.JPG')}

        self.path = path
        self.stride = stride
        self.labels = labels
        self.names = list(sorted(labels.keys()))
        self.shape = (len(labels), wp, hp, (4 if rotate else 1), (2 if flip else 1), (2 if enhance else 1))  # (files, x_patches, y_patches, rotations, flip, enhance)
        self.augment_size = np.prod(self.shape) / len(labels)

        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=MEANS, std=STD)
        ])

    def __getitem__(self, index):
        im, xpatch, ypatch, rotation, flip, enhance = np.unravel_index(index, self.shape)

        with Image.open(self.names[im]) as img:
            extractor = PatchExtractor(img=img, patch_size=PATCH_SIZE, stride=self.stride)
            patch = extractor.extract_patch((xpatch, ypatch))

            if rotation != 0:
                patch = patch.rotate(rotation * 90)

            if flip != 0:
                patch = patch.transpose(Image.FLIP_LEFT_RIGHT)

            if enhance != 0:
                factors = np.random.uniform(.5, 1.5, 3)
                patch = ImageEnhance.Color(patch).enhance(factors[0])
                patch = ImageEnhance.Contrast(patch).enhance(factors[1])
                patch = ImageEnhance.Brightness(patch).enhance(factors[2])

            label = self.labels[self.names[im]]
            return self.transforms(patch), label

    def __len__(self):
        return np.prod(self.shape)
