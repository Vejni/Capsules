from torch.utils.data import Subset, DataLoader
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import numpy as np
import torchvision
import random
import torch
import torch
import glob
import PIL
import os

VALIDATION_SET = 0.15
TRAINING_SET = 0.7
TEST_SET = 0.15

PATCH_SIZE = 512
LABELS = ["Grade 1", "Grade 2", "Grade 3"]
SEED = 123


def set_seed(seed):
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
        transforms.ToTensor()
    ])

    data = torchvision.datasets.ImageFolder(root=root_dir, transform=t)

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
        for i in range(3):
            if not os.path.exists(root_dir + "/" + t + "/" + str(i)):
                os.makedirs(root_dir + "/" + t + "/" + str(i))

    # Training images
    i = 0
    for inputs, labels in tqdm(train_data_loader):
        input = inputs[0]
        torchvision.utils.save_image(input, root_dir + "/train/" + str(labels[0].item()) + "/" + str(i) + ".JPG")
        i += 1
    
    # Test images
    i = 0
    for inputs, labels in tqdm(test_data_loader):
        input = inputs[0]
        torchvision.utils.save_image(input, root_dir + "/test/" + str(labels[0].item()) + "/" + str(i) + ".JPG")
        i += 1
    
    # Validation images
    i = 0
    for inputs, labels in tqdm(test_data_loader):
        input = inputs[0]
        torchvision.utils.save_image(input, root_dir + "/validation/" + str(labels[0].item()) + "/" + str(i) + ".JPG")
        i += 1
