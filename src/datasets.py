import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import torchvision
import random
import torch
import torch
import os

#from .patching import PatchExtractor

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

def imshow(img):
    import matplotlib.pyplot as plt
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def set_seed(seed=SEED):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def compute_normalization_stats(root_dir,  test_set=TEST_SET, training_set=TRAINING_SET, val_set=VALIDATION_SET):
    from tqdm import tqdm
    set_seed(SEED)
    assert test_set + training_set + val_set == 1, "Train/Test/Val Set sizes incorrect"
    BATCH_SIZE = 1

    t = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor()
    ])

    data = torchvision.datasets.ImageFolder(root=root_dir+"/Histopathological_Graded", transform=t)

    size = int((training_set+val_set)*len(data))
    test_size = len(data) - (size)
    train_data, _ = torch.utils.data.random_split(data, [size, test_size])

    data_loader = DataLoader(train_data, batch_size=BATCH_SIZE,  num_workers=0)
    # Compute normalization metrics
    mean = 0.
    std = 0.

    # Training images
    i = 0
    for inputs, _ in tqdm(data_loader):
        input = inputs[0]

        temp = input.view(3, -1)
        mean += temp.mean(1)
        std += temp.std(1)
        i += 1

    print("Printing Normalization Metrics")
    mean /= size
    std /= size
    print("Means:", mean)
    print("Std:", std)

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

    for mode in ["/imagewise_dataset/", "/patchwise_dataset/"]:
        for t in ["train", "test", "validation"]:
            for i in range(len(LABELS)):
                if not os.path.exists(root_dir + mode + t + "/" + LABELS[i]):
                    os.makedirs(root_dir + mode + t + "/" + LABELS[i])

    # Training images
    i = 0
    for inputs, labels in tqdm(train_data_loader):
        input = inputs[0]
        # ImageWise
        patches = input.unfold(1, PATCH_SIZE, PATCH_SIZE).unfold(2, PATCH_SIZE, PATCH_SIZE)
        patches = patches.permute(1,2,0,3,4).contiguous()
        patches = patches.contiguous().view(-1, 3, PATCH_SIZE, PATCH_SIZE)
        for j, patch in enumerate(patches):
            torchvision.utils.save_image(patch, root_dir + "/imagewise_dataset/train/" + LABELS[labels[0].item()] + "/image_" + str(i) + "patch_" + str(j) + ".JPG")
        
        # PatchWise
        patches = input.unfold(1, PATCH_SIZE, STRIDE).unfold(2, PATCH_SIZE, STRIDE)
        patches = patches.permute(1,2,0,3,4).contiguous()
        patches = patches.contiguous().view(-1, 3, PATCH_SIZE, PATCH_SIZE)
        for j, patch in enumerate(patches):
            torchvision.utils.save_image(patch, root_dir + "/patchwise_dataset/train/" + LABELS[labels[0].item()] + "/image_" + str(i) + "patch_" + str(j) + ".JPG")
        i += 1

    # Validation images
    i = 0
    for inputs, labels in tqdm(val_data_loader):
        input = inputs[0]
        # ImageWise
        patches = input.unfold(1, PATCH_SIZE, PATCH_SIZE).unfold(2, PATCH_SIZE, PATCH_SIZE)
        patches = patches.permute(1,2,0,3,4).contiguous()
        patches = patches.contiguous().view(-1, 3, PATCH_SIZE, PATCH_SIZE)
        for j, patch in enumerate(patches):
            torchvision.utils.save_image(patch, root_dir + "/imagewise_dataset/validation/" + LABELS[labels[0].item()] + "/image_" + str(i) + "patch_" + str(j) + ".JPG")
        
        # PatchWise
        patches = input.unfold(1, PATCH_SIZE, STRIDE).unfold(2, PATCH_SIZE, STRIDE)
        patches = patches.permute(1,2,0,3,4).contiguous()
        patches = patches.contiguous().view(-1, 3, PATCH_SIZE, PATCH_SIZE)
        for j, patch in enumerate(patches):
            torchvision.utils.save_image(patch, root_dir + "/patchwise_dataset/validation/" + LABELS[labels[0].item()] + "/image_" + str(i) + "patch_" + str(j) + ".JPG")
        i += 1

    # Test images - TODO
    i = 0
    for inputs, labels in tqdm(test_data_loader):
        input = inputs[0]
        # ImageWise
        patches = input.unfold(1, PATCH_SIZE, PATCH_SIZE).unfold(2, PATCH_SIZE, PATCH_SIZE)
        patches = patches.permute(1,2,0,3,4).contiguous()
        patches = patches.contiguous().view(-1, 3, PATCH_SIZE, PATCH_SIZE)
        for j, patch in enumerate(patches):
            torchvision.utils.save_image(patch, root_dir + "/imagewise_dataset/test/" + LABELS[labels[0].item()] + "/image_" + str(i) + "patch_" + str(j) + ".JPG")
        
        # PatchWise
        patches = input.unfold(1, PATCH_SIZE, STRIDE).unfold(2, PATCH_SIZE, STRIDE)
        patches = patches.permute(1,2,0,3,4).contiguous()
        patches = patches.contiguous().view(-1, 3, PATCH_SIZE, PATCH_SIZE)
        for j, patch in enumerate(patches):
            torchvision.utils.save_image(patch, root_dir + "/patchwise_dataset/test/" + LABELS[labels[0].item()] + "/image_" + str(i) + "patch_" + str(j) + ".JPG")
        i += 1
