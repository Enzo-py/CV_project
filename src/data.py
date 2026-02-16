import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torchvision
import torchvision.transforms as transforms

from config import DATA_DIR

from torch.utils.data import DataLoader


def get_data(BATCH_SIZE):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(45),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    trainset = torchvision.datasets.CIFAR100(root=DATA_DIR, train=True, download=True, transform=transform_train)
    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR100(root=DATA_DIR, train=False, download=True, transform=transform_test)
    test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    return train_loader, test_loader

def get_ood_data(BATCH_SIZE):
    transform_ood = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    # SVHN (OOD)
    ood_set = torchvision.datasets.SVHN(
        root=DATA_DIR, 
        split='test', 
        download=True, 
        transform=transform_ood
    )

    return DataLoader(ood_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
