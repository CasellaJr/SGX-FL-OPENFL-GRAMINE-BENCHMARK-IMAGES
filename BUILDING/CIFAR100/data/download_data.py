import torch
from torchvision import datasets
from torchvision import transforms
transform=None
train_data, val_data = (datasets.CIFAR100('data', train=train, download=True, transform=transform)
for train in (True, False))
X_train_tot, y_train_tot = train_data.data, train_data.targets
X_valid_tot, y_valid_tot = val_data.data, val_data.targets