import torch
from torchvision import datasets
from torchvision import transforms
transform=None
train_data = datasets.SVHN('data', "train", download=True, transform=transform)
val_data = datasets.SVHN('data', "test", download=True, transform=transform)
X_train_tot, y_train_tot = train_data.data, train_data.labels
X_valid_tot, y_valid_tot = val_data.data, val_data.labels