import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

class RotatedMNIST(Dataset):
    def __init__(self, mnist_dataset, rotation_range=(-180, 180)):
        self.mnist_dataset = mnist_dataset
        self.rotation_range = rotation_range

    def __len__(self):
        return len(self.mnist_dataset)

    def __getitem__(self, idx):
        image, label = self.mnist_dataset[idx]
        angle = np.random.uniform(*self.rotation_range)
        rotated_image = transforms.functional.rotate(image, angle)
        return rotated_image, label, angle
    
    