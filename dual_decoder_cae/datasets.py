import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import random
import numpy as np
import torch.nn.functional as F

class DualPurposeDataset(Dataset):
    def __init__(self, root_dir='../Data/', transform=None, patches_per_image=100, patch_size=16):
        self.root_dir = root_dir
        self.transform = transform
        self.patches_per_image = patches_per_image
        self.patch_size = patch_size
        self.image_paths = [os.path.join(root_dir, file) 
                           for file in os.listdir(root_dir) 
                           if file.endswith('.png')]
        self.patches = []
        self._extract_random_patches()

    def _extract_random_patches(self):
        for image_path in self.image_paths:
            image = Image.open(image_path).convert('L')
            if self.transform:
                image = self.transform(image)
            patches = self.extract_random_patches(image)
            self.patches.extend(patches)

    def extract_random_patches(self, image):
        patches = []
        h, w = image.shape[1:] if len(image.shape) == 3 else image.shape
        half_size = self.patch_size // 2

        for _ in range(self.patches_per_image):
            i = random.randint(half_size, h - half_size - 1)
            j = random.randint(half_size, w - half_size - 1)
            
            # Extract patch
            patch = image[:, i-half_size:i+half_size+1, j-half_size:j+half_size+1]
            
            # Create centered version (same as original for now, the model will learn to center)
            centered_patch = patch.clone()
            
            # Both patches should be tensors
            if torch.is_tensor(patch) and torch.is_tensor(centered_patch):
                patches.append((patch, centered_patch))

        return patches

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        original_patch, centered_patch = self.patches[idx]
        return original_patch, centered_patch  # Returns a tuple of tensors

def load_data_automated(data_root="../Data", batch_size=None, transform=None, 
                       patches_per_image=None, lattice_type='Cubic'):
    """Non-interactive version of load_data for automated configuration runs"""
    folder_map = {
        'Cubic': ('trainCubic', 'validationCubic', 'testCubic'),
        'BCC': ('trainBCC', 'validationBCC', 'testBCC'),
        'FCC': ('trainFCC', 'validationFCC', 'testFCC')
    }
    
    train_folder, val_folder, test_folder = folder_map[lattice_type]
    train_data_folder = os.path.join(data_root, train_folder)
    val_data_folder = os.path.join(data_root, val_folder)
    test_data_folder = os.path.join(data_root, test_folder)

    print(f"Loading training data from {train_data_folder}")
    print(f"Loading validation data from {val_data_folder}")
    print(f"Loading testing data from {test_data_folder}")

    train_data = DualPurposeDataset(train_data_folder, transform=transform, 
                                   patches_per_image=patches_per_image)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, 
                            num_workers=4, pin_memory=True)

    val_data = DualPurposeDataset(val_data_folder, transform=transform, 
                                 patches_per_image=patches_per_image)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, 
                          num_workers=4, pin_memory=True)

    test_data = DualPurposeDataset(test_data_folder, transform=transform, 
                                  patches_per_image=patches_per_image)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, 
                           num_workers=4, pin_memory=True)

    # Print dataset statistics
    total_patches = len(train_data)
    num_images = len(train_data.image_paths)
    print(f"Number of images in dataset: {num_images}")
    print(f"Patches per image: {patches_per_image}")
    print(f"Total number of patches extracted: {total_patches}")
    print(f"Average patches per image: {total_patches / num_images:.2f}")
    print(f"Patches per batch: {batch_size}")
    print(f"Number of batches per epoch: {len(train_loader)}")

    return train_loader, val_loader, test_loader, lattice_type

__all__ = ['DualPurposeDataset', 'load_data_automated']
