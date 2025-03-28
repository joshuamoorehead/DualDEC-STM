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
            # Load the full-sized image
            image = Image.open(image_path).convert('L')
            print(f"Original image size for {os.path.basename(image_path)}: {image.size}")
            
            # Convert to tensor without any resizing
            image_tensor = transforms.ToTensor()(image)
            
            # Apply only normalization if transform contains it
            if self.transform:
                # Extract only normalization from the transform
                for t in self.transform.transforms:
                    if isinstance(t, transforms.Normalize):
                        image_tensor = t(image_tensor)
            
            print(f"After tensor conversion: {image_tensor.shape}")
            
            # Extract 16x16 patches from the full-sized image
            patches = self.extract_random_patches(image_tensor)
            self.patches.extend(patches)

    def extract_random_patches(self, image):
        patches = []
        h, w = image.shape[1:] if len(image.shape) == 3 else image.shape
        
        # For extracting exactly 16x16 patches
        patch_size = self.patch_size
        
        for _ in range(self.patches_per_image):
            # We need at least patch_size/2 pixels from the edge in each direction
            # For a 16x16 patch, we need 8 pixels of padding
            padding = patch_size // 2
            
            # Ensure we have valid range for patch centers
            if h <= 2*padding or w <= 2*padding:
                print(f"Warning: Image too small ({h}x{w}) to extract {patch_size}x{patch_size} patches")
                continue
            
            # Select center points with enough padding for the patch
            i = random.randint(padding, h - padding - 1)
            j = random.randint(padding, w - padding - 1)
            
            # Extract exactly 16x16 patch
            # For even-sized patches (like 16x16), we need to handle centers carefully
            # Left side of center gets 8 pixels, right side gets 8 pixels (total 16)
            patch = image[:, i-padding:i+padding, j-padding:j+padding]
            
            # Check that patch has the expected size
            if patch.shape[1] != patch_size or patch.shape[2] != patch_size:
                print(f"Warning: Got patch of size {patch.shape} instead of ({1}, {patch_size}, {patch_size})")
                continue
                
            # Create centered version (same as original for now, model will learn to center)
            centered_patch = patch.clone()
            
            patches.append((patch, centered_patch))

        return patches

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        original_patch, centered_patch = self.patches[idx]
        return original_patch, centered_patch  # Returns a tuple of tensors

def load_data_automated(data_root="../Data", batch_size=None, transform=None, 
                       patches_per_image=None, lattice_type='Cubic', patch_size=16):
    """Non-interactive version of load_data for automated configuration runs"""
    folder_map = {
        'Cubic': ('trainCubic', 'validationCubic', 'testCubic'),
        'BCC': ('trainBCC', 'validationBCC', 'testBCC'),
        'FCC': ('trainFCC', 'validationFCC', 'testFCC')
    }
    
    # Ensure we're not applying resize in the transform
    safe_transform = None
    if transform is not None:
        # Create a new transform without any resize operations
        safe_transforms = []
        for t in transform.transforms:
            if not isinstance(t, transforms.Resize):
                safe_transforms.append(t)
        
        if safe_transforms:
            safe_transform = transforms.Compose(safe_transforms)
    
    train_folder, val_folder, test_folder = folder_map[lattice_type]
    train_data_folder = os.path.join(data_root, train_folder)
    val_data_folder = os.path.join(data_root, val_folder)
    test_data_folder = os.path.join(data_root, test_folder)

    print(f"Loading training data from {train_data_folder}")
    print(f"Loading validation data from {val_data_folder}")
    print(f"Loading testing data from {test_data_folder}")

    train_data = DualPurposeDataset(train_data_folder, transform=safe_transform, 
                                   patches_per_image=patches_per_image, patch_size=patch_size)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, 
                            num_workers=4, pin_memory=True)

    val_data = DualPurposeDataset(val_data_folder, transform=safe_transform, 
                                 patches_per_image=patches_per_image, patch_size=patch_size)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, 
                          num_workers=4, pin_memory=True)

    test_data = DualPurposeDataset(test_data_folder, transform=safe_transform, 
                                  patches_per_image=patches_per_image, patch_size=patch_size)
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