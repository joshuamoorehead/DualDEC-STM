import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
import seaborn as sns

class DualVisualizer:
    def __init__(self, writer):
        self.writer = writer
        
    def visualize_batch(self, original, noise_output, center_output, epoch, tag_prefix):
        """Visualize a batch of images with both decoder outputs"""
        # Create a comparison grid
        batch_size = original.size(0)
        num_examples = min(4, batch_size)  # Show up to 4 examples
        
        fig, axes = plt.subplots(num_examples, 3, figsize=(12, 4*num_examples))
        plt.suptitle(f'Epoch {epoch} Reconstructions')
        
        for i in range(num_examples):
            # Original
            axes[i, 0].imshow(original[i].cpu().squeeze(), cmap='gray')
            axes[i, 0].set_title('Original')
            axes[i, 0].axis('off')
            
            # Noise reduction output
            axes[i, 1].imshow(noise_output[i].cpu().squeeze(), cmap='gray')
            axes[i, 1].set_title('Noise Reduced')
            axes[i, 1].axis('off')
            
            # Centering output
            axes[i, 2].imshow(center_output[i].cpu().squeeze(), cmap='gray')
            axes[i, 2].set_title('Centered')
            axes[i, 2].axis('off')
        
        self.writer.add_figure(f'{tag_prefix}/reconstructions', fig, epoch)
        plt.close()
        
    def visualize_latent_space(self, latent_vectors, epoch, tag_prefix):
        """Visualize the latent space distribution"""
        # PCA for dimensionality reduction
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        latent_2d = pca.fit_transform(latent_vectors.cpu().numpy())
        
        # Create scatter plot
        fig, ax = plt.subplots(figsize=(10, 10))
        scatter = ax.scatter(latent_2d[:, 0], latent_2d[:, 1], 
                           c=range(len(latent_2d)), cmap='viridis')
        plt.colorbar(scatter)
        ax.set_title(f'Latent Space Distribution (Epoch {epoch})')
        ax.set_xlabel('First Principal Component')
        ax.set_ylabel('Second Principal Component')
        
        self.writer.add_figure(f'{tag_prefix}/latent_space', fig, epoch)
        plt.close()
        
    def visualize_attention_maps(self, center_output, original, epoch, tag_prefix):
        """Visualize attention maps for centering decoder"""
        # Compute attention maps
        attention = torch.abs(center_output - original)
        attention = F.interpolate(attention, scale_factor=2, mode='bilinear')
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        for i in range(min(4, attention.size(0))):
            # Original
            axes[0, i].imshow(original[i].cpu().squeeze(), cmap='gray')
            axes[0, i].set_title('Original')
            axes[0, i].axis('off')
            
            # Attention map
            axes[1, i].imshow(attention[i].cpu().squeeze(), cmap='hot')
            axes[1, i].set_title('Attention Map')
            axes[1, i].axis('off')
            
        self.writer.add_figure(f'{tag_prefix}/attention_maps', fig, epoch)
        plt.close()
