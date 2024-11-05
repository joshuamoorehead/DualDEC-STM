import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde

class LatentAnalyzer:
    def __init__(self, writer=None):
        self.writer = writer
        self.pca = PCA(n_components=3)
        self.history = []
        
    def collect_latent_vectors(self, model, dataloader, device):
        """Collect latent vectors for analysis"""
        model.eval()
        latent_vectors = []
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    original = batch[0].to(device)
                else:
                    original = batch.to(device)
                    
                # Get latent vectors
                _, _, latent = model(original, return_latent=True)
                latent_vectors.append(latent.cpu())
                
        return torch.cat(latent_vectors, dim=0)
    
    def analyze_distribution(self, latent_vectors, epoch, tag_prefix):
        """Analyze latent space distribution"""
        # Convert to numpy
        latent_np = latent_vectors.numpy()
        
        # Compute basic statistics
        mean = np.mean(latent_np, axis=0)
        std = np.std(latent_np, axis=0)
        
        # Plot distribution
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        
        # Individual dimension distributions
        for i in range(min(4, latent_np.shape[1])):
            row = i // 2
            col = i % 2
            sns.histplot(latent_np[:, i], ax=axes[row, col])
            axes[row, col].set_title(f'Dimension {i+1}')
            
        plt.tight_layout()
        if self.writer:
            self.writer.add_figure(f'{tag_prefix}/latent_distributions', fig, epoch)
        plt.close()
        
        return {'mean': mean, 'std': std}
    
    def analyze_temporal_evolution(self, epoch):
        """Analyze how latent space evolves over training"""
        if len(self.history) < 2:
            return
            
        # Compare current distribution with previous
        current = self.history[-1]
        previous = self.history[-2]
        
        displacement = np.mean(np.abs(current - previous))
        
        if self.writer:
            self.writer.add_scalar('latent/displacement', displacement, epoch)
    
    def visualize_cluster_structure(self, latent_vectors, epoch, tag_prefix):
        """Visualize cluster structure using t-SNE"""
        # Reduce dimensionality with PCA first
        pca_vectors = self.pca.fit_transform(latent_vectors.numpy())
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        tsne_vectors = tsne.fit_transform(pca_vectors)
        
        # Create density plot
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Calculate the point density
        xy = np.vstack([tsne_vectors[:, 0], tsne_vectors[:, 1]])
        z = gaussian_kde(xy)(xy)
        
        # Sort the points by density
        idx = z.argsort()
        x, y, z = tsne_vectors[idx, 0], tsne_vectors[idx, 1], z[idx]
        
        # Create the plot
        scatter = ax.scatter(x, y, c=z, s=50, alpha=0.5, cmap='viridis')
        plt.colorbar(scatter)
        
        ax.set_title(f'Latent Space Structure (Epoch {epoch})')
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        
        if self.writer:
            self.writer.add_figure(f'{tag_prefix}/latent_structure', fig, epoch)
        plt.close()
        
    def analyze_latent_space_coverage(self, latent_vectors):
        """Analyze how well the latent space is being utilized"""
        latent_np = latent_vectors.numpy()
        
        # Compute activation statistics
        activation_rate = np.mean(np.abs(latent_np) > 0.1, axis=0)
        
        # Compute correlation matrix
        correlation = np.corrcoef(latent_np.T)
        
        return {
            'activation_rates': activation_rate,
            'correlation_matrix': correlation
        }
        
    def run_full_analysis(self, model, dataloader, device, epoch, tag_prefix):
        """Run complete latent space analysis"""
        # Collect vectors
        latent_vectors = self.collect_latent_vectors(model, dataloader, device)
        
        # Store for temporal analysis
        self.history.append(latent_vectors.numpy())
        if len(self.history) > 10:  # Keep only last 10 epochs
            self.history.pop(0)
        
        # Run all analyses
        distribution_stats = self.analyze_distribution(latent_vectors, epoch, tag_prefix)
        self.analyze_temporal_evolution(epoch)
        self.visualize_cluster_structure(latent_vectors, epoch, tag_prefix)
        coverage_stats = self.analyze_latent_space_coverage(latent_vectors)
        
        return {
            'distribution': distribution_stats,
            'coverage': coverage_stats
        }
