import torch
import torch.nn as nn
import torch.nn.functional as F

class CenterLoss(nn.Module):
    def __init__(self, structural_weight=0.4, position_weight=0.3):
        super().__init__()
        self.structural_weight = structural_weight
        self.position_weight = position_weight
        
    def compute_centroid(self, x):
        """Compute the centroid of activation"""
        batch_size = x.size(0)
        h, w = x.size(2), x.size(3)
        
        # Create coordinate grid
        y_coords = torch.linspace(-1, 1, h).view(1, 1, -1, 1).expand(batch_size, 1, h, w).to(x.device)
        x_coords = torch.linspace(-1, 1, w).view(1, 1, 1, -1).expand(batch_size, 1, h, w).to(x.device)
        
        # Compute weights from activations
        weights = F.softmax(x.view(batch_size, -1), dim=1).view(batch_size, 1, h, w)
        
        # Compute weighted centroids
        y_center = (weights * y_coords).sum(dim=(2, 3))
        x_center = (weights * x_coords).sum(dim=(2, 3))
        
        return torch.cat([x_center, y_center], dim=1)
    
    def compute_structural_loss(self, output, target):
        """Compute loss based on structural similarity"""
        # Compute local mean
        kernel_size = 5
        padding = kernel_size // 2
        kernel = torch.ones(1, 1, kernel_size, kernel_size).to(output.device) / (kernel_size ** 2)
        
        mu_x = F.conv2d(output, kernel, padding=padding)
        mu_y = F.conv2d(target, kernel, padding=padding)
        
        mu_x_sq = mu_x ** 2
        mu_y_sq = mu_y ** 2
        mu_xy = mu_x * mu_y
        
        # Compute local variance and covariance
        sigma_x_sq = F.conv2d(output ** 2, kernel, padding=padding) - mu_x_sq
        sigma_y_sq = F.conv2d(target ** 2, kernel, padding=padding) - mu_y_sq
        sigma_xy = F.conv2d(output * target, kernel, padding=padding) - mu_xy
        
        # SSIM constants
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        # Compute SSIM
        ssim = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))
        
        return 1 - ssim.mean()
    
    def compute_position_loss(self, output, target):
        """Compute loss based on feature positions"""
        output_centroid = self.compute_centroid(output)
        target_centroid = self.compute_centroid(target)
        
        # Position loss
        position_loss = F.mse_loss(output_centroid, target_centroid)
        
        return position_loss
    
    def compute_symmetry_loss(self, output):
        """Compute loss based on radial symmetry around centroid"""
        batch_size = output.size(0)
        centroid = self.compute_centroid(output)
        
        # Create radial distance map from centroid
        h, w = output.size(2), output.size(3)
        y_coords = torch.linspace(-1, 1, h).view(1, 1, -1, 1).expand(batch_size, 1, h, w).to(output.device)
        x_coords = torch.linspace(-1, 1, w).view(1, 1, 1, -1).expand(batch_size, 1, h, w).to(output.device)
        
        # Compute distances from centroid
        x_dist = x_coords - centroid[:, 0].view(-1, 1, 1, 1)
        y_dist = y_coords - centroid[:, 1].view(-1, 1, 1, 1)
        distances = torch.sqrt(x_dist**2 + y_dist**2)
        
        # Compare values at similar distances
        symmetry_loss = torch.std(output * (distances < 0.5), dim=(2, 3)).mean()
        
        return symmetry_loss
    
    def forward(self, output, target):
        # Basic reconstruction loss
        reconstruction_loss = F.mse_loss(output, target)
        
        # Structural similarity loss
        structural_loss = self.compute_structural_loss(output, target)
        
        # Position loss
        position_loss = self.compute_position_loss(output, target)
        
        # Symmetry loss
        symmetry_loss = self.compute_symmetry_loss(output)
        
        # Combine losses
        total_loss = (reconstruction_loss + 
                     self.structural_weight * structural_loss +
                     self.position_weight * position_loss +
                     0.1 * symmetry_loss)
        
        return total_loss

class CombinedFeatureLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def compute_feature_correlation_loss(self, output, target):
        """Compute loss based on feature correlations"""
        # Flatten spatial dimensions
        output_flat = output.view(output.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        
        # Compute correlation matrices
        output_corr = torch.matmul(output_flat, output_flat.transpose(-2, -1))
        target_corr = torch.matmul(target_flat, target_flat.transpose(-2, -1))
        
        # Normalize correlation matrices
        output_norm = torch.norm(output_corr, p='fro')
        target_norm = torch.norm(target_corr, p='fro')
        
        if output_norm > 0 and target_norm > 0:
            output_corr = output_corr / output_norm
            target_corr = target_corr / target_norm
            
            # Compute Frobenius norm of difference
            correlation_loss = torch.norm(output_corr - target_corr, p='fro')
        else:
            correlation_loss = torch.tensor(0.0).to(output.device)
        
        return correlation_loss
        
    def forward(self, output_features, target_features):
        feature_loss = 0
        
        # Compute loss at each feature level if multiple feature levels are provided
        if isinstance(output_features, (list, tuple)):
            for out_feat, tar_feat in zip(output_features, target_features):
                feature_loss += self.compute_feature_correlation_loss(out_feat, tar_feat)
            feature_loss /= len(output_features)
        else:
            feature_loss = self.compute_feature_correlation_loss(output_features, target_features)
            
        return feature_loss
