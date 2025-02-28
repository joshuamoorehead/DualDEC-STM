# dual_decoder_cae/nas/combined_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class NASCombinedLoss(nn.Module):
    """
    Combined loss function for dual decoder architecture in NAS.
    Based on the DualDecoderLoss in dual_decoder_cae.
    """
    def __init__(self, center_weight=0.4, noise_weight=0.6):
        super(NASCombinedLoss, self).__init__()
        self.center_weight = center_weight
        self.noise_weight = noise_weight
        
        # Register Sobel filters for edge detection (from your NoiseLoss)
        self.register_buffer('sobel_x', torch.FloatTensor([[-1, 0, 1],
                                                          [-2, 0, 2],
                                                          [-1, 0, 1]]).view(1, 1, 3, 3))
        self.register_buffer('sobel_y', torch.FloatTensor([[-1, -2, -1],
                                                          [0, 0, 0],
                                                          [1, 2, 1]]).view(1, 1, 3, 3))
    
    def compute_noise_loss(self, noise_output, target):
        """Compute loss for noise decoder based on your NoiseLoss"""
        # Basic reconstruction loss
        mse_loss = F.mse_loss(noise_output, target)
        
        # Frequency domain loss (simplified for NAS)
        freq_output = torch.fft.fft2(noise_output)
        freq_target = torch.fft.fft2(target)
        freq_loss = F.mse_loss(torch.abs(freq_output), torch.abs(freq_target))
        
        # Edge preservation loss using Sobel filters
        if hasattr(self, 'sobel_x') and hasattr(self, 'sobel_y'):
            output_grad_x = F.conv2d(noise_output, self.sobel_x, padding=1)
            output_grad_y = F.conv2d(noise_output, self.sobel_y, padding=1)
            target_grad_x = F.conv2d(target, self.sobel_x, padding=1)
            target_grad_y = F.conv2d(target, self.sobel_y, padding=1)
            
            grad_loss = (F.mse_loss(output_grad_x, target_grad_x) + 
                         F.mse_loss(output_grad_y, target_grad_y)) / 2
            
            return mse_loss + 0.1 * freq_loss + 0.2 * grad_loss
        
        return mse_loss + 0.1 * freq_loss
    
    def compute_center_loss(self, center_output, target):
        """Compute loss for center decoder based on your CenterLoss"""
        # Basic reconstruction loss
        mse_loss = F.mse_loss(center_output, target)
        
        # Compute centroids (simplified from your implementation)
        def compute_centroid(x):
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
        
        try:
            output_centroids = compute_centroid(center_output)
            target_centroids = compute_centroid(target)
            
            # Position loss
            position_loss = F.mse_loss(output_centroids, target_centroids)
            
            return mse_loss + 0.2 * position_loss
        except Exception as e:
            # Fallback to just MSE if centroid calculation fails
            return mse_loss
    
    def forward(self, center_output, noise_output, target):
        """
        Calculate combined loss for center and noise outputs
        
        Args:
            center_output (Tensor): Output from center decoder
            noise_output (Tensor): Output from noise decoder
            target (Tensor): Target/original image
            
        Returns:
            total_loss (Tensor): Combined weighted loss
            center_loss (Tensor): Center decoder loss
            noise_loss (Tensor): Noise decoder loss
        """
        # Center loss
        center_loss = self.compute_center_loss(center_output, target)
        
        # Noise loss
        noise_loss = self.compute_noise_loss(noise_output, target)
        
        # Combined loss
        total_loss = self.center_weight * center_loss + self.noise_weight * noise_loss
        
        return total_loss, center_loss, noise_loss