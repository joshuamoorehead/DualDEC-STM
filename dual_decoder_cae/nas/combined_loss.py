# dual_decoder_cae/nas/combined_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class NASCombinedLoss(nn.Module):
    """
    Combined loss function for dual decoder architecture in NAS with searchable weights.
    Based on the DualDecoderLoss in dual_decoder_cae.
    """
    def __init__(self, center_weight=0.4, noise_weight=0.6, use_searchable_weights=False):
        super(NASCombinedLoss, self).__init__()
        self.center_weight = center_weight
        self.noise_weight = noise_weight
        self.use_searchable_weights = use_searchable_weights
        
        # Create Sobel filters as regular tensors (we'll move them to the right device when needed)
        self.sobel_x = torch.FloatTensor([[-1, 0, 1],
                                          [-2, 0, 2],
                                          [-1, 0, 1]]).view(1, 1, 3, 3)
        self.sobel_y = torch.FloatTensor([[-1, -2, -1],
                                          [0, 0, 0],
                                          [1, 2, 1]]).view(1, 1, 3, 3)
    
    def compute_noise_loss(self, noise_output, target):
        """Compute loss for noise decoder based on your NoiseLoss"""
        # Ensure target has same shape as output
        if noise_output.shape != target.shape:
            target = F.interpolate(target, size=noise_output.shape[2:], 
                                 mode='bilinear', align_corners=False)
        
        # Basic reconstruction loss
        mse_loss = F.mse_loss(noise_output, target)
        
        # Frequency domain loss (simplified for NAS)
        freq_output = torch.fft.fft2(noise_output)
        freq_target = torch.fft.fft2(target)
        freq_loss = F.mse_loss(torch.abs(freq_output), torch.abs(freq_target))
        
        # Edge preservation loss using Sobel filters
        # Explicitly move Sobel filters to the same device as the input
        device = noise_output.device
        sobel_x = self.sobel_x.to(device)
        sobel_y = self.sobel_y.to(device)
        
        output_grad_x = F.conv2d(noise_output, sobel_x, padding=1)
        output_grad_y = F.conv2d(noise_output, sobel_y, padding=1)
        target_grad_x = F.conv2d(target, sobel_x, padding=1)
        target_grad_y = F.conv2d(target, sobel_y, padding=1)
        
        grad_loss = (F.mse_loss(output_grad_x, target_grad_x) + 
                     F.mse_loss(output_grad_y, target_grad_y)) / 2
        
        return mse_loss + 0.1 * freq_loss + 0.2 * grad_loss
    
    def compute_center_loss(self, center_output, target):
        """Compute loss for center decoder based on your CenterLoss"""
        # Ensure target has same shape as output
        if center_output.shape != target.shape:
            target = F.interpolate(target, size=center_output.shape[2:], 
                                 mode='bilinear', align_corners=False)
            
        # Basic reconstruction loss
        mse_loss = F.mse_loss(center_output, target)
        
        # Compute centroids (simplified from your implementation)
        def compute_centroid(x):
            batch_size = x.size(0)
            h, w = x.size(2), x.size(3)
            
            # Create coordinate grid directly on the correct device
            y_coords = torch.linspace(-1, 1, h, device=x.device).view(1, 1, -1, 1).expand(batch_size, 1, h, w)
            x_coords = torch.linspace(-1, 1, w, device=x.device).view(1, 1, 1, -1).expand(batch_size, 1, h, w)
            
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
            print(f"Warning: centroid calculation failed: {e}")
            # Fallback to just MSE if centroid calculation fails
            return mse_loss
    
    def forward(self, center_output, noise_output, target, model=None):
        """
        Calculate combined loss for center and noise outputs
        
        Args:
            center_output (Tensor): Output from center decoder
            noise_output (Tensor): Output from noise decoder
            target (Tensor): Target/original image
            model (NASDualDecoderAE, optional): The model with searchable weights
            
        Returns:
            total_loss (Tensor): Combined weighted loss
            center_loss (Tensor): Center decoder loss
            noise_loss (Tensor): Noise decoder loss
        """
        # Ensure target is on the same device as outputs
        target = target.to(center_output.device)
        
        # Center loss
        center_loss = self.compute_center_loss(center_output, target)
        
        # Noise loss
        noise_loss = self.compute_noise_loss(noise_output, target)
        
        # Get weights - either fixed or from the model's searchable weights
        if self.use_searchable_weights and model is not None and hasattr(model, 'get_loss_weights'):
            center_weight, noise_weight = model.get_loss_weights()
        else:
            center_weight = self.center_weight
            noise_weight = self.noise_weight
        
        # Combined loss
        total_loss = center_weight * center_loss + noise_weight * noise_loss
        
        return total_loss, center_loss, noise_loss