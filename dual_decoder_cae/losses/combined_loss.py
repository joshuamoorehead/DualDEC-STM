import torch
import torch.nn as nn
import torch.nn.functional as F

class DualDecoderLoss(nn.Module):
    def __init__(self, noise_weight=0.6, center_weight=0.4):
        super().__init__()
        self.noise_weight = noise_weight
        self.center_weight = center_weight
        
    def compute_noise_loss(self, noise_output, target):
        # Resize target to match noise_output if sizes differ
        if noise_output.shape != target.shape:
            target = F.interpolate(target, size=noise_output.shape[2:],
                                  mode='bilinear', align_corners=False)
        # Basic reconstruction loss
        mse_loss = F.mse_loss(noise_output, target)
        # Add frequency domain loss for noise reduction
        freq_output = torch.fft.fft2(noise_output)
        freq_target = torch.fft.fft2(target)
        freq_loss = F.mse_loss(torch.abs(freq_output), torch.abs(freq_target))
        return mse_loss + 0.1 * freq_loss
        
    def compute_center_loss(self, center_output, target):
        # Resize target to match center_output if sizes differ
        if center_output.shape != target.shape:
            target = F.interpolate(target, size=center_output.shape[2:],
                                  mode='bilinear', align_corners=False)
        # Basic reconstruction loss
        mse_loss = F.mse_loss(center_output, target)
        
        # Compute centroids
        def compute_centroid(x):
            # Normalize and compute weighted average of positions
            weights = F.softmax(x.view(x.size(0), -1), dim=1)
            h, w = x.size(2), x.size(3)
            pos_y, pos_x = torch.meshgrid(torch.arange(h, device=x.device), 
                                         torch.arange(w, device=x.device), 
                                         indexing='ij')
            pos_x = pos_x.float()
            pos_y = pos_y.float()
            weighted_x = (weights * pos_x.view(-1)).sum(dim=1)
            weighted_y = (weights * pos_y.view(-1)).sum(dim=1)
            return torch.stack([weighted_x, weighted_y], dim=1)
            
        output_centroids = compute_centroid(center_output)
        target_centroids = compute_centroid(target)
        
        # Centroid alignment loss
        center_loss = F.mse_loss(output_centroids, target_centroids)
        return mse_loss + 0.2 * center_loss
        
    def forward(self, noise_output, center_output, target):
        noise_loss = self.compute_noise_loss(noise_output, target)
        center_loss = self.compute_center_loss(center_output, target)
        # Combine losses with weights
        total_loss = self.noise_weight * noise_loss + self.center_weight * center_loss
        return total_loss, noise_loss, center_loss