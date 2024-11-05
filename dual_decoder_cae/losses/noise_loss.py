import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NoiseLoss(nn.Module):
    def __init__(self, freq_weight=0.3, gradient_weight=0.2):
        super().__init__()
        self.freq_weight = freq_weight
        self.gradient_weight = gradient_weight
        
        # Sobel filters for edge detection
        self.register_buffer('sobel_x', torch.FloatTensor([[-1, 0, 1], 
                                                         [-2, 0, 2], 
                                                         [-1, 0, 1]]).view(1, 1, 3, 3))
        self.register_buffer('sobel_y', torch.FloatTensor([[-1, -2, -1], 
                                                         [0, 0, 0], 
                                                         [1, 2, 1]]).view(1, 1, 3, 3))
    
    def compute_frequency_loss(self, output, target):
        """Compute loss in frequency domain"""
        # FFT transform
        output_fft = torch.fft.fft2(output)
        target_fft = torch.fft.fft2(target)
        
        # Compute magnitude spectrum
        output_magnitude = torch.abs(output_fft)
        target_magnitude = torch.abs(target_fft)
        
        # Focus on high-frequency components (noise)
        freq_loss = F.mse_loss(output_magnitude, target_magnitude)
        
        return freq_loss
    
    def compute_gradient_loss(self, output, target):
        """Compute gradient-based loss for edge preservation"""
        # Apply Sobel filters
        output_grad_x = F.conv2d(output, self.sobel_x, padding=1)
        output_grad_y = F.conv2d(output, self.sobel_y, padding=1)
        target_grad_x = F.conv2d(target, self.sobel_x, padding=1)
        target_grad_y = F.conv2d(target, self.sobel_y, padding=1)
        
        # Compute gradient loss
        grad_loss_x = F.mse_loss(output_grad_x, target_grad_x)
        grad_loss_y = F.mse_loss(output_grad_y, target_grad_y)
        
        return (grad_loss_x + grad_loss_y) / 2
    
    def compute_local_variance_loss(self, output, target):
        """Compute loss based on local variance differences"""
        kernel_size = 3
        padding = kernel_size // 2
        
        # Compute local means
        kernel = torch.ones(1, 1, kernel_size, kernel_size).to(output.device) / (kernel_size ** 2)
        local_mean_output = F.conv2d(output, kernel, padding=padding)
        local_mean_target = F.conv2d(target, kernel, padding=padding)
        
        # Compute local variances
        local_var_output = F.conv2d(output ** 2, kernel, padding=padding) - local_mean_output ** 2
        local_var_target = F.conv2d(target ** 2, kernel, padding=padding) - local_mean_target ** 2
        
        # Compare local variances
        variance_loss = F.mse_loss(local_var_output, local_var_target)
        
        return variance_loss
    
    def forward(self, output, target):
        # Basic reconstruction loss
        reconstruction_loss = F.mse_loss(output, target)
        
        # Frequency domain loss
        freq_loss = self.compute_frequency_loss(output, target)
        
        # Gradient loss for edge preservation
        gradient_loss = self.compute_gradient_loss(output, target)
        
        # Local variance loss
        variance_loss = self.compute_local_variance_loss(output, target)
        
        # Combine all losses
        total_loss = (reconstruction_loss + 
                     self.freq_weight * freq_loss + 
                     self.gradient_weight * gradient_loss + 
                     0.1 * variance_loss)
        
        return total_loss
