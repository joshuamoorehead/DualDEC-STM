# dual_decoder_cae/utils/metrics.py
import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy import fftpack

def calculate_ssim(pred, target, window_size=11, full=False):
    """
    Calculate structural similarity index measure (SSIM) between prediction and target
    
    Args:
        pred (Tensor): Predicted image [B, C, H, W]
        target (Tensor): Target image [B, C, H, W]
        window_size (int): Size of the Gaussian window
        full (bool): Whether to return the full SSIM image or just the mean
        
    Returns:
        float: SSIM value
    """
    # Convert to numpy for SSIM calculation
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    
    # Calculate SSIM for each image in batch
    ssim_values = []
    for p, t in zip(pred_np, target_np):
        p_img = p.squeeze()  # Remove channel dimension
        t_img = t.squeeze()
        ssim_val = ssim(p_img, t_img, data_range=t_img.max() - t_img.min())
        ssim_values.append(ssim_val)
    
    return np.mean(ssim_values)

def calculate_psnr(pred, target):
    """
    Calculate peak signal-to-noise ratio (PSNR) between prediction and target
    
    Args:
        pred (Tensor): Predicted image [B, C, H, W]
        target (Tensor): Target image [B, C, H, W]
        
    Returns:
        float: PSNR value
    """
    # Ensure values are in range [0, 1]
    pred = torch.clamp(pred, 0, 1)
    target = torch.clamp(target, 0, 1)
    
    # Calculate MSE
    mse = F.mse_loss(pred, target).item()
    
    # Handle case where MSE is 0 (perfect prediction)
    if mse == 0:
        return float('inf')
    
    # Calculate PSNR
    max_pixel = 1.0
    psnr = 20 * torch.log10(torch.tensor(max_pixel) / torch.sqrt(torch.tensor(mse)))
    
    return psnr.item()


class DualMetrics:
    def __init__(self, device='cuda'):
        self.device = device
        
    def compute_noise_metrics(self, denoised, original):
        """Compute metrics specific to noise reduction"""
        with torch.no_grad():
            # Convert to numpy for some calculations
            denoised_np = denoised.cpu().numpy()
            original_np = original.cpu().numpy()
            
            # Basic MSE
            mse = F.mse_loss(denoised, original).item()
            
            # SSIM
            ssim_values = []
            for d, o in zip(denoised_np, original_np):
                ssim_val = ssim(d.squeeze(), o.squeeze(), data_range=1.0)
                ssim_values.append(ssim_val)
            avg_ssim = np.mean(ssim_values)
            
            # Frequency domain analysis
            fft_denoised = np.abs(fftpack.fft2(denoised_np))
            fft_original = np.abs(fftpack.fft2(original_np))
            freq_mse = np.mean((fft_denoised - fft_original) ** 2)
            
            return {
                'noise_mse': mse,
                'noise_ssim': avg_ssim,
                'freq_mse': freq_mse
            }
    
    def compute_centering_metrics(self, centered, original):
        """Compute metrics specific to centering quality"""
        with torch.no_grad():
            # Basic MSE
            mse = F.mse_loss(centered, original).item()
            
            # Compute centroids
            def compute_centroid(x):
                weights = torch.softmax(x.view(x.size(0), -1), dim=1)
                h, w = x.size(2), x.size(3)
                pos_y, pos_x = torch.meshgrid(torch.arange(h), torch.arange(w))
                pos_x = pos_x.to(x.device).float()
                pos_y = pos_y.to(x.device).float()
                weighted_x = (weights * pos_x.view(-1)).sum(dim=1)
                weighted_y = (weights * pos_y.view(-1)).sum(dim=1)
                return torch.stack([weighted_x, weighted_y], dim=1)
            
            centered_centroids = compute_centroid(centered)
            original_centroids = compute_centroid(original)
            centroid_distance = torch.norm(centered_centroids - original_centroids, dim=1).mean().item()
            
            return {
                'center_mse': mse,
                'centroid_distance': centroid_distance
            }
    
    def compute_all_metrics(self, noise_output, center_output, target):
        """Compute all metrics for both decoders"""
        noise_metrics = self.compute_noise_metrics(noise_output, target)
        center_metrics = self.compute_centering_metrics(center_output, target)
        
        # Combine metrics
        return {**noise_metrics, **center_metrics}