# dual_decoder_cae/nas/evaluator.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import copy
import time

# Import metrics module from your project
from dual_decoder_cae.utils.metrics import calculate_ssim, calculate_psnr

class DARTSEvaluator:
    """
    Evaluator for DARTS neural architecture search.
    Uses bi-level optimization:
    - Inner loop: train model weights
    - Outer loop: update architecture parameters
    """
    def __init__(self, model, train_dataset, val_dataset, 
                 combined_loss_fn, batch_size=64, w_lr=0.001, alpha_lr=0.0001):
        self.model = model
        self.batch_size = batch_size
        
        # Create data loaders
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Loss function
        self.criterion = combined_loss_fn
        
        # Create optimizers
        # 1. For model weights
        self.w_optimizer = optim.Adam(self.get_model_weights(), lr=w_lr)
        
        # 2. For architecture parameters (alphas)
        self.alpha_optimizer = optim.Adam(self.get_arch_parameters(), lr=alpha_lr)
        
        # Schedulers can be added if needed
        
    def get_model_weights(self):
        """Get all model weights except architecture parameters"""
        arch_params = set(self.get_arch_parameters())
        return [p for p in self.model.parameters() if p not in arch_params]
    
    def get_arch_parameters(self):
        """Get all architecture parameters (weights in MixedOp, MixedActivation, etc.)"""
        arch_params = []
        for name, module in self.model.named_modules():
            if hasattr(module, 'weights') and isinstance(module.weights, nn.Parameter):
                arch_params.append(module.weights)
        return arch_params
    
    def train_step(self, x, target):
        """Inner optimization loop (update model weights)"""
        # Forward pass
        center_output, noise_output, z = self.model(x)
        
        # Calculate loss - assuming target is the original patch and
        # we want to reconstruct it with both decoders
        loss, center_loss, noise_loss = self.criterion(center_output, noise_output, target)
        
        # Backward pass and optimize
        self.w_optimizer.zero_grad()
        loss.backward()
        self.w_optimizer.step()
        
        return loss.item(), center_output, noise_output
    
    def valid_step(self, x, target):
        """Outer optimization loop (update architecture parameters)"""
        # Forward pass
        center_output, noise_output, z = self.model(x)
        
        # Calculate loss
        loss, center_loss, noise_loss = self.criterion(center_output, noise_output, target)
        
        # Backward pass and optimize architecture parameters
        self.alpha_optimizer.zero_grad()
        loss.backward()
        self.alpha_optimizer.step()
        
        return loss.item(), center_output, noise_output
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        train_loss = 0.0
        device = next(self.model.parameters()).device
        
        # Split train data into two parts for the two-level optimization
        train_data_list = list(self.train_loader)
        split = len(train_data_list) // 2
        
        # Progress tracking
        pbar = tqdm(total=len(train_data_list), desc=f"Epoch {epoch}")
        
        # First part: update weights
        for i, (x, target) in enumerate(train_data_list[:split]):
            x, target = x.to(device), target.to(device)
            loss, _, _ = self.train_step(x, target)
            train_loss += loss
            pbar.update(1)
            pbar.set_postfix({'train_loss': f'{loss:.4f}'})
        
        # Second part: update architecture parameters
        for i, (x, target) in enumerate(train_data_list[split:]):
            x, target = x.to(device), target.to(device)
            loss, _, _ = self.valid_step(x, target)
            train_loss += loss
            pbar.update(1)
            pbar.set_postfix({'train_loss': f'{loss:.4f}', 'arch_update': True})
        
        pbar.close()
        return train_loss / len(train_data_list)
    
    def validate(self):
        """Validate model on validation dataset"""
        self.model.eval()
        val_loss = 0.0
        ssim_scores = []
        psnr_scores = []
        device = next(self.model.parameters()).device
        
        # Progress tracking
        pbar = tqdm(total=len(self.val_loader), desc="Validation")
        
        with torch.no_grad():
            for x, target in self.val_loader:
                x, target = x.to(device), target.to(device)
                
                # Forward pass
                center_output, noise_output, z = self.model(x)
                
                # Calculate loss
                loss, center_loss, noise_loss = self.criterion(center_output, noise_output, target)
                val_loss += loss.item()
                
                # Calculate metrics
                # Focus on center_output for SSIM and PSNR as it's the main reconstruction
                ssim = calculate_ssim(center_output, target)
                psnr = calculate_psnr(center_output, target)
                
                ssim_scores.append(ssim)
                psnr_scores.append(psnr)
                
                pbar.update(1)
                pbar.set_postfix({'val_loss': f'{loss.item():.4f}', 'ssim': f'{ssim:.4f}'})
        
        pbar.close()
        avg_val_loss = val_loss / len(self.val_loader)
        avg_ssim = np.mean(ssim_scores)
        avg_psnr = np.mean(psnr_scores)
        
        return avg_val_loss, avg_ssim, avg_psnr
    
    def search(self, epochs=50):
        """Main search method"""
        best_val_loss = float('inf')
        best_arch = None
        best_model_weights = None
        start_time = time.time()
        
        metrics = {
            'train_loss': [],
            'val_loss': [],
            'ssim': [],
            'psnr': []
        }
        
        # Progress tracking for entire search
        print(f"Starting NAS search for {epochs} epochs")
        
        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss, ssim, psnr = self.validate()
            
            # Store metrics
            metrics['train_loss'].append(train_loss)
            metrics['val_loss'].append(val_loss)
            metrics['ssim'].append(ssim)
            metrics['psnr'].append(psnr)
            
            # Track best architecture
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                # Save architecture parameters
                best_arch = []
                for param in self.get_arch_parameters():
                    best_arch.append(param.data.clone())
                
                # Save model weights
                best_model_weights = copy.deepcopy(self.model.state_dict())
            
            # Calculate epoch time and ETA
            epoch_time = time.time() - epoch_start
            elapsed_time = time.time() - start_time
            eta = elapsed_time / epoch * (epochs - epoch)
            
            # Format timings
            eta_h, eta_m = divmod(int(eta), 3600)
            eta_m, eta_s = divmod(eta_m, 60)
            
            # Print status with timing info
            print(f"Epoch {epoch}/{epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, " 
                  f"SSIM: {ssim:.4f}, PSNR: {psnr:.2f} "
                  f"{'[BEST]' if is_best else ''}")
            print(f"Time: {epoch_time:.2f}s | ETA: {eta_h}h {eta_m}m {eta_s}s")
            print("-" * 80)
        
        # Load best architecture parameters and weights
        for param, best_param in zip(self.get_arch_parameters(), best_arch):
            param.data.copy_(best_param)
        
        self.model.load_state_dict(best_model_weights)
        
        # Calculate total time
        total_time = time.time() - start_time
        hours, remainder = divmod(int(total_time), 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"NAS search completed in {hours}h {minutes}m {seconds}s")
        print(f"Best validation loss: {best_val_loss:.4f}")
        
        return metrics, self.model
    
    def derive_architecture(self):
        """
        Derive the final architecture by selecting the strongest operations
        This would create a fixed architecture by selecting the operations with highest weights
        """
        # To be implemented based on your specific needs
        # This would create a new model with fixed operations instead of mixed ops
        pass