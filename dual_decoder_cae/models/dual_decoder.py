import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedEncoder(nn.Module):
    def __init__(self, latent_dim=10):
        super().__init__()
        self.conv_layers = nn.Sequential(
            # First block
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second block - increased channels
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third block
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Calculate flattened size based on input patch size (17x17)
        self.flat_size = 64 * 2 * 2
        
        self.fc_layers = nn.Sequential(
            nn.Linear(self.flat_size, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, latent_dim)
        )
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, self.flat_size)
        return self.fc_layers(x)

class NoiseDecoder(nn.Module):
    def __init__(self, latent_dim=10):
        super().__init__()
        self.initial_size = 2 * 2 * 64
        
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, self.initial_size),
            nn.LeakyReLU(0.2)
        )
        
        self.conv_transpose = nn.Sequential(
            # First block
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(32),
            
            # Second block
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(16),
            
            # Final block - focus on noise reduction
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh()  # Normalized output for noise reduction
        )
    
    def forward(self, z):
        z = self.fc(z)
        z = z.view(-1, 64, 2, 2)
        return self.conv_transpose(z)

class CenteringDecoder(nn.Module):
    def __init__(self, latent_dim=10):
        super().__init__()
        self.initial_size = 2 * 2 * 64
        
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, self.initial_size),
            nn.LeakyReLU(0.2)
        )
        
        self.conv_transpose = nn.Sequential(
            # First block with attention
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(32),
            
            # Second block with attention
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(16),
            
            # Final block - focus on structural features
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh()  # Normalized output for feature enhancement
        )
        
    def forward(self, z):
        z = self.fc(z)
        z = z.view(-1, 64, 2, 2)
        return self.conv_transpose(z)

class DualDecoderAE(nn.Module):
    def __init__(self, latent_dim=10):
        super().__init__()
        self.encoder = EnhancedEncoder(latent_dim)
        self.noise_decoder = NoiseDecoder(latent_dim)
        self.center_decoder = CenteringDecoder(latent_dim)
        
    def forward(self, x, return_latent=False):
        latent = self.encoder(x)
        noise_reduced = self.noise_decoder(latent)
        centered = self.center_decoder(latent)
        
        if return_latent:
            return noise_reduced, centered, latent
        return noise_reduced, centered
