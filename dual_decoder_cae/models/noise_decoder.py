import torch
import torch.nn as nn
import torch.nn.functional as F

class NoiseDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 
                              kernel_size=3, stride=2, 
                              padding=1, output_padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_channels, out_channels, 
                     kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
        
    def forward(self, x):
        return self.block(x)

class WaveletDecomposition(nn.Module):
    def __init__(self):
        super().__init__()
        # Haar wavelet kernels
        self.register_buffer('haar_wx', torch.FloatTensor([[[[ 1,  1], 
                                                           [-1, -1]]]]).div(2))
        self.register_buffer('haar_wy', torch.FloatTensor([[[[ 1, -1], 
                                                           [ 1, -1]]]]).div(2))
        
    def forward(self, x):
        # Compute wavelet coefficients
        dx = F.conv2d(x, self.haar_wx, stride=2, padding=0)
        dy = F.conv2d(x, self.haar_wy, stride=2, padding=0)
        return dx, dy

class NoiseDecoder(nn.Module):
    def __init__(self, latent_dim=10, initial_size=4):
        super().__init__()
        
        self.initial_size = initial_size
        hidden_dim = 64
        
        # Initial projection from latent space
        self.latent_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim * initial_size * initial_size),
            nn.LeakyReLU(0.2)
        )
        
        # Wavelet decomposition for frequency analysis
        self.wavelet = WaveletDecomposition()
        
        # Main decoder path
        self.decoder_blocks = nn.ModuleList([
            NoiseDecoderBlock(hidden_dim, hidden_dim//2),
            NoiseDecoderBlock(hidden_dim//2, hidden_dim//4)
        ])
        
        # Final convolution for reconstruction
        self.final_conv = nn.Sequential(
            nn.Conv2d(hidden_dim//4, 1, kernel_size=3, padding=1),
            nn.Tanh()
        )
        
        # Noise attention module
        self.noise_attention = nn.Sequential(
            nn.Conv2d(hidden_dim//4, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, z):
        # Project and reshape latent vector
        x = self.latent_proj(z)
        x = x.view(-1, 64, self.initial_size, self.initial_size)
        
        # Process through decoder blocks
        features = []
        for block in self.decoder_blocks:
            x = block(x)
            features.append(x)
        
        # Compute noise attention mask
        attention = self.noise_attention(x)
        
        # Final reconstruction with attention
        output = self.final_conv(x)
        cleaned = output * attention
        
        return cleaned

class NoiseAnalysis(nn.Module):
    def __init__(self):
        super().__init__()
        self.freq_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 1, kernel_size=1)
        )
        
    def forward(self, x):
        # Convert to frequency domain
        fft = torch.fft.fft2(x)
        magnitude = torch.abs(fft)
        phase = torch.angle(fft)
        
        # Analyze frequency components
        freq_features = self.freq_conv(magnitude.unsqueeze(1))
        return freq_features, phase
