import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels//2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels//2, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        attention = self.conv(x)
        return x * attention

class CenteringBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 
                              kernel_size=3, stride=2, 
                              padding=1, output_padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
        self.attention = SpatialAttention(out_channels)
        
    def forward(self, x):
        x = self.block(x)
        x = self.attention(x)
        return x

class CenterDecoder(nn.Module):
    def __init__(self, latent_dim=10, initial_size=4):
        super().__init__()
        
        self.initial_size = initial_size
        hidden_dim = 64
        
        # Project latent vector
        self.latent_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim * initial_size * initial_size),
            nn.LeakyReLU(0.2)
        )
        
        # Centering decoder blocks
        self.decoder_blocks = nn.ModuleList([
            CenteringBlock(hidden_dim, hidden_dim//2),
            CenteringBlock(hidden_dim//2, hidden_dim//4)
        ])
        
        # Position encoding
        self.position_encoding = self.create_position_encoding(hidden_dim//4)
        
        # Final convolution
        self.final_conv = nn.Sequential(
            nn.Conv2d(hidden_dim//4 * 2, hidden_dim//4, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_dim//4, 1, kernel_size=1),
            nn.Tanh()
        )
        
        # Center prediction
        self.center_pred = nn.Sequential(
            nn.Linear(latent_dim, 2),
            nn.Sigmoid()
        )
        
    def create_position_encoding(self, channels):
        """Create fixed position encodings"""
        pe = torch.zeros(channels, 17, 17)  # For 17x17 output
        for i in range(17):
            for j in range(17):
                for k in range(channels):
                    if k % 2 == 0:
                        pe[k, i, j] = torch.sin(torch.tensor(i / (10000 ** (k/channels))))
                    else:
                        pe[k, i, j] = torch.cos(torch.tensor(i / (10000 ** ((k-1)/channels))))
        return nn.Parameter(pe, requires_grad=False)
    
    def forward(self, z):
        # Predict center coordinates
        center = self.center_pred(z)
        
        # Project and reshape latent vector
        x = self.latent_proj(z)
        x = x.view(-1, 64, self.initial_size, self.initial_size)
        
        # Process through decoder blocks
        for block in self.decoder_blocks:
            x = block(x)
        
        # Add position encoding
        pos_encoding = self.position_encoding.unsqueeze(0).expand(x.size(0), -1, -1, -1)
        x = torch.cat([x, pos_encoding], dim=1)
        
        # Generate final output
        output = self.final_conv(x)
        
        return output, center

class CenteringAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.query_conv = nn.Conv2d(channels, channels//8, kernel_size=1)
        self.key_conv = nn.Conv2d(channels, channels//8, kernel_size=1)
        self.value_conv = nn.Conv2d(channels, channels, kernel_size=1)
        
    def forward(self, x):
        batch_size, C, H, W = x.size()
        
        # Generate Q, K, V
        Q = self.query_conv(x).view(batch_size, -1, H*W)
        K = self.key_conv(x).view(batch_size, -1, H*W)
        V = self.value_conv(x).view(batch_size, -1, H*W)
        
        # Compute attention scores
        attention = F.softmax(torch.bmm(Q.transpose(1, 2), K), dim=2)
        
        # Apply attention to values
        out = torch.bmm(V, attention.transpose(1, 2))
        out = out.view(batch_size, C, H, W)
        
        return out + x  # Residual connection
