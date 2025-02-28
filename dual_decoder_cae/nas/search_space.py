# dual_decoder_cae/nas/search_space.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MixedOp(nn.Module):
    """
    Mixed operation for DARTS that implements a weighted combination of candidate operations.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(MixedOp, self).__init__()
        self.ops = nn.ModuleList()
        # Define candidate operations for convolutional layers
        self.ops.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(out_channels)
        ))  # Standard Conv3x3
        
        self.ops.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2, stride=stride),
            nn.BatchNorm2d(out_channels)
        ))  # Conv5x5
        
        self.ops.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=2, dilation=2, stride=stride),
            nn.BatchNorm2d(out_channels)
        ))  # Dilated Conv
        
        # Depthwise separable convolution
        self.ops.append(nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=stride, groups=in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        ))
        
        # Weights for architectural parameters
        self.weights = nn.Parameter(torch.randn(len(self.ops)))
        
    def forward(self, x):
        # Weighted sum of all operations
        weights = F.softmax(self.weights, dim=0)
        return sum(w * op(x) for w, op in zip(weights, self.ops))


class MixedActivation(nn.Module):
    """
    Mixed activation function layer for DARTS - allows searching for the optimal activation function.
    Based on your Magellan application mentioning Swish, Mish, GELU.
    """
    def __init__(self):
        super(MixedActivation, self).__init__()
        
        # Define list of activation functions
        # Using custom lambda functions so we don't need to import from activation_layers.py
        
        # Regular activations
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.2)
        
        # Swish activation: x * sigmoid(x)
        self.swish = lambda x: x * torch.sigmoid(x)
        
        # Mish activation: x * tanh(softplus(x))
        self.mish = lambda x: x * torch.tanh(F.softplus(x))
        
        # GELU activation
        self.gelu = lambda x: 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0/torch.pi)) * 
                                                       (x + 0.044715 * torch.pow(x, 3))))
        
        # Store all activations in a list for easy access
        self.activations = [self.relu, self.leaky_relu, self.swish, self.mish, self.gelu]
        
        # Weights for architectural parameters
        self.weights = nn.Parameter(torch.randn(len(self.activations)))
        
    def forward(self, x):
        # Weighted sum of all activation functions
        weights = F.softmax(self.weights, dim=0)
        return sum(w * act(x) for w, act in zip(weights, self.activations))


class MixedUpsampling(nn.Module):
    """
    Mixed upsampling layer for DARTS in the decoder - allows searching for optimal upsampling method.
    """
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(MixedUpsampling, self).__init__()
        self.ops = nn.ModuleList()
        
        # Transposed convolution
        self.ops.append(nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=4, stride=2, padding=1
        ))
        
        # Bilinear upsampling + Conv
        self.ops.append(nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        ))
        
        # Nearest upsampling + Conv
        self.ops.append(nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode='nearest'),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        ))
        
        # PixelShuffle upsampling
        self.ops.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels * scale_factor**2, kernel_size=3, padding=1),
            nn.PixelShuffle(scale_factor),
            nn.BatchNorm2d(out_channels)
        ))
        
        # Weights for architectural parameters
        self.weights = nn.Parameter(torch.randn(len(self.ops)))
        
    def forward(self, x):
        # Weighted sum of all operations
        weights = F.softmax(self.weights, dim=0)
        return sum(w * op(x) for w, op in zip(weights, self.ops))


class SpatialAttentionMixed(nn.Module):
    """
    Mixed attention mechanisms for spatial attention in decoders
    """
    def __init__(self, channels):
        super(SpatialAttentionMixed, self).__init__()
        
        # Basic channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channels // 4, channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        # Weights for architectural parameters
        self.weights = nn.Parameter(torch.randn(2))
        
    def forward(self, x):
        weights = F.softmax(self.weights, dim=0)
        
        # Channel attention
        channel_att = self.channel_attention(x)
        channel_out = x * channel_att
        
        # Spatial attention
        spatial_att = self.spatial_attention(x)
        spatial_out = x * spatial_att
        
        # Weighted combination
        return weights[0] * channel_out + weights[1] * spatial_out