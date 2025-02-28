# dual_decoder_cae/nas/darts.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from dual_decoder_cae.nas.search_space import MixedOp, MixedActivation, MixedUpsampling

class NASEncoder(nn.Module):
    """
    Encoder with learnable architecture parameters for NAS,
    based on the EnhancedEncoder structure from your existing model
    """
    def __init__(self, in_channels=1, init_features=32, latent_dim=10):
        super(NASEncoder, self).__init__()
        
        # Initial convolution
        self.init_conv = nn.Conv2d(in_channels, init_features, kernel_size=3, padding=1)
        
        # First block
        self.mixed_op1 = MixedOp(init_features, init_features*2)
        self.mixed_act1 = MixedActivation()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second block - increased channels
        self.mixed_op2 = MixedOp(init_features*2, init_features*4)
        self.mixed_act2 = MixedActivation()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Third block
        self.mixed_op3 = MixedOp(init_features*4, init_features*8)
        self.mixed_act3 = MixedActivation()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate flattened size based on input patch size (default 17x17)
        # After 3 max pooling operations (2x2), size is reduced to 2x2
        self.flat_size = init_features*8 * 2 * 2
        
        # FC layers
        self.fc1 = nn.Linear(self.flat_size, 128)
        self.fc_act = MixedActivation()
        self.fc2 = nn.Linear(128, latent_dim)
        
    def forward(self, x):
        # Convolutional layers
        x = self.init_conv(x)
        
        x = self.mixed_op1(x)
        x = self.mixed_act1(x)
        x = self.pool1(x)
        
        x = self.mixed_op2(x)
        x = self.mixed_act2(x)
        x = self.pool2(x)
        
        x = self.mixed_op3(x)
        x = self.mixed_act3(x)
        x = self.pool3(x)
        
        # Flatten and FC layers
        x = x.view(-1, self.flat_size)
        x = self.fc1(x)
        x = self.fc_act(x)
        x = self.fc2(x)
        
        return x


class NASCenterDecoder(nn.Module):
    """
    Center Decoder with learnable architecture parameters for NAS,
    based on your CenteringDecoder structure
    """
    def __init__(self, latent_dim=10, init_features=32, output_channels=1):
        super(NASCenterDecoder, self).__init__()
        
        self.initial_size = 2  # Initial feature map size for 17x17 input patches
        hidden_dim = 64
        
        # Project latent vector
        self.fc1 = nn.Linear(latent_dim, 128)
        self.act1 = MixedActivation()
        self.fc2 = nn.Linear(128, hidden_dim * self.initial_size * self.initial_size)
        self.act2 = MixedActivation()
        
        # Upsampling blocks with mixed operations
        self.up1 = MixedUpsampling(hidden_dim, hidden_dim//2)
        self.act_up1 = MixedActivation()
        
        self.up2 = MixedUpsampling(hidden_dim//2, hidden_dim//4)
        self.act_up2 = MixedActivation()
        
        self.up3 = MixedUpsampling(hidden_dim//4, hidden_dim//8)
        self.act_up3 = MixedActivation()
        
        # Final convolution
        self.final_conv = nn.Sequential(
            nn.Conv2d(hidden_dim//8, output_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )
        
    def forward(self, z):
        # Project and reshape latent vector
        x = self.fc1(z)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        
        # Reshape to initial feature map
        x = x.view(-1, 64, self.initial_size, self.initial_size)
        
        # Upsampling path
        x = self.up1(x)
        x = self.act_up1(x)
        
        x = self.up2(x)
        x = self.act_up2(x)
        
        x = self.up3(x)
        x = self.act_up3(x)
        
        # Final convolution
        x = self.final_conv(x)
        
        return x


class NASNoiseDecoder(nn.Module):
    """
    Noise Decoder with learnable architecture parameters for NAS,
    based on your NoiseDecoder structure
    """
    def __init__(self, latent_dim=10, init_features=32, output_channels=1):
        super(NASNoiseDecoder, self).__init__()
        
        self.initial_size = 2  # Initial feature map size for 17x17 input patches
        hidden_dim = 64
        
        # Project latent vector
        self.fc1 = nn.Linear(latent_dim, 128)
        self.act1 = MixedActivation()
        self.fc2 = nn.Linear(128, hidden_dim * self.initial_size * self.initial_size)
        self.act2 = MixedActivation()
        
        # Upsampling blocks with mixed operations
        self.up1 = MixedUpsampling(hidden_dim, hidden_dim//2)
        self.act_up1 = MixedActivation()
        
        self.up2 = MixedUpsampling(hidden_dim//2, hidden_dim//4)
        self.act_up2 = MixedActivation()
        
        self.up3 = MixedUpsampling(hidden_dim//4, hidden_dim//8)
        self.act_up3 = MixedActivation()
        
        # Final convolution
        self.final_conv = nn.Sequential(
            nn.Conv2d(hidden_dim//8, output_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )
        
        # Noise attention module
        self.noise_attention = nn.Sequential(
            nn.Conv2d(hidden_dim//8, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, z):
        # Project and reshape latent vector
        x = self.fc1(z)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        
        # Reshape to initial feature map
        x = x.view(-1, 64, self.initial_size, self.initial_size)
        
        # Upsampling path
        x = self.up1(x)
        x = self.act_up1(x)
        
        x = self.up2(x)
        x = self.act_up2(x)
        
        x = self.up3(x)
        x = self.act_up3(x)
        
        # Compute noise attention mask
        attention = self.noise_attention(x)
        
        # Final reconstruction with attention
        output = self.final_conv(x)
        cleaned = output * attention
        
        return cleaned


class NASDualDecoderAE(nn.Module):
    """
    Complete Dual Decoder Autoencoder with NAS, combining
    the encoder and both decoders
    """
    def __init__(self, in_channels=1, init_features=32, latent_dim=10, output_channels=1):
        super(NASDualDecoderAE, self).__init__()
        
        # NAS Encoder
        self.encoder = NASEncoder(in_channels, init_features, latent_dim)
        
        # NAS Dual Decoders
        self.center_decoder = NASCenterDecoder(latent_dim, init_features, output_channels)
        self.noise_decoder = NASNoiseDecoder(latent_dim, init_features, output_channels)
        
    def forward(self, x, return_latent=True):
        # Encode
        z = self.encoder(x)
        
        # Decode with dual decoders
        center_output = self.center_decoder(z)
        noise_output = self.noise_decoder(z)
        
        if return_latent:
            return center_output, noise_output, z
        return center_output, noise_output