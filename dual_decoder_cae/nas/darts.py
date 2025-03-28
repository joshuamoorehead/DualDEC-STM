# dual_decoder_cae/nas/darts.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from dual_decoder_cae.nas.search_space import MixedOp, MixedActivation,MixedLossWeights, MixedUpsampling, MixedChannelWidth, MixedDropout, MixedLatentDim, MixedNormalization, MixedSkipConnection

class NASEncoder(nn.Module):
    """
    Encoder with learnable architecture parameters for NAS,
    based on the EnhancedEncoder structure but with additional
    searchable parameters for latent dimension, channel width,
    normalization type, and regularization.
    """
    def __init__(self, in_channels=1, init_features=32, latent_dims=None, 
                 width_mults=None, dropout_rates=None):
        super(NASEncoder, self).__init__()
        
        if latent_dims is None:
            latent_dims = [8, 16, 32, 64, 128]
        if width_mults is None:
            width_mults = [0.5, 0.75, 1.0, 1.25, 1.5]
        if dropout_rates is None:
            dropout_rates = [0.0, 0.1, 0.2, 0.3, 0.5]
            
        # Channel widths for each block
        self.width1 = MixedChannelWidth(init_features, width_mults)
        self.width2 = MixedChannelWidth(init_features*2, width_mults)  # Approximate, will adjust
        self.width3 = MixedChannelWidth(init_features*4, width_mults)  # Approximate, will adjust
        
        # Initial convolution
        self.init_conv = nn.Conv2d(in_channels, init_features, kernel_size=3, padding=1)
        self.init_norm = MixedNormalization(init_features)
        
        # First block
        self.mixed_op1 = MixedOp(init_features, init_features*2)
        self.mixed_norm1 = MixedNormalization(init_features*2)
        self.mixed_act1 = MixedActivation()
        self.dropout1 = MixedDropout(dropout_rates)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second block - increased channels
        # Will use the width determined by width1
        self.mixed_op2 = lambda x: MixedOp(x.size(1), init_features*4).to(x.device)(x)   # Dynamic channels
        self.mixed_norm2 = lambda x: MixedNormalization(x.size(1)).to(x.device)(x)     # Dynamic normalization
        self.mixed_act2 = MixedActivation()
        self.dropout2 = MixedDropout(dropout_rates)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Third block
        # Will use the width determined by width2
        self.mixed_op3 = lambda x: MixedOp(x.size(1), init_features*8).to(x.device)(x)  # Dynamic channels
        self.mixed_norm3 = lambda x: MixedNormalization(x.size(1)).to(x.device)(x)     # Dynamic normalization
        self.mixed_act3 = MixedActivation()
        self.dropout3 = MixedDropout(dropout_rates)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # FC layers
        # The flat_size will be determined dynamically in forward()
        self.fc1 = nn.Linear(init_features*8*2*2, 128)  # Placeholder, will be replaced in forward
        self.fc_norm = MixedNormalization(128, is_2d=False)
        self.fc_act = MixedActivation()
        self.fc_dropout = MixedDropout(dropout_rates)
        
        # Latent dimension search
        self.latent_dim_search = MixedLatentDim(128, latent_dims)
        
    def forward(self, x):
        # Initial conv
        x = self.init_conv(x)
        x = self.init_norm(x)
        
        # Block 1
        x = self.mixed_op1(x)
        x = self.mixed_norm1(x)
        x = self.mixed_act1(x)
        x = self.dropout1(x)
        x = self.pool1(x)
        
        # Apply width adjustment for block 2
        x = self.width1(x)
        
        # Block 2
        x = self.mixed_op2(x)
        x = self.mixed_norm2(x)
        x = self.mixed_act2(x)
        x = self.dropout2(x)
        x = self.pool2(x)
        
        # Apply width adjustment for block 3
        x = self.width2(x)
        
        # Block 3
        x = self.mixed_op3(x)
        x = self.mixed_norm3(x)
        x = self.mixed_act3(x)
        x = self.dropout3(x)
        x = self.pool3(x)
        
        # Apply width adjustment for FC
        x = self.width3(x)
        
        # Flatten for FC layers
        flat_size = x.size(1) * x.size(2) * x.size(3)
        x = x.view(-1, flat_size)
        
        # Replace FC1 with correct dimensions if needed
        if not hasattr(self, '_fc1_initialized') or self._fc1_initialized != flat_size:
            self.fc1 = nn.Linear(flat_size, 128).to(x.device)
            self._fc1_initialized = flat_size
        
        # FC layers
        x = self.fc1(x)
        x = self.fc_norm(x)
        x = self.fc_act(x)
        x = self.fc_dropout(x)
        
        # Apply latent dimension search
        x = self.latent_dim_search(x)
        
        return x

class NASCenterDecoder(nn.Module):
    """
    Center Decoder with enhanced searchable architecture parameters
    """
    def __init__(self, max_latent_dim=128, init_features=32, output_channels=1,
                 width_mults=None, dropout_rates=None):
        super(NASCenterDecoder, self).__init__()
        
        if width_mults is None:
            width_mults = [0.5, 0.75, 1.0, 1.25, 1.5]
        if dropout_rates is None:
            dropout_rates = [0.0, 0.1, 0.2, 0.3, 0.5]
            
        self.initial_size = 2  # Initial feature map size for 17x17 input patches
        hidden_dim = 64
        self.output_channels = output_channels
        
        # Channel width search
        self.width1 = MixedChannelWidth(hidden_dim, width_mults)
        self.width2 = MixedChannelWidth(hidden_dim//2, width_mults)
        self.width3 = MixedChannelWidth(hidden_dim//4, width_mults)
        
        # Project latent vector with dynamic latent dim
        self.fc1 = nn.Linear(max_latent_dim, 128)  # Use max possible latent dim
        self.fc_norm1 = MixedNormalization(128, is_2d=False)
        self.act1 = MixedActivation()
        self.dropout1 = MixedDropout(dropout_rates)
        
        self.fc2 = nn.Linear(128, hidden_dim * self.initial_size * self.initial_size)
        self.fc_norm2 = MixedNormalization(hidden_dim * self.initial_size * self.initial_size, is_2d=False)
        self.act2 = MixedActivation()
        self.dropout2 = MixedDropout(dropout_rates)
        
        # Upsampling blocks with mixed operations
        self.up1 = MixedUpsampling(hidden_dim, hidden_dim//2)
        self.norm_up1 = lambda x: MixedNormalization(x.size(1)).to(x.device)(x) # Dynamic normalization
        self.act_up1 = MixedActivation()
        self.dropout_up1 = MixedDropout(dropout_rates)
        
        self.up2 = lambda x: MixedUpsampling(x.size(1), hidden_dim//4).to(x.device)(x)  # Dynamic channels
        self.norm_up2 = lambda x: MixedNormalization(x.size(1)).to(x.device)(x)  # Dynamic normalization
        self.act_up2 = MixedActivation()
        self.dropout_up2 = MixedDropout(dropout_rates)
        
        self.up3 = lambda x: MixedUpsampling(x.size(1), hidden_dim//8).to(x.device)(x)  # Dynamic channels
        self.norm_up3 = lambda x: MixedNormalization(x.size(1)).to(x.device)(x)  # Dynamic normalization
        self.act_up3 = MixedActivation()
        self.dropout_up3 = MixedDropout(dropout_rates)
        
        # Skip connections search
        self.skip1 = MixedSkipConnection(hidden_dim//2)
        self.skip2 = lambda x, y: MixedSkipConnection(x.size(1)).to(x.device)(x, y)  # Dynamic channels
        self.skip3 = lambda x, y: MixedSkipConnection(x.size(1)).to(x.device)(x, y)  # Dynamic channels
        
        # Final convolution - will be created dynamically in forward pass
        self.final_act = nn.Tanh()
        
    def forward(self, z):
        # Handle variable latent dimensions by only using the needed part
        # Assuming z might be zero-padded to max_latent_dim
        
        # Project and reshape latent vector
        x = self.fc1(z)
        x = self.fc_norm1(x)
        x = self.act1(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.fc_norm2(x)
        x = self.act2(x)
        x = self.dropout2(x)
        
        # Reshape to initial feature map
        x = x.view(-1, 64, self.initial_size, self.initial_size)
        
        # Store for skip connections
        skip1_input = x
        
        # Upsampling path with skip connections
        x = self.up1(x)
        x = self.norm_up1(x)
        x = self.act_up1(x)
        x = self.dropout_up1(x)
        
        # Apply width and skip connection
        x_width1 = self.width1(skip1_input)
        x = self.skip1(x, x_width1)
        skip2_input = x
        
        x = self.up2(x)
        x = self.norm_up2(x)
        x = self.act_up2(x)
        x = self.dropout_up2(x)
        
        # Apply width and skip connection
        x_width2 = self.width2(skip2_input)
        x = self.skip2(x, x_width2)
        skip3_input = x
        
        x = self.up3(x)
        x = self.norm_up3(x)
        x = self.act_up3(x)
        x = self.dropout_up3(x)
        
        # Apply width and skip connection
        x_width3 = self.width3(skip3_input)
        x = self.skip3(x, x_width3)
        
        # Final convolution - create dynamically based on current channel count
        final_conv = nn.Conv2d(x.size(1), self.output_channels, kernel_size=3, padding=1).to(x.device)
        x = final_conv(x)
        x = self.final_act(x)
        
        return x
        
    def get_active_dim(self):
        """Return the currently active latent dimension size"""
        # To be implemented for architecture extraction
        if hasattr(self, 'fc1') and isinstance(self.fc1, nn.Linear):
            return self.fc1.in_features
        return None


class NASNoiseDecoder(nn.Module):
    """
    Noise Decoder with enhanced searchable architecture parameters
    """
    def __init__(self, max_latent_dim=128, init_features=32, output_channels=1,
                 width_mults=None, dropout_rates=None):
        super(NASNoiseDecoder, self).__init__()
        
        if width_mults is None:
            width_mults = [0.5, 0.75, 1.0, 1.25, 1.5]
        if dropout_rates is None:
            dropout_rates = [0.0, 0.1, 0.2, 0.3, 0.5]
            
        self.initial_size = 2  # Initial feature map size for 17x17 input patches
        hidden_dim = 64
        self.output_channels = output_channels
        
        # Channel width search
        self.width1 = MixedChannelWidth(hidden_dim, width_mults)
        self.width2 = MixedChannelWidth(hidden_dim//2, width_mults)
        self.width3 = MixedChannelWidth(hidden_dim//4, width_mults)
        
        # Project latent vector with dynamic latent dim
        self.fc1 = nn.Linear(max_latent_dim, 128)  # Use max possible latent dim
        self.fc_norm1 = MixedNormalization(128, is_2d=False)
        self.act1 = MixedActivation()
        self.dropout1 = MixedDropout(dropout_rates)
        
        self.fc2 = nn.Linear(128, hidden_dim * self.initial_size * self.initial_size)
        self.fc_norm2 = MixedNormalization(hidden_dim * self.initial_size * self.initial_size, is_2d=False)
        self.act2 = MixedActivation()
        self.dropout2 = MixedDropout(dropout_rates)
        
        # Upsampling blocks with mixed operations
        self.up1 = MixedUpsampling(hidden_dim, hidden_dim//2)
        self.norm_up1 = lambda x: MixedNormalization(x.size(1)).to(x.device)(x)  # Dynamic normalization
        self.act_up1 = MixedActivation()
        self.dropout_up1 = MixedDropout(dropout_rates)
        
        self.up2 = lambda x: MixedUpsampling(x.size(1), hidden_dim//4).to(x.device)(x)  # Dynamic channels
        self.norm_up2 = lambda x: MixedNormalization(x.size(1)).to(x.device)(x)  # Dynamic normalization
        self.act_up2 = MixedActivation()
        self.dropout_up2 = MixedDropout(dropout_rates)
        
        self.up3 = lambda x: MixedUpsampling(x.size(1), hidden_dim//8).to(x.device)(x)  # Dynamic channels
        self.norm_up3 = lambda x: MixedNormalization(x.size(1)).to(x.device)(x)  # Dynamic normalization
        self.act_up3 = MixedActivation()
        self.dropout_up3 = MixedDropout(dropout_rates)
        
        # Skip connections search
        self.skip1 = MixedSkipConnection(hidden_dim//2)
        self.skip2 = lambda x, y: MixedSkipConnection(x.size(1)).to(x.device)(x, y)  # Dynamic channels
        self.skip3 = lambda x, y: MixedSkipConnection(x.size(1)).to(x.device)(x, y)  # Dynamic channels
        
        # Final convolution - will be created dynamically in forward pass
        self.final_act = nn.Tanh()
        
    def forward(self, z):
        # Project and reshape latent vector
        x = self.fc1(z)
        x = self.fc_norm1(x)
        x = self.act1(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.fc_norm2(x)
        x = self.act2(x)
        x = self.dropout2(x)
        
        # Reshape to initial feature map
        x = x.view(-1, 64, self.initial_size, self.initial_size)
        
        # Store for skip connections
        skip1_input = x
        
        # Upsampling path with skip connections
        x = self.up1(x)
        x = self.norm_up1(x)
        x = self.act_up1(x)
        x = self.dropout_up1(x)
        
        # Apply width and skip connection
        x_width1 = self.width1(skip1_input)
        x = self.skip1(x, x_width1)
        skip2_input = x
        
        x = self.up2(x)
        x = self.norm_up2(x)
        x = self.act_up2(x)
        x = self.dropout_up2(x)
        
        # Apply width and skip connection
        x_width2 = self.width2(skip2_input)
        x = self.skip2(x, x_width2)
        skip3_input = x
        
        x = self.up3(x)
        x = self.norm_up3(x)
        x = self.act_up3(x)
        x = self.dropout_up3(x)
        
        # Apply width and skip connection
        x_width3 = self.width3(skip3_input)
        x = self.skip3(x, x_width3)
        
        # Create noise attention dynamically based on current channel count
        noise_attention = nn.Sequential(
            nn.Conv2d(x.size(1), 1, kernel_size=1),
            nn.Sigmoid()
        ).to(x.device)
        
        # Compute noise attention mask
        attention = noise_attention(x)
        
        # Final convolution - create dynamically based on current channel count
        final_conv = nn.Conv2d(x.size(1), self.output_channels, kernel_size=3, padding=1).to(x.device)
        x = final_conv(x)
        x = self.final_act(x)
        
        # Apply noise attention
        output = x
        cleaned = output * attention
        
        return cleaned
    
    def get_active_dim(self):
        """Return the currently active latent dimension size"""
        # To be implemented for architecture extraction
        if hasattr(self, 'fc1') and isinstance(self.fc1, nn.Linear):
            return self.fc1.in_features
        return None


class NASDualDecoderAE(nn.Module):
    """
    Complete Dual Decoder Autoencoder with enhanced NAS.
    This class combines the encoder and both decoders with all
    searchable architecture parameters including:
    - Latent space dimensionality
    - Model depth (implicit in the skip connections)
    - Channel widths
    - Normalization layers
    - Skip connections
    - Regularization strength (dropout)
    - Loss function weights
    """
    def __init__(self, in_channels=1, init_features=32, output_channels=1,
                 latent_dims=None, width_mults=None, dropout_rates=None):
        super(NASDualDecoderAE, self).__init__()
        
        if latent_dims is None:
            latent_dims = [8, 16, 32, 64, 128]
        if width_mults is None:
            width_mults = [0.5, 0.75, 1.0, 1.25, 1.5]
        if dropout_rates is None:
            dropout_rates = [0.0, 0.1, 0.2, 0.3, 0.5]
            
        # Store the maximum latent dimension for decoders
        self.max_latent_dim = max(latent_dims)
        
        # NAS Encoder with searchable parameters
        self.encoder = NASEncoder(
            in_channels=in_channels,
            init_features=init_features, 
            latent_dims=latent_dims,
            width_mults=width_mults,
            dropout_rates=dropout_rates
        )
        
        # NAS Dual Decoders with searchable parameters
        self.center_decoder = NASCenterDecoder(
            max_latent_dim=self.max_latent_dim,
            init_features=init_features,
            output_channels=output_channels,
            width_mults=width_mults,
            dropout_rates=dropout_rates
        )
        
        self.noise_decoder = NASNoiseDecoder(
            max_latent_dim=self.max_latent_dim,
            init_features=init_features,
            output_channels=output_channels,
            width_mults=width_mults,
            dropout_rates=dropout_rates
        )
        
        # Mixed loss weights for balancing center and noise decoder losses
        self.loss_weights = MixedLossWeights(num_losses=2)
        
    def forward(self, x, return_latent=True):
        # Encode
        z = self.encoder(x)
        
        # If z's dimension is less than max_latent_dim, pad with zeros
        if z.size(1) < self.max_latent_dim:
            padding = torch.zeros(z.size(0), self.max_latent_dim - z.size(1), 
                                 device=z.device)
            z_padded = torch.cat([z, padding], dim=1)
        else:
            z_padded = z
        
        # Decode with dual decoders
        center_output = self.center_decoder(z_padded)
        noise_output = self.noise_decoder(z_padded)
        
        if return_latent:
            return center_output, noise_output, z
        return center_output, noise_output
    
    def get_loss_weights(self):
        """Returns the current weights for center and noise losses"""
        return self.loss_weights()