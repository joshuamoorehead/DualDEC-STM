# dual_decoder_cae/nas/extract_architecture.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
from collections import Counter

from dual_decoder_cae.models.activation_layers import Swish, Mish, GELU
from dual_decoder_cae.nas.search_space import (
    MixedOp, MixedActivation, MixedUpsampling, MixedLatentDim,
    MixedNormalization, MixedChannelWidth, MixedSkipConnection, MixedDropout
)

class EnhancedFixedOperationEncoder(nn.Module):
    """
    Enhanced encoder with fixed operations extracted from NAS
    Including support for normalization, regularization, and channel widths
    """
    def __init__(self, in_channels=1, init_features=32, latent_dim=64, 
                 activation_type='relu', kernel_sizes=None, dilations=None,
                 norm_types=None, dropout_rates=None, channel_widths=None):
        super(EnhancedFixedOperationEncoder, self).__init__()
        
        # Set activation function
        if activation_type == 'relu':
            self.activation = nn.ReLU()
        elif activation_type == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2)
        elif activation_type == 'swish':
            self.activation = Swish()
        elif activation_type == 'mish':
            self.activation = Mish()
        elif activation_type == 'gelu':
            self.activation = GELU()
        else:
            raise ValueError(f"Unsupported activation type: {activation_type}")
        
        # Default kernel sizes and dilations if not provided
        if kernel_sizes is None:
            kernel_sizes = [3, 3, 3]
        if dilations is None:
            dilations = [1, 1, 1]
        if norm_types is None:
            norm_types = ['batch', 'batch', 'batch']  # BatchNorm, LayerNorm, InstanceNorm, GroupNorm
        if dropout_rates is None:
            dropout_rates = [0.0, 0.0, 0.0]
        if channel_widths is None:
            channel_widths = [2, 4, 8]  # Multipliers for init_features
        
        # Initial convolution
        self.init_conv = nn.Conv2d(in_channels, init_features, kernel_size=3, padding=1)
        self.init_norm = self._create_norm_layer(init_features, norm_types[0])
        
        # Encoder blocks with normalization and dropout
        self.block1 = nn.Sequential(
            nn.Conv2d(init_features, int(init_features * channel_widths[0]), kernel_size=kernel_sizes[0], 
                     padding=kernel_sizes[0]//2 * dilations[0], dilation=dilations[0], stride=2),
            self._create_norm_layer(int(init_features * channel_widths[0]), norm_types[0]),
            self.activation,
            nn.Dropout2d(dropout_rates[0])
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(int(init_features * channel_widths[0]), int(init_features * channel_widths[1]), 
                     kernel_size=kernel_sizes[1], 
                     padding=kernel_sizes[1]//2 * dilations[1], dilation=dilations[1], stride=2),
            self._create_norm_layer(int(init_features * channel_widths[1]), norm_types[1]),
            self.activation,
            nn.Dropout2d(dropout_rates[1])
        )
        
        self.block3 = nn.Sequential(
            nn.Conv2d(int(init_features * channel_widths[1]), int(init_features * channel_widths[2]), 
                     kernel_size=kernel_sizes[2], 
                     padding=kernel_sizes[2]//2 * dilations[2], dilation=dilations[2], stride=2),
            self._create_norm_layer(int(init_features * channel_widths[2]), norm_types[2]),
            self.activation,
            nn.Dropout2d(dropout_rates[2])
        )
        
        # Calculate feature size after 3 downsampling operations (divide by 2^3 = 8)
        self.feature_size = lambda input_size: input_size // 8
        
        # To be initialized with input size in forward method
        self.flatten_size = None
        self.fc = None
        self.latent_dim = latent_dim
        self.dropout_fc = nn.Dropout(dropout_rates[0])  # Use first dropout rate for FC
        
    def _create_norm_layer(self, num_features, norm_type):
        """Create a normalization layer based on the specified type"""
        if norm_type == 'batch':
            return nn.BatchNorm2d(num_features)
        elif norm_type == 'layer':
            return nn.GroupNorm(1, num_features)  # LayerNorm equivalent for 2D
        elif norm_type == 'instance':
            return nn.InstanceNorm2d(num_features)
        elif norm_type == 'group':
            num_groups = min(32, num_features)
            return nn.GroupNorm(num_groups, num_features)
        else:
            raise ValueError(f"Unsupported normalization type: {norm_type}")
        
    def initialize_fc(self, x):
        # Get input dimensions and create FC layer at first forward pass
        with torch.no_grad():
            # Pass through convolutional layers
            x = self.init_conv(x)
            x = self.init_norm(x)
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            
            # Calculate flattened size
            self.flatten_size = x.size(1) * x.size(2) * x.size(3)
            
            # Initialize fully connected layer
            self.fc = nn.Linear(self.flatten_size, self.latent_dim).to(x.device)
    
    def forward(self, x):
        # Initialize FC layer if not done yet
        if self.fc is None:
            self.initialize_fc(x)
            
        # Forward pass through convolutional layers
        x = self.init_conv(x)
        x = self.init_norm(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        # Flatten and project to latent space
        x = x.view(x.size(0), -1)
        x = self.dropout_fc(x)
        x = self.fc(x)
        
        return x


class EnhancedFixedOperationDecoder(nn.Module):
    """
    Enhanced decoder with fixed operations extracted from NAS
    Including support for normalization, regularization, skip connections, and channel widths
    """
    def __init__(self, latent_dim=64, init_features=32, output_channels=1,
                activation_type='relu', upsampling_types=None, norm_types=None,
                dropout_rates=None, channel_widths=None, use_skip_connections=False):
        super(EnhancedFixedOperationDecoder, self).__init__()
        
        # Set activation function
        if activation_type == 'relu':
            self.activation = nn.ReLU()
        elif activation_type == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2)
        elif activation_type == 'swish':
            self.activation = Swish()
        elif activation_type == 'mish':
            self.activation = Mish()
        elif activation_type == 'gelu':
            self.activation = GELU()
        else:
            raise ValueError(f"Unsupported activation type: {activation_type}")
        
        # Default parameters if not provided
        if upsampling_types is None:
            # Options: 'transpose', 'bilinear', 'nearest', 'pixelshuffle'
            upsampling_types = ['transpose', 'transpose', 'transpose']
        if norm_types is None:
            norm_types = ['batch', 'batch', 'batch']
        if dropout_rates is None:
            dropout_rates = [0.0, 0.0, 0.0]
        if channel_widths is None:
            channel_widths = [8, 4, 2]  # Decreasing multipliers for init_features
        
        self.init_features = init_features
        self.latent_dim = latent_dim
        self.use_skip_connections = use_skip_connections
        
        # Dropout for latent projection
        self.dropout_fc1 = nn.Dropout(dropout_rates[0])
        self.dropout_fc2 = nn.Dropout(dropout_rates[0])
        
        # Upsampling blocks with normalization and dropout
        hidden_dim = int(init_features * channel_widths[0])
        self.block1 = nn.ModuleDict({
            'upsample': self._create_upsampling_block(
                hidden_dim, int(init_features * channel_widths[1]), upsampling_types[0]),
            'norm': self._create_norm_layer(int(init_features * channel_widths[1]), norm_types[0]),
            'dropout': nn.Dropout2d(dropout_rates[0])
        })
        
        self.block2 = nn.ModuleDict({
            'upsample': self._create_upsampling_block(
                int(init_features * channel_widths[1]), int(init_features * channel_widths[2]), upsampling_types[1]),
            'norm': self._create_norm_layer(int(init_features * channel_widths[2]), norm_types[1]),
            'dropout': nn.Dropout2d(dropout_rates[1])
        })
        
        self.block3 = nn.ModuleDict({
            'upsample': self._create_upsampling_block(
                int(init_features * channel_widths[2]), init_features, upsampling_types[2]),
            'norm': self._create_norm_layer(init_features, norm_types[2]),
            'dropout': nn.Dropout2d(dropout_rates[2])
        })
        
        # Skip connection processing (if enabled)
        if use_skip_connections:
            self.skip1 = nn.Conv2d(hidden_dim, int(init_features * channel_widths[1]), kernel_size=1)
            self.skip2 = nn.Conv2d(int(init_features * channel_widths[1]), 
                                 int(init_features * channel_widths[2]), kernel_size=1)
            self.skip3 = nn.Conv2d(int(init_features * channel_widths[2]), init_features, kernel_size=1)
        
        # Final convolution to get the right number of output channels
        self.final_conv = nn.Conv2d(init_features, output_channels, kernel_size=3, padding=1)
        self.final_act = nn.Tanh()  # Typical for image output
        
        # To be initialized with input size in forward method
        self.fc1 = None
        self.fc2 = None
        self.reshape_size = None
        
    def _create_norm_layer(self, num_features, norm_type):
        """Create a normalization layer based on the specified type"""
        if norm_type == 'batch':
            return nn.BatchNorm2d(num_features)
        elif norm_type == 'layer':
            return nn.GroupNorm(1, num_features)  # LayerNorm equivalent for 2D
        elif norm_type == 'instance':
            return nn.InstanceNorm2d(num_features)
        elif norm_type == 'group':
            num_groups = min(32, num_features)
            return nn.GroupNorm(num_groups, num_features)
        else:
            raise ValueError(f"Unsupported normalization type: {norm_type}")
        
    def _create_upsampling_block(self, in_channels, out_channels, upsampling_type):
        """Create an upsampling block based on the specified type"""
        if upsampling_type == 'transpose':
            return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        elif upsampling_type == 'bilinear':
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            )
        elif upsampling_type == 'nearest':
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            )
        elif upsampling_type == 'pixelshuffle':
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2)  # scale_factor=2 -> x4 channels reduction
            )
        else:
            raise ValueError(f"Unsupported upsampling type: {upsampling_type}")
    
    def initialize_fc(self, x, input_spatial_size):
        # Initial feature map size after three 2x downsampling steps
        feature_map_size = input_spatial_size // 8
        
        # Use actual latent dimension from input rather than assumed value
        latent_dim = x.size(1)
        hidden_dim = int(self.init_features * 8)  # Assuming channel_widths[0] is 8
        
        self.reshape_size = (hidden_dim, feature_map_size, feature_map_size)
        flattened_size = hidden_dim * feature_map_size * feature_map_size
        
        # Create FC layers from actual latent size to flattened feature maps
        self.fc1 = nn.Linear(latent_dim, 256).to(x.device)
        self.fc2 = nn.Linear(256, flattened_size).to(x.device)
    
    def forward(self, x, input_spatial_size):
        # Initialize FC layers if not done yet
        if self.fc1 is None:
            self.initialize_fc(x, input_spatial_size)
            
        # Project from latent space and reshape to feature maps
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout_fc1(x)
        
        x = self.fc2(x)
        x = self.activation(x)
        x = self.dropout_fc2(x)
        
        # Reshape to initial feature map
        batch_size = x.size(0)
        x = x.view(batch_size, *self.reshape_size)
        
        # Store for skip connections
        skip1 = x if self.use_skip_connections else None
        
        # Upsampling path with skip connections
        x = self.block1['upsample'](x)
        x = self.block1['norm'](x)
        x = self.activation(x)
        x = self.block1['dropout'](x)
        
        if self.use_skip_connections and skip1 is not None:
            x = x + self.skip1(skip1)
        
        skip2 = x if self.use_skip_connections else None
        
        x = self.block2['upsample'](x)
        x = self.block2['norm'](x)
        x = self.activation(x)
        x = self.block2['dropout'](x)
        
        if self.use_skip_connections and skip2 is not None:
            x = x + self.skip2(skip2)
        
        skip3 = x if self.use_skip_connections else None
        
        x = self.block3['upsample'](x)
        x = self.block3['norm'](x)
        x = self.activation(x)
        x = self.block3['dropout'](x)
        
        if self.use_skip_connections and skip3 is not None:
            x = x + self.skip3(skip3)
        
        # Final convolution
        x = self.final_conv(x)
        x = self.final_act(x)
        
        return x


class EnhancedFixedDualDecoderAE(nn.Module):
    """
    Complete Dual Decoder Autoencoder with fixed operations
    extracted from enhanced NAS results
    """
    def __init__(self, in_channels=1, init_features=32, latent_dim=64, output_channels=1,
                encoder_activation='swish', center_decoder_activation='relu', 
                noise_decoder_activation='mish',
                encoder_kernels=None, encoder_dilations=None,
                center_upsampling=None, noise_upsampling=None,
                encoder_norm_types=None, center_norm_types=None, noise_norm_types=None,
                encoder_dropout_rates=None, center_dropout_rates=None, noise_dropout_rates=None,
                encoder_channel_widths=None, center_channel_widths=None, noise_channel_widths=None,
                center_skip_connections=False, noise_skip_connections=False):
        super(EnhancedFixedDualDecoderAE, self).__init__()
        
        # Create enhanced fixed operations encoder
        self.encoder = EnhancedFixedOperationEncoder(
            in_channels=in_channels,
            init_features=init_features,
            latent_dim=latent_dim,
            activation_type=encoder_activation,
            kernel_sizes=encoder_kernels,
            dilations=encoder_dilations,
            norm_types=encoder_norm_types,
            dropout_rates=encoder_dropout_rates,
            channel_widths=encoder_channel_widths
        )
        
        # Create enhanced fixed operations decoders
        self.center_decoder = EnhancedFixedOperationDecoder(
            latent_dim=latent_dim,
            init_features=init_features,
            output_channels=output_channels,
            activation_type=center_decoder_activation,
            upsampling_types=center_upsampling,
            norm_types=center_norm_types,
            dropout_rates=center_dropout_rates,
            channel_widths=center_channel_widths,
            use_skip_connections=center_skip_connections
        )
        
        self.noise_decoder = EnhancedFixedOperationDecoder(
            latent_dim=latent_dim,
            init_features=init_features,
            output_channels=output_channels,
            activation_type=noise_decoder_activation,
            upsampling_types=noise_upsampling,
            norm_types=noise_norm_types,
            dropout_rates=noise_dropout_rates,
            channel_widths=noise_channel_widths,
            use_skip_connections=noise_skip_connections
        )
        
    def forward(self, x):
        input_spatial_size = x.size(2)  # Assuming square inputs
        
        # Encode
        z = self.encoder(x)
        
        # Instead of using a fixed reshape size in the decoder,
        # We need to query the actual encoder output size
        encoder_output_channels = z.size(1)
        
        # Decode with dual decoders - make sure to pass the correct channel info
        center_output = self.center_decoder(z, input_spatial_size)
        noise_output = self.noise_decoder(z, input_spatial_size)
        
        return center_output, noise_output, z


def extract_enhanced_architecture(nas_model):
    """
    Extract the best architecture from a trained NAS model with enhanced parameters
    
    Args:
        nas_model: A trained NASDualDecoderAE model with enhanced search space
        
    Returns:
        dict: Configuration for best architecture
    """
    architecture = {
        'latent_dim': None,
        'encoder': {
            'activation': None,
            'kernel_sizes': [],
            'dilations': [],
            'norm_types': [],
            'dropout_rates': [],
            'channel_widths': []
        },
        'center_decoder': {
            'activation': None,
            'upsampling_types': [],
            'norm_types': [],
            'dropout_rates': [],
            'channel_widths': [],
            'use_skip_connections': False
        },
        'noise_decoder': {
            'activation': None,
            'upsampling_types': [],
            'norm_types': [],
            'dropout_rates': [],
            'channel_widths': [],
            'use_skip_connections': False
        },
        'loss_weights': {
            'center_weight': 0.5,
            'noise_weight': 0.5
        }
    }
    
    # Extract latent dimension
    for name, module in nas_model.named_modules():
        if isinstance(module, MixedLatentDim):
            weights = F.softmax(module.weights, dim=0).detach().cpu().numpy()
            best_idx = np.argmax(weights)
            architecture['latent_dim'] = module.dim_options[best_idx]
    
    # Activation function options
    activation_functions = ['relu', 'leaky_relu', 'swish', 'mish', 'gelu']
    
    # Upsampling options
    upsampling_types = ['transpose', 'bilinear', 'nearest', 'pixelshuffle']
    
    # Normalization options
    norm_types = ['batch', 'layer', 'instance', 'group']
    
    # Process encoder activations
    encoder_activations = []
    for name, module in nas_model.named_modules():
        if isinstance(module, MixedActivation) and name.startswith('encoder'):
            weights = F.softmax(module.weights, dim=0).detach().cpu().numpy()
            best_idx = np.argmax(weights)
            encoder_activations.append(activation_functions[best_idx])
    
    # Use most common activation for encoder
    if encoder_activations:
        architecture['encoder']['activation'] = Counter(encoder_activations).most_common(1)[0][0]
    
    # Process encoder kernels and dilations
    for name, module in nas_model.named_modules():
        if isinstance(module, MixedOp) and name.startswith('encoder'):
            weights = F.softmax(module.weights, dim=0).detach().cpu().numpy()
            best_idx = np.argmax(weights)
            
            # Map operation index to kernel size and dilation
            if best_idx == 0:  # Standard Conv3x3
                architecture['encoder']['kernel_sizes'].append(3)
                architecture['encoder']['dilations'].append(1)
            elif best_idx == 1:  # Conv5x5
                architecture['encoder']['kernel_sizes'].append(5)
                architecture['encoder']['dilations'].append(1)
            elif best_idx == 2:  # Dilated Conv
                architecture['encoder']['kernel_sizes'].append(3)
                architecture['encoder']['dilations'].append(2)
            elif best_idx == 3:  # Depthwise separable
                architecture['encoder']['kernel_sizes'].append(3)
                architecture['encoder']['dilations'].append(1)
    
    # Process encoder normalizations
    for name, module in nas_model.named_modules():
        if isinstance(module, MixedNormalization) and name.startswith('encoder'):
            weights = F.softmax(module.weights, dim=0).detach().cpu().numpy()
            best_idx = np.argmax(weights)
            architecture['encoder']['norm_types'].append(norm_types[best_idx])
    
    # Process encoder dropout rates
    for name, module in nas_model.named_modules():
        if isinstance(module, MixedDropout) and name.startswith('encoder'):
            weights = F.softmax(module.weights, dim=0).detach().cpu().numpy()
            best_idx = np.argmax(weights)
            architecture['encoder']['dropout_rates'].append(module.dropout_rates[best_idx])
    
    # Process encoder channel widths
    for name, module in nas_model.named_modules():
        if isinstance(module, MixedChannelWidth) and name.startswith('encoder'):
            weights = F.softmax(module.weights, dim=0).detach().cpu().numpy()
            best_idx = np.argmax(weights)
            architecture['encoder']['channel_widths'].append(module.width_options[best_idx])
    
    # Process center decoder activations
    center_activations = []
    for name, module in nas_model.named_modules():
        if isinstance(module, MixedActivation) and name.startswith('center_decoder'):
            weights = F.softmax(module.weights, dim=0).detach().cpu().numpy()
            best_idx = np.argmax(weights)
            center_activations.append(activation_functions[best_idx])
    
    # Use most common activation for center decoder
    if center_activations:
        architecture['center_decoder']['activation'] = Counter(center_activations).most_common(1)[0][0]
    
    # Process center decoder upsampling types
    for name, module in nas_model.named_modules():
        if isinstance(module, MixedUpsampling) and name.startswith('center_decoder'):
            weights = F.softmax(module.weights, dim=0).detach().cpu().numpy()
            best_idx = np.argmax(weights)
            architecture['center_decoder']['upsampling_types'].append(upsampling_types[best_idx])
    
    # Process center decoder normalizations
    for name, module in nas_model.named_modules():
        if isinstance(module, MixedNormalization) and name.startswith('center_decoder'):
            weights = F.softmax(module.weights, dim=0).detach().cpu().numpy()
            best_idx = np.argmax(weights)
            architecture['center_decoder']['norm_types'].append(norm_types[best_idx])
    
    # Process center decoder dropout rates
    for name, module in nas_model.named_modules():
        if isinstance(module, MixedDropout) and name.startswith('center_decoder'):
            weights = F.softmax(module.weights, dim=0).detach().cpu().numpy()
            best_idx = np.argmax(weights)
            architecture['center_decoder']['dropout_rates'].append(module.dropout_rates[best_idx])
    
    # Process center decoder channel widths
    for name, module in nas_model.named_modules():
        if isinstance(module, MixedChannelWidth) and name.startswith('center_decoder'):
            weights = F.softmax(module.weights, dim=0).detach().cpu().numpy()
            best_idx = np.argmax(weights)
            architecture['center_decoder']['channel_widths'].append(module.width_options[best_idx])
    
    # Process center decoder skip connections
    skip_connections_center = []
    for name, module in nas_model.named_modules():
        if isinstance(module, MixedSkipConnection) and name.startswith('center_decoder'):
            weights = F.softmax(module.weights, dim=0).detach().cpu().numpy()
            # Skip connection is active if any option other than 0 (no skip) is chosen
            skip_active = weights.argmax() > 0
            skip_connections_center.append(skip_active)
    
    # Set skip connection flag for center decoder if any are active
    architecture['center_decoder']['use_skip_connections'] = any(skip_connections_center)
    
    # Process noise decoder activations
    noise_activations = []
    for name, module in nas_model.named_modules():
        if isinstance(module, MixedActivation) and name.startswith('noise_decoder'):
            weights = F.softmax(module.weights, dim=0).detach().cpu().numpy()
            best_idx = np.argmax(weights)
            noise_activations.append(activation_functions[best_idx])
    
    # Use most common activation for noise decoder
    if noise_activations:
        architecture['noise_decoder']['activation'] = Counter(noise_activations).most_common(1)[0][0]
    
    # Process noise decoder upsampling types
    for name, module in nas_model.named_modules():
        if isinstance(module, MixedUpsampling) and name.startswith('noise_decoder'):
            weights = F.softmax(module.weights, dim=0).detach().cpu().numpy()
            best_idx = np.argmax(weights)
            architecture['noise_decoder']['upsampling_types'].append(upsampling_types[best_idx])
    
    # Process noise decoder normalizations
    for name, module in nas_model.named_modules():
        if isinstance(module, MixedNormalization) and name.startswith('noise_decoder'):
            weights = F.softmax(module.weights, dim=0).detach().cpu().numpy()
            best_idx = np.argmax(weights)
            architecture['noise_decoder']['norm_types'].append(norm_types[best_idx])
    
    # Process noise decoder dropout rates
    for name, module in nas_model.named_modules():
        if isinstance(module, MixedDropout) and name.startswith('noise_decoder'):
            weights = F.softmax(module.weights, dim=0).detach().cpu().numpy()
            best_idx = np.argmax(weights)
            architecture['noise_decoder']['dropout_rates'].append(module.dropout_rates[best_idx])
    
    # Process noise decoder channel widths
    for name, module in nas_model.named_modules():
        if isinstance(module, MixedChannelWidth) and name.startswith('noise_decoder'):
            weights = F.softmax(module.weights, dim=0).detach().cpu().numpy()
            best_idx = np.argmax(weights)
            architecture['noise_decoder']['channel_widths'].append(module.width_options[best_idx])
    
    # Process noise decoder skip connections
    skip_connections_noise = []
    for name, module in nas_model.named_modules():
        if isinstance(module, MixedSkipConnection) and name.startswith('noise_decoder'):
            weights = F.softmax(module.weights, dim=0).detach().cpu().numpy()
            # Skip connection is active if any option other than 0 (no skip) is chosen
            skip_active = weights.argmax() > 0
            skip_connections_noise.append(skip_active)
    
    # Set skip connection flag for noise decoder if any are active
    architecture['noise_decoder']['use_skip_connections'] = any(skip_connections_noise)
    
    # Extract loss weights if available
    if hasattr(nas_model, 'loss_weights'):
        weights = nas_model.get_loss_weights()
        if len(weights) >= 2:
            architecture['loss_weights']['center_weight'] = weights[0].item()
            architecture['loss_weights']['noise_weight'] = weights[1].item()
    
    return architecture


def create_enhanced_fixed_model_from_nas(nas_model, in_channels=1, init_features=32, output_channels=1):
    """
    Create a fixed model based on the best architecture found from enhanced NAS
    
    Args:
        nas_model: A trained NASDualDecoderAE model with enhanced search
        
    Returns:
        EnhancedFixedDualDecoderAE: A model with fixed operations based on NAS results
    """
    # Extract architecture configuration
    architecture = extract_enhanced_architecture(nas_model)

    # Determine the device from the NAS model
    device = next(nas_model.parameters()).device
    
    # Ensure we have complete lists for all parameters with proper defaults
    # Add defaults if lists are too short
    def ensure_list_length(lst, default_value, target_length=3):
        if lst is None:
            return [default_value] * target_length
        while len(lst) < target_length:
            lst.append(default_value)
        return lst
    
    # Apply to all architecture lists
    architecture['encoder']['kernel_sizes'] = ensure_list_length(
        architecture['encoder']['kernel_sizes'], 3)
    architecture['encoder']['dilations'] = ensure_list_length(
        architecture['encoder']['dilations'], 1)
    architecture['encoder']['norm_types'] = ensure_list_length(
        architecture['encoder']['norm_types'], 'batch')
    architecture['encoder']['dropout_rates'] = ensure_list_length(
        architecture['encoder']['dropout_rates'], 0.0)
    architecture['encoder']['channel_widths'] = ensure_list_length(
        architecture['encoder']['channel_widths'], 1.0)
    
    architecture['center_decoder']['upsampling_types'] = ensure_list_length(
        architecture['center_decoder']['upsampling_types'], 'transpose')
    architecture['center_decoder']['norm_types'] = ensure_list_length(
        architecture['center_decoder']['norm_types'], 'batch')
    architecture['center_decoder']['dropout_rates'] = ensure_list_length(
        architecture['center_decoder']['dropout_rates'], 0.0)
    architecture['center_decoder']['channel_widths'] = ensure_list_length(
        architecture['center_decoder']['channel_widths'], 1.0)
    
    architecture['noise_decoder']['upsampling_types'] = ensure_list_length(
        architecture['noise_decoder']['upsampling_types'], 'transpose')
    architecture['noise_decoder']['norm_types'] = ensure_list_length(
        architecture['noise_decoder']['norm_types'], 'batch')
    architecture['noise_decoder']['dropout_rates'] = ensure_list_length(
        architecture['noise_decoder']['dropout_rates'], 0.0)
    architecture['noise_decoder']['channel_widths'] = ensure_list_length(
        architecture['noise_decoder']['channel_widths'], 1.0)
    
    # Handle missing latent dimension with a default
    if not architecture['latent_dim']:
        architecture['latent_dim'] = 64
    
    # Create fixed model
    fixed_model = EnhancedFixedDualDecoderAE(
        in_channels=in_channels,
        init_features=init_features,
        latent_dim=architecture['latent_dim'],
        output_channels=output_channels,
        encoder_activation=architecture['encoder']['activation'] or 'relu',
        center_decoder_activation=architecture['center_decoder']['activation'] or 'relu',
        noise_decoder_activation=architecture['noise_decoder']['activation'] or 'relu',
        encoder_kernels=architecture['encoder']['kernel_sizes'],
        encoder_dilations=architecture['encoder']['dilations'],
        center_upsampling=architecture['center_decoder']['upsampling_types'],
        noise_upsampling=architecture['noise_decoder']['upsampling_types'],
        encoder_norm_types=architecture['encoder']['norm_types'],
        center_norm_types=architecture['center_decoder']['norm_types'],
        noise_norm_types=architecture['noise_decoder']['norm_types'],
        encoder_dropout_rates=architecture['encoder']['dropout_rates'],
        center_dropout_rates=architecture['center_decoder']['dropout_rates'],
        noise_dropout_rates=architecture['noise_decoder']['dropout_rates'],
        encoder_channel_widths=architecture['encoder']['channel_widths'],
        center_channel_widths=architecture['center_decoder']['channel_widths'],
        noise_channel_widths=architecture['noise_decoder']['channel_widths'],
        center_skip_connections=architecture['center_decoder']['use_skip_connections'],
        noise_skip_connections=architecture['noise_decoder']['use_skip_connections']
    )
    device = next(nas_model.parameters()).device
    fixed_model = fixed_model.to(device)
    
    return fixed_model, architecture


if __name__ == "__main__":
    import argparse
    import os
    import json
    
    parser = argparse.ArgumentParser(description="Extract best architecture from enhanced NAS results")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the saved NAS model (.pt file)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the extracted model and architecture")
    parser.add_argument("--init_features", type=int, default=32,
                        help="Initial number of features")
    
    args = parser.parse_args()
    
    # Load the NAS model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Create a new NAS model instance with the enhanced search space
    from dual_decoder_cae.nas.darts import NASDualDecoderAE
    
    # We don't know the exact latent dimensions used, so use a placeholder
    # The actual latent dimension will be extracted from the model
    nas_model = NASDualDecoderAE(
        in_channels=1,
        init_features=args.init_features,
        output_channels=1,
        latent_dims=[8, 16, 32, 64, 128],
        width_mults=[0.5, 0.75, 1.0, 1.25, 1.5],
        dropout_rates=[0.0, 0.1, 0.2, 0.3, 0.5]
    ).to(device)
    
    # Load the state dict
    nas_model.load_state_dict(checkpoint)
    
    # Extract best architecture and create fixed model
    fixed_model, architecture = create_enhanced_fixed_model_from_nas(
        nas_model,
        in_channels=1,
        init_features=args.init_features,
        output_channels=1
    )
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save the fixed model
    torch.save(fixed_model.state_dict(), os.path.join(args.output_dir, 'fixed_model.pt'))
    
    # Save the architecture configuration as JSON
    with open(os.path.join(args.output_dir, 'architecture.json'), 'w') as f:
        json.dump(architecture, f, indent=4)
    
    print(f"Extracted architecture and fixed model saved to {args.output_dir}")
    print("Architecture configuration:")
    print(json.dumps(architecture, indent=4))