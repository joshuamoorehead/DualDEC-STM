# dual_decoder_cae/nas/extract_architecture.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from dual_decoder_cae.nas.darts import NASDualDecoderAE
from dual_decoder_cae.models.activation_layers import Swish, Mish, GELU

class FixedOperationEncoder(nn.Module):
    """
    Encoder with fixed operations extracted from NAS
    """
    def __init__(self, in_channels=1, init_features=32, latent_dim=64, 
                 activation_type='relu', kernel_sizes=None, dilations=None):
        super(FixedOperationEncoder, self).__init__()
        
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
        
        # Initial convolution
        self.init_conv = nn.Conv2d(in_channels, init_features, kernel_size=3, padding=1)
        
        # Encoder blocks
        self.block1 = nn.Sequential(
            nn.Conv2d(init_features, init_features*2, kernel_size=kernel_sizes[0], 
                     padding=kernel_sizes[0]//2 * dilations[0], dilation=dilations[0], stride=2),
            nn.BatchNorm2d(init_features*2),
            self.activation
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(init_features*2, init_features*4, kernel_size=kernel_sizes[1], 
                     padding=kernel_sizes[1]//2 * dilations[1], dilation=dilations[1], stride=2),
            nn.BatchNorm2d(init_features*4),
            self.activation
        )
        
        self.block3 = nn.Sequential(
            nn.Conv2d(init_features*4, init_features*8, kernel_size=kernel_sizes[2], 
                     padding=kernel_sizes[2]//2 * dilations[2], dilation=dilations[2], stride=2),
            nn.BatchNorm2d(init_features*8),
            self.activation
        )
        
        # Calculate feature size after 3 downsampling operations (divide by 2^3 = 8)
        self.feature_size = lambda input_size: input_size // 8
        
        # To be initialized with input size in forward method
        self.flatten_size = None
        self.fc = None
        self.latent_dim = latent_dim
        
    def initialize_fc(self, x):
        # Get input dimensions and create FC layer at first forward pass
        with torch.no_grad():
            # Pass through convolutional layers
            x = self.init_conv(x)
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            
            # Calculate flattened size
            self.flatten_size = x.size(1) * x.size(2) * x.size(3)
            
            # Initialize fully connected layer
            self.fc = nn.Linear(self.flatten_size, self.latent_dim)
    
    def forward(self, x):
        # Initialize FC layer if not done yet
        if self.fc is None:
            self.initialize_fc(x)
            
        # Forward pass through convolutional layers
        x = self.init_conv(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        # Flatten and project to latent space
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x


class FixedOperationDecoder(nn.Module):
    """
    Decoder with fixed operations extracted from NAS
    """
    def __init__(self, latent_dim=64, init_features=32, output_channels=1,
                activation_type='relu', upsampling_types=None):
        super(FixedOperationDecoder, self).__init__()
        
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
        
        # Default upsampling types if not provided
        if upsampling_types is None:
            # Options: 'transpose', 'bilinear', 'nearest', 'pixelshuffle'
            upsampling_types = ['transpose', 'transpose', 'transpose']
        
        self.init_features = init_features
        self.latent_dim = latent_dim
        
        # To be initialized with input size in forward method
        self.fc = None
        self.reshape_size = None
        
        # Upsampling blocks
        self.block1 = self._create_upsampling_block(
            init_features*8, init_features*4, upsampling_types[0])
        
        self.block2 = self._create_upsampling_block(
            init_features*4, init_features*2, upsampling_types[1])
        
        self.block3 = self._create_upsampling_block(
            init_features*2, init_features, upsampling_types[2])
        
        # Final convolution to get the right number of output channels
        self.final_conv = nn.Conv2d(init_features, output_channels, kernel_size=3, padding=1)
        
    def _create_upsampling_block(self, in_channels, out_channels, upsampling_type):
        """Create an upsampling block based on the specified type"""
        if upsampling_type == 'transpose':
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                self.activation
            )
        elif upsampling_type == 'bilinear':
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                self.activation
            )
        elif upsampling_type == 'nearest':
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                self.activation
            )
        elif upsampling_type == 'pixelshuffle':
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),  # scale_factor=2 -> x4 channels reduction
                nn.BatchNorm2d(out_channels),
                self.activation
            )
        else:
            raise ValueError(f"Unsupported upsampling type: {upsampling_type}")
    
    def initialize_fc(self, x, input_spatial_size):
        # Initial feature map size after three 2x downsampling steps
        feature_map_size = input_spatial_size // 8
        self.reshape_size = (self.init_features*8, feature_map_size, feature_map_size)
        flattened_size = self.init_features*8 * feature_map_size * feature_map_size
        
        # Create FC layer from latent space to flattened feature maps
        self.fc = nn.Linear(self.latent_dim, flattened_size)
    
    def forward(self, x, input_spatial_size):
        # Initialize FC layer if not done yet
        if self.fc is None:
            self.initialize_fc(x, input_spatial_size)
            
        # Project from latent space and reshape to feature maps
        x = self.fc(x)
        batch_size = x.size(0)
        x = x.view(batch_size, *self.reshape_size)
        
        # Upsampling path
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        # Final convolution
        x = self.final_conv(x)
        
        return x


class FixedDualDecoderAE(nn.Module):
    """
    Complete Dual Decoder Autoencoder with fixed operations
    extracted from NAS results
    """
    def __init__(self, in_channels=1, init_features=32, latent_dim=64, output_channels=1,
                encoder_activation='swish', center_decoder_activation='relu', 
                noise_decoder_activation='mish',
                encoder_kernels=None, encoder_dilations=None,
                center_upsampling=None, noise_upsampling=None):
        super(FixedDualDecoderAE, self).__init__()
        
        # Create fixed operations encoder
        self.encoder = FixedOperationEncoder(
            in_channels=in_channels,
            init_features=init_features,
            latent_dim=latent_dim,
            activation_type=encoder_activation,
            kernel_sizes=encoder_kernels,
            dilations=encoder_dilations
        )
        
        # Create fixed operations decoders
        self.center_decoder = FixedOperationDecoder(
            latent_dim=latent_dim,
            init_features=init_features,
            output_channels=output_channels,
            activation_type=center_decoder_activation,
            upsampling_types=center_upsampling
        )
        
        self.noise_decoder = FixedOperationDecoder(
            latent_dim=latent_dim,
            init_features=init_features,
            output_channels=output_channels,
            activation_type=noise_decoder_activation,
            upsampling_types=noise_upsampling
        )
        
    def forward(self, x):
        input_spatial_size = x.size(2)  # Assuming square inputs
        
        # Encode
        z = self.encoder(x)
        
        # Decode with dual decoders
        center_output = self.center_decoder(z, input_spatial_size)
        noise_output = self.noise_decoder(z, input_spatial_size)
        
        return center_output, noise_output, z


def extract_best_architecture(nas_model):
    """
    Extract the best architecture from a trained NAS model by analyzing
    the architecture parameters (weights in MixedOp, etc.)
    
    Args:
        nas_model: A trained NASDualDecoderAE model
        
    Returns:
        dict: Configuration for best architecture
    """
    architecture = {
        'encoder': {
            'activation': None,
            'kernel_sizes': [],
            'dilations': []
        },
        'center_decoder': {
            'activation': None,
            'upsampling_types': []
        },
        'noise_decoder': {
            'activation': None,
            'upsampling_types': []
        }
    }
    
    # Activation function options
    activation_functions = ['relu', 'leaky_relu', 'swish', 'mish', 'gelu']
    
    # Upsampling options
    upsampling_types = ['transpose', 'bilinear', 'nearest', 'pixelshuffle']
    
    # Process encoder
    # Collect encoder activation choices
    encoder_activations = []
    for name, module in nas_model.named_modules():
        if isinstance(module, nn.Module) and name.startswith('encoder.mixed_act'):
            weights = F.softmax(module.weights, dim=0).detach().cpu().numpy()
            best_idx = np.argmax(weights)
            encoder_activations.append(activation_functions[best_idx])
    
    # Use most common activation for encoder
    if encoder_activations:
        from collections import Counter
        architecture['encoder']['activation'] = Counter(encoder_activations).most_common(1)[0][0]
    
    # Process encoder kernels and dilations
    for name, module in nas_model.named_modules():
        if isinstance(module, nn.Module) and name.startswith('encoder.mixed_op'):
            weights = F.softmax(module.weights, dim=0).detach().cpu().numpy()
            best_idx = np.argmax(weights)
            
            # Determine kernel size and dilation based on best operation
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
    
    # Process center decoder
    # Collect center decoder activation choices
    center_activations = []
    for name, module in nas_model.named_modules():
        if isinstance(module, nn.Module) and name.startswith('center_decoder.mixed_act'):
            weights = F.softmax(module.weights, dim=0).detach().cpu().numpy()
            best_idx = np.argmax(weights)
            center_activations.append(activation_functions[best_idx])
    
    # Use most common activation for center decoder
    if center_activations:
        from collections import Counter
        architecture['center_decoder']['activation'] = Counter(center_activations).most_common(1)[0][0]
    
    # Process center decoder upsampling types
    for name, module in nas_model.named_modules():
        if isinstance(module, nn.Module) and name.startswith('center_decoder.mixed_up'):
            weights = F.softmax(module.weights, dim=0).detach().cpu().numpy()
            best_idx = np.argmax(weights)
            architecture['center_decoder']['upsampling_types'].append(upsampling_types[best_idx])
    
    # Process noise decoder
    # Collect noise decoder activation choices
    noise_activations = []
    for name, module in nas_model.named_modules():
        if isinstance(module, nn.Module) and name.startswith('noise_decoder.mixed_act'):
            weights = F.softmax(module.weights, dim=0).detach().cpu().numpy()
            best_idx = np.argmax(weights)
            noise_activations.append(activation_functions[best_idx])
    
    # Use most common activation for noise decoder
    if noise_activations:
        from collections import Counter
        architecture['noise_decoder']['activation'] = Counter(noise_activations).most_common(1)[0][0]
    
    # Process noise decoder upsampling types
    for name, module in nas_model.named_modules():
        if isinstance(module, nn.Module) and name.startswith('noise_decoder.mixed_up'):
            weights = F.softmax(module.weights, dim=0).detach().cpu().numpy()
            best_idx = np.argmax(weights)
            architecture['noise_decoder']['upsampling_types'].append(upsampling_types[best_idx])
    
    return architecture


def create_fixed_model_from_nas(nas_model, in_channels=1, init_features=32, latent_dim=64, output_channels=1):
    """
    Create a fixed model based on the best architecture found from NAS
    
    Args:
        nas_model: A trained NASDualDecoderAE model
        
    Returns:
        FixedDualDecoderAE: A model with fixed operations based on NAS results
    """
    # Extract architecture configuration
    architecture = extract_best_architecture(nas_model)
    
    # Create fixed model
    fixed_model = FixedDualDecoderAE(
        in_channels=in_channels,
        init_features=init_features,
        latent_dim=latent_dim,
        output_channels=output_channels,
        encoder_activation=architecture['encoder']['activation'],
        center_decoder_activation=architecture['center_decoder']['activation'],
        noise_decoder_activation=architecture['noise_decoder']['activation'],
        encoder_kernels=architecture['encoder']['kernel_sizes'],
        encoder_dilations=architecture['encoder']['dilations'],
        center_upsampling=architecture['center_decoder']['upsampling_types'],
        noise_upsampling=architecture['noise_decoder']['upsampling_types']
    )
    
    return fixed_model, architecture


if __name__ == "__main__":
    import argparse
    import os
    import json
    
    parser = argparse.ArgumentParser(description="Extract best architecture from NAS results")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the saved NAS model (.pt file)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the extracted model and architecture")
    parser.add_argument("--latent_dim", type=int, default=64,
                        help="Latent dimension size")
    parser.add_argument("--init_features", type=int, default=32,
                        help="Initial number of features")
    
    args = parser.parse_args()
    
    # Load the NAS model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Create a new NAS model instance
    nas_model = NASDualDecoderAE(
        in_channels=1,
        init_features=args.init_features,
        latent_dim=args.latent_dim,
        output_channels=1
    ).to(device)
    
    # Load the state dict
    nas_model.load_state_dict(checkpoint['model_state_dict'])
    
    # Extract best architecture and create fixed model
    fixed_model, architecture = create_fixed_model_from_nas(
        nas_model,
        in_channels=1,
        init_features=args.init_features,
        latent_dim=args.latent_dim,
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