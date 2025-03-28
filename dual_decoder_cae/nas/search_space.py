# dual_decoder_cae/nas/search_space.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MixedLatentDim(nn.Module):
    """
    Mixed operation for DARTS that implements a weighted selection of latent space dimensions.
    This allows the NAS to search for the optimal bottleneck size.
    """
    def __init__(self, in_features, dim_options=None):
        super(MixedLatentDim, self).__init__()
        if dim_options is None:
            # Default options for latent dimensions
            self.dim_options = [8, 16, 32, 64, 128]
        else:
            self.dim_options = dim_options
            
        # Create a linear projection for each dimension option
        self.projections = nn.ModuleList([
            nn.Linear(in_features, dim) for dim in self.dim_options
        ])
        
        # Weights for architectural parameters
        self.weights = nn.Parameter(torch.randn(len(self.dim_options)))
        
    def forward(self, x):
        # Get softmax of weights
        weights = F.softmax(self.weights, dim=0)
        
        # Since we can't directly combine different dimension sizes,
        # we pad all outputs to the largest dimension
        max_dim = max(self.dim_options)
        
        results = []
        for i, proj in enumerate(self.projections):
            # Project to the current dimension
            out = proj(x)
            
            # Pad with zeros if needed (to the maximum dimension)
            if self.dim_options[i] < max_dim:
                padding = torch.zeros(out.size(0), max_dim - self.dim_options[i], 
                                      device=out.device)
                out = torch.cat([out, padding], dim=1)
            
            # Apply weight
            results.append(weights[i] * out)
        
        # Sum the weighted outputs
        return sum(results)
    
    def get_active_dim(self):
        """Return the dimension with the highest weight (for architecture derivation)"""
        weights = F.softmax(self.weights, dim=0)
        return self.dim_options[weights.argmax().item()]

class MixedNormalization(nn.Module):
    """
    Mixed normalization layer for DARTS - allows searching for the optimal normalization type.
    Options include BatchNorm, LayerNorm, InstanceNorm, and GroupNorm
    """
    def __init__(self, num_features, is_2d=True):
        super(MixedNormalization, self).__init__()
        self.ops = nn.ModuleList()
        self.is_2d = is_2d
        
        # For 2D inputs (convolutional features)
        if is_2d:
            # BatchNorm2d
            self.ops.append(nn.BatchNorm2d(num_features))
            
            # LayerNorm (for 2D, need to specify normalized_shape as [C, H, W],
            # but H and W are dynamic, so we use a functional approach in forward)
            self.ln_weight = nn.Parameter(torch.ones(num_features))
            self.ln_bias = nn.Parameter(torch.zeros(num_features))
            
            # InstanceNorm2d
            self.ops.append(nn.InstanceNorm2d(num_features))
            
            # GroupNorm - use num_groups=min(32, num_features)
            num_groups = min(32, num_features)
            self.ops.append(nn.GroupNorm(num_groups, num_features))
        
        # For 1D inputs (fully connected features)
        else:
            # BatchNorm1d
            self.ops.append(nn.BatchNorm1d(num_features))
            
            # LayerNorm
            self.ops.append(nn.LayerNorm(num_features))
            
            # No direct 1D equivalent for InstanceNorm in this context
            # Use a dummy module for simplicity that just returns input
            self.ops.append(nn.Identity())
            
            # GroupNorm can still be used for 1D
            num_groups = min(32, num_features)
            self.ops.append(nn.GroupNorm(num_groups, num_features))
        
        # Weights for architectural parameters
        self.weights = nn.Parameter(torch.randn(len(self.ops) if is_2d else len(self.ops)))
    
    def forward(self, x):
        weights = F.softmax(self.weights, dim=0)
        
        if self.is_2d:
            results = []
            for i, op in enumerate(self.ops):
                # Special handling for LayerNorm in 2D case
                if i == 1:  # LayerNorm
                    # Apply LayerNorm manually
                    # Normalize along last 3 dimensions (C, H, W)
                    mean = x.mean(dim=(1, 2, 3), keepdim=True)
                    var = x.var(dim=(1, 2, 3), unbiased=False, keepdim=True)
                    normalized = (x - mean) / torch.sqrt(var + 1e-5)
                    # Scale and shift
                    shape = (1, x.size(1), 1, 1)  # For broadcasting
                    out = normalized * self.ln_weight.view(*shape) + self.ln_bias.view(*shape)
                    results.append(weights[i] * out)
                else:
                    results.append(weights[i] * op(x))
        else:
            # 1D case - straightforward
            results = [weights[i] * op(x) for i, op in enumerate(self.ops)]
        
        return sum(results)


class MixedChannelWidth(nn.Module):
    """
    Mixed operation to search for optimal channel width/number of filters
    """
    def __init__(self, in_channels, width_options=None):
        super(MixedChannelWidth, self).__init__()
        if width_options is None:
            # Width multiplier options
            self.width_options = [0.5, 0.75, 1.0, 1.25, 1.5]
        else:
            self.width_options = width_options
        
        self.base_channels = in_channels
        self.ops = nn.ModuleDict()  # Use ModuleDict instead for dynamic creation
        
        # Weights for architectural parameters
        self.weights = nn.Parameter(torch.randn(len(self.width_options)))
    
    def forward(self, x):
        # Create convolution layers dynamically based on actual input channels
        in_channels = x.size(1)
        
        # Check if we need to create ops for this input channel size
        if str(in_channels) not in self.ops:
            for width_mult in self.width_options:
                out_channels = int(self.base_channels * width_mult)
                if str(in_channels) not in self.ops:
                    self.ops[str(in_channels)] = nn.ModuleList()
                self.ops[str(in_channels)].append(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1).to(x.device)
                )
        
        weights = F.softmax(self.weights, dim=0)
        
        # Since we have different channel dimensions, we need to pad to the largest
        max_channels = int(self.base_channels * max(self.width_options))
        
        results = []
        for i, op in enumerate(self.ops[str(in_channels)]):
            out = op(x)
            out_channels = out.size(1)
            
            # Pad with zeros if needed
            if out_channels < max_channels:
                padding = torch.zeros(out.size(0), max_channels - out_channels, 
                                     out.size(2), out.size(3), device=out.device)
                out = torch.cat([out, padding], dim=1)
            
            results.append(weights[i] * out)
        
        return sum(results)
    
    def get_active_width(self):
        """Return the width with the highest weight"""
        weights = F.softmax(self.weights, dim=0)
        width_mult = self.width_options[weights.argmax().item()]
        return int(self.base_channels * width_mult)

class MixedSkipConnection(nn.Module):
    """
    Mixed operation to search for optimal skip connection pattern
    """
    def __init__(self, channels):
        super(MixedSkipConnection, self).__init__()
        
        # We'll create these dynamically to handle varying channel counts
        self.base_channels = channels
        self.adjustment = None
        self.gate = None
        
        # 3 options: no skip, regular skip, gated skip
        self.weights = nn.Parameter(torch.randn(3))
    
    def forward(self, x, skip_input):
        # Ensure dimensions match
        if x.shape != skip_input.shape:
            skip_input = F.interpolate(skip_input, size=x.shape[2:])
            
            # If channels don't match, create and use adjustment layer dynamically
            if x.shape[1] != skip_input.shape[1]:
                # Create adjustment layer with correct dimensions
                self.adjustment = nn.Conv2d(
                    skip_input.shape[1], x.shape[1], kernel_size=1
                ).to(x.device)
                skip_input = self.adjustment(skip_input)
        
        # Create gate dynamically if needed
        if self.gate is None or not hasattr(self.gate[0], 'in_channels') or self.gate[0].in_channels != x.shape[1]:
            self.gate = nn.Sequential(
                nn.Conv2d(x.shape[1], x.shape[1], kernel_size=1),
                nn.Sigmoid()
            ).to(x.device)
        
        weights = F.softmax(self.weights, dim=0)
        
        # Option 1: No skip connection
        result = weights[0] * x
        
        # Option 2: Regular skip connection
        result = result + weights[1] * (x + skip_input)
        
        # Option 3: Gated skip connection
        gate_value = self.gate(skip_input)
        result = result + weights[2] * (x + skip_input * gate_value)
        
        return result
    
class MixedLossWeights(nn.Module):
    """
    Searchable weights for combined loss functions.
    This module allows the NAS to find optimal weights between
    different loss components (e.g., center decoder loss vs noise decoder loss).
    """
    def __init__(self, num_losses=2, weight_options=None):
        super(MixedLossWeights, self).__init__()
        
        if weight_options is None:
            # Default weight ratio options between losses
            self.weight_options = [0.2, 0.5, 1.0, 2.0, 5.0]
        else:
            self.weight_options = weight_options
            
        # Number of loss components to balance
        self.num_losses = num_losses
        
        # Create weights for each loss component and each weight option
        self.alpha = nn.ParameterList([
            nn.Parameter(torch.randn(len(self.weight_options)))
            for _ in range(num_losses)
        ])
    
    def forward(self):
        """
        Returns a list of weights for each loss component.
        The weights will sum to num_losses to keep the total loss magnitude consistent.
        """
        # Get softmax weights for each loss component
        weights = []
        for i in range(self.num_losses):
            weight_dist = F.softmax(self.alpha[i], dim=0)
            # For each loss, compute weighted sum of options
            loss_weight = sum(w * opt for w, opt in zip(weight_dist, self.weight_options))
            weights.append(loss_weight)
        
        # Normalize the weights to sum to num_losses
        total = sum(weights)
        normalized_weights = [w * self.num_losses / total for w in weights]
        
        return normalized_weights
    
class MixedDropout(nn.Module):
    """
    Mixed operation to search for optimal dropout rate (regularization strength)
    """
    def __init__(self, dropout_rates=None):
        super(MixedDropout, self).__init__()
        if dropout_rates is None:
            self.dropout_rates = [0.0, 0.1, 0.2, 0.3, 0.5]
        else:
            self.dropout_rates = dropout_rates
        
        # Weights for architectural parameters
        self.weights = nn.Parameter(torch.randn(len(self.dropout_rates)))
    
    def forward(self, x):
        if not self.training:
            return x  # No dropout during evaluation
            
        weights = F.softmax(self.weights, dim=0)
        
        result = 0
        for i, rate in enumerate(self.dropout_rates):
            if rate == 0.0:
                # No dropout
                result = result + weights[i] * x
            else:
                # Apply dropout with the current rate
                mask = torch.bernoulli(torch.full_like(x, 1 - rate)) / (1 - rate)
                result = result + weights[i] * (x * mask)
        
        return result

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