# dual_decoder_cae/run_nas.py
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
import json

# Import modules from your project
from dual_decoder_cae.config.hyperparameters import ExperimentConfig
from dual_decoder_cae.datasets import DualPurposeDataset, load_data_automated
from dual_decoder_cae.nas.combined_loss import NASCombinedLoss
from dual_decoder_cae.nas.darts import NASDualDecoderAE
from dual_decoder_cae.nas.evaluator import DARTSEvaluator
from dual_decoder_cae.utils.visualization import DualVisualizer
from dual_decoder_cae.nas.extract_architecture import extract_enhanced_architecture, create_enhanced_fixed_model_from_nas

# Create a default configuration
config = ExperimentConfig()

# Create a hyperparameters dictionary based on your config structure
hyperparameters = {
    'init_features': config.model.initial_filters if hasattr(config.model, 'initial_filters') else config.model.encoder_channels[1],
    'latent_dim': config.training.latent_dim,
    'center_loss_weight': config.training.center_weight,
    'noise_loss_weight': config.training.noise_weight
}

def estimate_search_time(model, dataset, batch_size=64, num_iterations=5):
    """
    Estimate the time required for the NAS search by timing a few iterations
    
    Args:
        model: The NAS model
        dataset: Training dataset
        batch_size: Batch size for estimation
        num_iterations: Number of iterations to use for estimation
        
    Returns:
        estimated_time_per_epoch: Estimated time per epoch in seconds
    """
    device = next(model.parameters()).device
    
    # Create a small dataloader
    temp_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                            num_workers=2, pin_memory=True)
    
    # Create temporary optimizer and loss function
    temp_w_optimizer = optim.Adam(model.parameters(), lr=0.001)
    temp_alpha_optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.0001)
    temp_criterion = NASCombinedLoss(
        center_weight=hyperparameters['center_loss_weight'],
        noise_weight=hyperparameters['noise_loss_weight'],
        use_searchable_weights=True
    )
    
    # Time the forward and backward passes
    model.train()
    
    # Warm-up pass
    for x, target in temp_loader:
        x, target = x.to(device), target.to(device)
        center_output, noise_output, _ = model(x)
        loss, _, _ = temp_criterion(center_output, noise_output, target, model=model)
        temp_w_optimizer.zero_grad()
        loss.backward()
        temp_w_optimizer.step()
        break
    
    # Timed passes
    start_time = time.time()
    for i, (x, target) in enumerate(temp_loader):
        if i >= num_iterations:
            break
            
        x, target = x.to(device), target.to(device)
        
        # First pass - update weights
        center_output, noise_output, _ = model(x)
        loss, _, _ = temp_criterion(center_output, noise_output, target, model=model)
        temp_w_optimizer.zero_grad()
        loss.backward()
        temp_w_optimizer.step()
        
        # Second pass - update architecture params
        center_output, noise_output, _ = model(x)
        loss, _, _ = temp_criterion(center_output, noise_output, target, model=model)
        temp_alpha_optimizer.zero_grad()
        loss.backward()
        temp_alpha_optimizer.step()
    
    end_time = time.time()
    
    # Calculate average time per iteration
    avg_time_per_iteration = (end_time - start_time) / num_iterations
    
    # Estimate time per epoch
    num_batches = len(dataset) // batch_size
    estimated_time_per_epoch = avg_time_per_iteration * num_batches
    
    return estimated_time_per_epoch


def run_nas(crystal_type='Cubic', learning_rate=0.001, nas_epochs=20, 
            batch_size=64, patches_per_image=3000, data_dir='../Data', 
            estimate_time=True, patch_size=17):
    """
    Run Neural Architecture Search for the Dual Decoder Autoencoder
    with enhanced search space including latent dimensionality, model depth,
    channel widths, normalization layers, skip connections, dropout rates,
    and loss function weights.
    
    Args:
        crystal_type (str): Type of crystal structure ('Cubic', 'BCC', 'FCC')
        learning_rate (float): Learning rate for weight optimizer
        nas_epochs (int): Number of NAS search epochs
        batch_size (int): Batch size for training
        patches_per_image (int): Number of patches to extract per image
        data_dir (str): Directory containing the data
        estimate_time (bool): Whether to estimate search time before starting
        patch_size (int): Size of image patches to extract
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directory for NAS results
    nas_dir = os.path.join('runs/nas_experiment', crystal_type)
    os.makedirs(nas_dir, exist_ok=True)
    
    # Create run directory with hyperparameters
    run_dir = os.path.join(nas_dir, f'enhanced_nas_lr_{learning_rate}_bs_{batch_size}_epochs_{nas_epochs}_patches_{patches_per_image}')
    os.makedirs(run_dir, exist_ok=True)
    
    # Initialize tensorboard
    writer = SummaryWriter(run_dir)
    
    # Define transforms - use transforms that match your dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Load dataset using your data loading function
    print("Loading datasets...")
    try:
        train_loader, val_loader, test_loader, _ = load_data_automated(
            data_root=data_dir,
            batch_size=batch_size,
            transform=transform,
            patches_per_image=patches_per_image,
            lattice_type=crystal_type
        )
        
        # Get training dataset for time estimation
        train_data = train_loader.dataset
        
        print(f"Loaded {len(train_data)} training samples")
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise
    
    # Define search spaces for architecture parameters
    latent_dims = [8, 16, 32, 64, 128]  # Options for latent space dimensionality
    width_mults = [0.5, 0.75, 1.0, 1.25, 1.5]  # Width multiplier options
    dropout_rates = [0.0, 0.1, 0.2, 0.3, 0.5]  # Dropout rate options
    
    # Create model with enhanced search space
    print("Creating model with enhanced search space for NAS...")
    model = NASDualDecoderAE(
        in_channels=1,
        init_features=hyperparameters['init_features'],
        output_channels=1,
        latent_dims=latent_dims,
        width_mults=width_mults,
        dropout_rates=dropout_rates
    ).to(device)
    
    # Print model summary
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Print search space
    print("\nEnhanced Neural Architecture Search Space:")
    print(f"Latent Dimensions: {latent_dims}")
    print(f"Channel Width Multipliers: {width_mults}")
    print(f"Dropout Rates: {dropout_rates}")
    print("Normalization Types: BatchNorm, LayerNorm, InstanceNorm, GroupNorm")
    print("Skip Connection Types: None, Regular, Gated")
    print("Loss Function Weights: Searchable balance between center and noise decoders")
    
    # Estimate search time if requested
    if estimate_time:
        print("\nEstimating search time for enhanced NAS...")
        time_per_epoch = estimate_search_time(model, train_data, batch_size)
        total_estimated_time = time_per_epoch * nas_epochs
        
        # Convert to hours, minutes, seconds
        hours = int(total_estimated_time // 3600)
        minutes = int((total_estimated_time % 3600) // 60)
        seconds = int(total_estimated_time % 60)
        
        print(f"Estimated time per epoch: {time_per_epoch:.2f} seconds")
        print(f"Estimated total search time for {nas_epochs} epochs: {hours}h {minutes}m {seconds}s")
        
        # Ask for confirmation to proceed
        response = input(f"Proceed with enhanced NAS search? This will take approximately {hours}h {minutes}m. (y/n): ")
        if response.lower() != 'y':
            print("NAS search cancelled.")
            return None, None
    
    # Initialize loss function with searchable weights
    criterion = NASCombinedLoss(
        center_weight=hyperparameters['center_loss_weight'],
        noise_weight=hyperparameters['noise_loss_weight'],
        use_searchable_weights=True  # Enable searchable weights
    )
    
    # Create DARTS evaluator
    evaluator = DARTSEvaluator(
        model=model,
        train_dataset=train_data,
        val_dataset=val_loader.dataset,
        combined_loss_fn=criterion,
        batch_size=batch_size,
        w_lr=learning_rate,
        alpha_lr=learning_rate / 10  # Architecture params typically use lower LR
    )
    
    # Run NAS search
    print(f"\nStarting Enhanced Neural Architecture Search for {nas_epochs} epochs...")
    start_time = time.time()
    
    metrics, best_model = evaluator.search(epochs=nas_epochs)
    
    end_time = time.time()
    search_time = end_time - start_time
    hours = int(search_time // 3600)
    minutes = int((search_time % 3600) // 60)
    seconds = int(search_time % 60)
    print(f"Enhanced NAS search completed in {hours}h {minutes}m {seconds}s")
    
    # Extract the architecture parameters
    print("\nExtracting architecture parameters from best model...")
    arch_info = extract_enhanced_architecture(best_model)
    
    # Save architecture info as JSON
    with open(os.path.join(run_dir, 'architecture_info.json'), 'w') as f:
        json.dump(arch_info, f, indent=4)
    
    """ # Create a fixed model based on the best architecture
    fixed_model, _ = create_enhanced_fixed_model_from_nas(
        best_model,
        in_channels=1,
        init_features=hyperparameters['init_features'],
        output_channels=1
    ) """


        # Save the NAS model (skip fixed model creation)
    torch.save({
        'model_state_dict': best_model.state_dict(),
        'architecture': arch_info,
        'metrics': metrics,
        'search_time': search_time
    }, os.path.join(run_dir, 'best_nas_model.pt'))

    # Plot training curves
    plt.figure(figsize=(16, 12))

    # Loss curves
    plt.subplot(2, 2, 1)
    plt.plot(metrics['train_loss'], label='Train Loss')
    plt.plot(metrics['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')

    # SSIM curve
    plt.subplot(2, 2, 2)
    plt.plot(metrics['ssim'], label='SSIM')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.title('SSIM Curve')

    # PSNR curve
    plt.subplot(2, 2, 3)
    plt.plot(metrics['psnr'], label='PSNR')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (dB)')
    plt.title('PSNR Curve')

    # Latent dimension weights
    if 'latent_dim' in arch_info and arch_info['latent_dim'] is not None:
        plt.subplot(2, 2, 4)
        plt.bar(range(len(latent_dims)),
                [0.2] * len(latent_dims),  # Placeholder heights - replace with actual weights if available
                tick_label=latent_dims)
        plt.axvline(x=latent_dims.index(arch_info['latent_dim']), color='r', linestyle='--')
        plt.xlabel('Latent Dimension')
        plt.ylabel('Normalized Weight')
        plt.title(f'Selected Latent Dimension: {arch_info["latent_dim"]}')

    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'nas_training_curves.png'))

    # Print architecture summary
    print("\nBest Architecture Summary:")
    print(f"- Latent Dimension: {arch_info['latent_dim']}")

    if 'encoder' in arch_info:
        print("Encoder:")
        print(f"- Activation: {arch_info['encoder']['activation']}")
        print(f"- Channel Widths: {arch_info['encoder']['channel_widths']}")
        
    if 'center_decoder' in arch_info:
        print("Center Decoder:")
        print(f"- Activation: {arch_info['center_decoder']['activation']}")
        print(f"- Upsampling: {arch_info['center_decoder']['upsampling_types']}")
        print(f"- Skip Connections: {arch_info['center_decoder']['use_skip_connections']}")
        
    if 'noise_decoder' in arch_info:
        print("Noise Decoder:")
        print(f"- Activation: {arch_info['noise_decoder']['activation']}")
        print(f"- Upsampling: {arch_info['noise_decoder']['upsampling_types']}")
        print(f"- Skip Connections: {arch_info['noise_decoder']['use_skip_connections']}")
        
    if 'loss_weights' in arch_info:
        print("Loss Weights:")
        print(f"- Center Weight: {arch_info['loss_weights']['center_weight']:.2f}")
        print(f"- Noise Weight: {arch_info['loss_weights']['noise_weight']:.2f}")

    # Visualize some reconstructions using your DualVisualizer
    print("\nGenerating reconstruction visualizations...")
    best_model.eval()  # Use the best_model directly instead of fixed_model
    with torch.no_grad():
        val_loader = DataLoader(val_loader.dataset, batch_size=8, shuffle=True)
        orig, _ = next(iter(val_loader))
        orig = orig.to(device)
        center_output, noise_output, latent = best_model(orig)  # Use best_model
        
        # Use your visualizer
        visualizer = DualVisualizer(writer)
        visualizer.visualize_batch(
            orig, noise_output, center_output,
            nas_epochs, f"Enhanced_NAS_{crystal_type}"
        )
        
        # Also visualize latent space
        visualizer.visualize_latent_space(
            latent, nas_epochs, f"Enhanced_NAS_{crystal_type}"
        )

    print(f"\nEnhanced NAS completed. Results saved to {run_dir}")
    return best_model, arch_info  # Return just the best_model and architecture info


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Enhanced Neural Architecture Search for DualDEC-STM")
    parser.add_argument("--crystal", type=str, default="Cubic", choices=["Cubic", "BCC", "FCC"],
                        help="Crystal type (Cubic, BCC, or FCC)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=20, help="Number of NAS search epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--patches", type=int, default=3000, help="Patches per image")
    parser.add_argument("--data-dir", type=str, default="../Data", help="Data directory")
    parser.add_argument("--patch-size", type=int, default=17, help="Patch size")
    parser.add_argument("--no-time-estimate", action="store_true", help="Skip time estimation")
    
    args = parser.parse_args()
    
    run_nas(
        crystal_type=args.crystal,
        learning_rate=args.lr,
        nas_epochs=args.epochs,
        batch_size=args.batch_size,
        patches_per_image=args.patches,
        data_dir=args.data_dir,
        estimate_time=not args.no_time_estimate,
        patch_size=args.patch_size
    )