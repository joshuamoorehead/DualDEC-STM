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

# Import modules from your project
from dual_decoder_cae.config.hyperparameters import ExperimentConfig
from dual_decoder_cae.datasets import DualPurposeDataset, load_data_automated
from dual_decoder_cae.nas.combined_loss import NASCombinedLoss
from dual_decoder_cae.nas.darts import NASDualDecoderAE
from dual_decoder_cae.nas.evaluator import DARTSEvaluator
from dual_decoder_cae.utils.visualization import DualVisualizer

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
        noise_weight=hyperparameters['noise_loss_weight']
    )
    
    # Time the forward and backward passes
    model.train()
    
    # Warm-up pass
    for x, target in temp_loader:
        x, target = x.to(device), target.to(device)
        center_output, noise_output, _ = model(x)
        loss, _, _ = temp_criterion(center_output, noise_output, target)
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
        loss, _, _ = temp_criterion(center_output, noise_output, target)
        temp_w_optimizer.zero_grad()
        loss.backward()
        temp_w_optimizer.step()
        
        # Second pass - update architecture params
        center_output, noise_output, _ = model(x)
        loss, _, _ = temp_criterion(center_output, noise_output, target)
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
    run_dir = os.path.join(nas_dir, f'lr_{learning_rate}_bs_{batch_size}_nas_epochs_{nas_epochs}_patches_{patches_per_image}')
    os.makedirs(run_dir, exist_ok=True)
    
    # Initialize tensorboard
    writer = SummaryWriter(run_dir)
    
    # Define transforms - use transforms that match your dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((16, 16)),
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
    
    # Create model
    print("Creating model for NAS...")
    model = NASDualDecoderAE(
        in_channels=1,
        init_features=hyperparameters['init_features'],
        latent_dim=hyperparameters['latent_dim'],
        output_channels=1
    ).to(device)
    
    # Print model summary
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Estimate search time if requested
    if estimate_time:
        print("Estimating search time...")
        time_per_epoch = estimate_search_time(model, train_data, batch_size)
        total_estimated_time = time_per_epoch * nas_epochs
        
        # Convert to hours, minutes, seconds
        hours = int(total_estimated_time // 3600)
        minutes = int((total_estimated_time % 3600) // 60)
        seconds = int(total_estimated_time % 60)
        
        print(f"Estimated time per epoch: {time_per_epoch:.2f} seconds")
        print(f"Estimated total search time for {nas_epochs} epochs: {hours}h {minutes}m {seconds}s")
        
        # Ask for confirmation to proceed
        response = input(f"Proceed with NAS search? This will take approximately {hours}h {minutes}m. (y/n): ")
        if response.lower() != 'y':
            print("NAS search cancelled.")
            return None, None
    
    # Initialize loss function
    criterion = NASCombinedLoss(
        center_weight=hyperparameters['center_loss_weight'],
        noise_weight=hyperparameters['noise_loss_weight']
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
    print(f"Starting Neural Architecture Search for {nas_epochs} epochs...")
    start_time = time.time()
    
    metrics, best_model = evaluator.search(epochs=nas_epochs)
    
    end_time = time.time()
    search_time = end_time - start_time
    hours = int(search_time // 3600)
    minutes = int((search_time % 3600) // 60)
    seconds = int(search_time % 60)
    print(f"NAS search completed in {hours}h {minutes}m {seconds}s")
    
    # Plot training curves
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(metrics['train_loss'], label='Train Loss')
    plt.plot(metrics['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    
    plt.subplot(1, 3, 2)
    plt.plot(metrics['ssim'], label='SSIM')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.title('SSIM Curve')
    
    plt.subplot(1, 3, 3)
    plt.plot(metrics['psnr'], label='PSNR')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (dB)')
    plt.title('PSNR Curve')
    
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'nas_training_curves.png'))
    
    # Save architecture weights
    torch.save({
        'model_state_dict': best_model.state_dict(),
        'metrics': metrics,
        'search_time': search_time
    }, os.path.join(run_dir, 'best_nas_model.pt'))
    
    # Visualize some reconstructions using your DualVisualizer
    print("Generating reconstruction visualizations...")
    model.eval()
    with torch.no_grad():
        val_loader = DataLoader(val_loader.dataset, batch_size=8, shuffle=True)
        orig, _ = next(iter(val_loader))
        orig = orig.to(device)
        center_output, noise_output, latent = model(orig)
        
        # Use your visualizer
        visualizer = DualVisualizer(writer)
        visualizer.visualize_batch(
            orig, noise_output, center_output, 
            nas_epochs, f"NAS_{crystal_type}"
        )
        
        # Also visualize latent space
        visualizer.visualize_latent_space(
            latent, nas_epochs, f"NAS_{crystal_type}"
        )
    
    print(f"NAS completed. Results saved to {run_dir}")
    return best_model, metrics

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Neural Architecture Search for DualDEC-STM")
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