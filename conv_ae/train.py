import torch
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.nn as nn
import time
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from models import Autoencoder
from datasets import load_data_automated 
from utils import get_available_gpus
import os
import torchvision
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import ScalarFormatter
from tensorboard.backend.event_processing import event_accumulator
import json
from collections import defaultdict

# Configuration dictionary
configs = [
    {
        "name": "baseline",
        "learning_rate": 0.001,
        "batch_size": 1024,
        "epochs": 100,
        "patches_per_image": 3000
    },
    {
        "name": "lower_lr",
        "learning_rate": 0.0001,
        "batch_size": 1024,
        "epochs": 100,
        "patches_per_image": 3000
    },
    {
        "name": "small_batch",
        "learning_rate": 0.001,
        "batch_size": 256,
        "epochs": 100,
        "patches_per_image": 3000
    },
    {
        "name": "large_batch",
        "learning_rate": 0.002,
        "batch_size": 2048,
        "epochs": 100,
        "patches_per_image": 3000
    },
    {
        "name": "more_patches",
        "learning_rate": 0.001,
        "batch_size": 1024,
        "epochs": 100,
        "patches_per_image": 4900
    },
    {
        "name": "extended_training",
        "learning_rate": 0.001,
        "batch_size": 1024,
        "epochs": 200,
        "patches_per_image": 3000
    },
    {
        "name": "lr_decay",
        "learning_rate": 0.001,
        "batch_size": 1024,
        "epochs": 100,
        "patches_per_image": 3000
    }
]

# Device selection
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Training on: {device}')

# Define the transform
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def print_gpu_utilization():
    device = torch.cuda.current_device()
    print(f"GPU Utilization:")
    print(f"GPU {device}: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB / {torch.cuda.get_device_properties(device).total_memory / 1024**3:.2f} GB")


def calculate_similarity(original_patches, reconstructed_patches):
    original_np = original_patches.cpu().numpy().squeeze()
    reconstructed_np = reconstructed_patches.cpu().numpy().squeeze()
    
    print(f"Original patch shape: {original_np.shape}")
    print(f"Reconstructed patch shape: {reconstructed_np.shape}")
    
    # Ensure the patches are 17x17
    assert original_np.shape[-2:] == (17, 17), f"Expected 17x17 patches, got {original_np.shape[-2:]}"
    
    # Calculate SSIM
    try:
        ssim_values = []
        for orig, recon in zip(original_np, reconstructed_np):
            ssim_value = ssim(orig, recon, data_range=1, win_size=11)  # Using win_size=11 for 17x17 patches
            ssim_values.append(ssim_value)
        
        avg_ssim = np.mean(ssim_values)
        print(f"Average SSIM: {avg_ssim}")
        return avg_ssim
    except Exception as e:
        print(f"SSIM calculation failed: {str(e)}")
        print("Falling back to pixel-wise comparison")
        similarity = 1 - np.mean(np.abs(original_np - reconstructed_np))
        return similarity
    
def log_reconstructed_patches(model, data_loader, writer, epoch, lattice_type, device, num_patches=3):
    model.eval()
    
    try:
        # Update file paths to use the new directory structure
        tracking_dir = os.path.join('tracking', lattice_type)
        tracked_patches_file = os.path.join(tracking_dir, f'tracked_patches.pt')
        tracked_indices_file = os.path.join(tracking_dir, f'tracked_indices.pt')
        
        # If it's the first epoch or the file doesn't exist, select new patches to track
        if epoch == 0 or not os.path.exists(tracked_patches_file):
            all_patches = next(iter(data_loader))
            total_patches = all_patches.size(0)
            tracked_indices = torch.randperm(total_patches)[:num_patches]
            tracked_patches = all_patches[tracked_indices]
            # Save both patches and indices
            torch.save(tracked_patches, tracked_patches_file)
            torch.save(tracked_indices, tracked_indices_file)
        else:
            # Load the same tracked patches
            tracked_patches = torch.load(tracked_patches_file)
        
        original_patches = tracked_patches.to(device)
        
        with torch.no_grad():
            reconstructed_patches = model(original_patches)
        
        # Calculate MSE and Similarity
        mse = F.mse_loss(original_patches, reconstructed_patches).item()
        similarity_value = calculate_similarity(original_patches, reconstructed_patches)

        writer.add_scalar(f'{lattice_type}/Reconstruction_MSE', mse, epoch)
        writer.add_scalar(f'{lattice_type}/Reconstruction_Similarity', similarity_value, epoch)

        print(f"Epoch {epoch}: Reconstruction MSE = {mse:.4f}, Similarity = {similarity_value:.4f}")

        # Log individual comparisons
        for i in range(num_patches):
            # Create a side-by-side comparison
            comparison = torch.cat([original_patches[i], reconstructed_patches[i]], dim=2)
            # Resize for better visibility
            comparison_resized = F.interpolate(comparison.unsqueeze(0), scale_factor=8, mode='nearest').squeeze(0)
            # Normalize for visualization
            comparison_normalized = (comparison_resized - comparison_resized.min()) / (comparison_resized.max() - comparison_resized.min())
            writer.add_image(f'{lattice_type}/Patch_Comparison_{i}', comparison_normalized, epoch, dataformats='CHW')

        # Optional: You can still keep the grid view if desired
        all_patches = torch.cat([original_patches, reconstructed_patches], dim=0)
        patch_grid = torchvision.utils.make_grid(all_patches, nrow=num_patches, padding=2, normalize=True)
        patch_grid_resized = F.interpolate(patch_grid.unsqueeze(0), scale_factor=4, mode='nearest').squeeze(0)
        writer.add_image(f'{lattice_type}/All_Patches_Grid', patch_grid_resized, epoch, dataformats='CHW')
    
    except Exception as e:
        print(f"Error in log_reconstructed_patches: {str(e)}")
        print(f"Patch shape: {original_patches.shape if 'original_patches' in locals() else 'Unknown'}")
    
    return {'mse': mse, 'ssim': similarity_value}



def visualize_latent_space(model, data_loader, writer, epoch, lattice_type, n_samples=1000):
    model.eval()
    latent_vectors = []
    
    with torch.no_grad():
        for batch in data_loader:
            if len(latent_vectors) >= n_samples:
                break
            batch = batch.to(next(model.parameters()).device)
            _, latent = model(batch, return_latent=True)
            latent_vectors.append(latent.cpu())

    latent_vectors = torch.cat(latent_vectors, dim=0)[:n_samples]

    pca = PCA(n_components=3)
    latent_pca = pca.fit_transform(latent_vectors)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(latent_pca[:, 0], latent_pca[:, 1], latent_pca[:, 2], c=range(len(latent_pca)), cmap='viridis')
    ax.set_title(f'PCA of Latent Space (Epoch {epoch})')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    plt.colorbar(scatter)
    writer.add_figure(f'{lattice_type}/Latent_Space_PCA', fig, epoch)
    plt.close()

def train_autoencoder(learning_rate, batch_size, num_epochs, patches_per_image, lattice_type, resume_training=False):
    start_time = time.time()
    train_loader, val_loader, test_loader, _ = load_data_automated(
        data_root="../Data", 
        batch_size=batch_size, 
        transform=transform, 
        patches_per_image=patches_per_image,
        lattice_type=lattice_type
    )

    num_images = len(train_loader.dataset.image_paths) if hasattr(train_loader.dataset, 'image_paths') else 1

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Autoencoder().to(device)
    
    weights_dir = os.path.join('models', 'weights', lattice_type)
    run_name = f'lr_{learning_rate}_bs_{batch_size}_epochs_{num_epochs}_patches_{patches_per_image}_images_{num_images}'
    log_dir = os.path.join('runs', 'autoencoder_experiment', lattice_type, run_name)
    
    start_epoch = 0
    best_loss = float('inf')
    final_mse = None
    final_ssim = None

    # Load checkpoint if resuming
    checkpoint_path = os.path.join(weights_dir, 'checkpoint.pt')
    if resume_training and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['loss']
        print(f"Resuming training from epoch {start_epoch}")
    
    writer = SummaryWriter(log_dir=log_dir)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if resume_training and os.path.exists(checkpoint_path):
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Add scheduler for lr_decay configuration
    if run_name.startswith('lr_decay'):
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        if resume_training and checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print("Using learning rate decay scheduler: StepLR with step_size=30, gamma=0.1")

    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()
        model.train()
        train_loss = 0.0

        for batch_idx, patches in enumerate(train_loader):
            patches = patches.to(device)
            outputs, _ = model(patches, return_latent=True)
            loss = criterion(outputs, patches)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * patches.size(0)

        train_loss = train_loss / len(train_loader.dataset)
        writer.add_scalar(f'Loss/train', train_loss, epoch)
        
        if run_name.startswith('lr_decay'):
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            writer.add_scalar('Learning_Rate', current_lr, epoch)

        if epoch % 10 == 0:
            patch_metrics = log_reconstructed_patches(model, train_loader, writer, epoch, lattice_type, device)
            final_mse = patch_metrics['mse']
            final_ssim = patch_metrics['ssim']
            visualize_latent_space(model, train_loader, writer, epoch, lattice_type)

        if val_loader is not None:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for data in val_loader:
                    images = data.to(device)
                    outputs, _ = model(images, return_latent=True)
                    loss = criterion(outputs, images)
                    val_loss += loss.item() * images.size(0)
            val_loss = val_loss / len(val_loader.dataset)

            writer.add_scalar(f'Loss/val', val_loss, epoch)
            print(f'Epoch: {epoch+1} \tTraining Loss: {train_loss:.6f} \tValidation Loss: {val_loss:.6f} \tTime taken: {time.time() - epoch_start_time:.2f} sec')
            
            if run_name.startswith('lr_decay'):
                print(f'Current Learning Rate: {current_lr}')

            if val_loss < best_loss:
                best_loss = val_loss
                weights_path = os.path.join(weights_dir, f'best_model_weights.pth')
                checkpoint_path = os.path.join(weights_dir, f'checkpoint.pt')
                
                torch.save(model.state_dict(), weights_path)
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if run_name.startswith('lr_decay') else None,
                    'loss': best_loss,
                    'hyperparameters': {
                        'learning_rate': learning_rate,
                        'batch_size': batch_size,
                        'patches_per_image': patches_per_image
                    }
                }
                torch.save(checkpoint, checkpoint_path)
        else:
            print(f'Epoch: {epoch+1} \tTraining Loss: {train_loss:.6f} \tTime taken: {time.time() - epoch_start_time:.2f} sec')
    
    total_time = time.time() - start_time
    final_metrics = {
        'final_train_loss': train_loss,
        'final_val_loss': val_loss if val_loader else None,
        'best_val_loss': best_loss if val_loader else None,
        'final_mse': final_mse,
        'final_ssim': final_ssim,
        'training_time': total_time,
        'hyperparameters': {
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'epochs': num_epochs,
            'patches_per_image': patches_per_image
        }
    }
    
    config_name = next(cfg['name'] for cfg in configs 
                      if cfg['learning_rate'] == learning_rate 
                      and cfg['batch_size'] == batch_size 
                      and cfg['epochs'] == num_epochs 
                      and cfg['patches_per_image'] == patches_per_image)
    
    save_metrics(lattice_type, config_name, final_metrics)
    
    print(f'Training complete! Total time: {total_time/3600:.2f} hours')
    print(f'Final MSE: {final_mse:.6f}, Final SSIM: {final_ssim:.6f}')
    writer.close()
    

def extract_tensorboard_data(log_dir):
    """Extract data from TensorBoard logs into a structured format."""
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()
    
    # Get lattice type from directory path
    lattice_type = None
    for lt in ['Cubic', 'BCC', 'FCC']:
        if lt in log_dir:
            lattice_type = lt
            break
    
    data = {
        'Loss/train': [],
        'Loss/val': [],
        'Reconstruction_MSE': [],
        'Reconstruction_Similarity': []
    }
    
    # Map the metric names to their logged versions
    metric_map = {
        'Loss/train': 'Loss/train',
        'Loss/val': 'Loss/val',
        'Reconstruction_MSE': f'{lattice_type}/Reconstruction_MSE',
        'Reconstruction_Similarity': f'{lattice_type}/Reconstruction_Similarity'
    }
    
    # Get all scalar events
    for metric, logged_name in metric_map.items():
        if logged_name in ea.Tags()['scalars']:
            events = ea.Scalars(logged_name)
            data[metric] = [(event.step, event.value) for event in events]
            print(f"Found {len(data[metric])} events for {metric}")
    
    return data

def create_training_plots(all_results, save_path='analysis/figures/'):
    """Create enhanced training and validation plots for all lattice types."""
    # Make sure the save directory exists
    os.makedirs(save_path, exist_ok=True)

    lattice_types = ['Cubic', 'BCC', 'FCC']
    metrics = ['Loss/train', 'Loss/val']
    
    # Enhanced style configuration
    styles = {
        'baseline': {'color': '#1f77b4', 'linestyle': '-', 'marker': 'o', 'label': 'Baseline'},
        'lower_lr': {'color': '#ff7f0e', 'linestyle': '--', 'marker': 's', 'label': 'Lower LR'},
        'small_batch': {'color': '#2ca02c', 'linestyle': ':', 'marker': '^', 'label': 'Small Batch'},
        'large_batch': {'color': '#d62728', 'linestyle': '-.', 'marker': 'v', 'label': 'Large Batch'},
        'more_patches': {'color': '#9467bd', 'linestyle': '-', 'marker': 'D', 'label': 'More Patches'},
        'extended_training': {'color': '#8c564b', 'linestyle': '--', 'marker': 'p', 'label': 'Extended'},
        'lr_decay': {'color': '#e377c2', 'linestyle': ':', 'marker': '*', 'label': 'LR Decay'}
    }
    
    # Apply style defaults to all configurations
    for style in styles.values():
        style.update({
            'markevery': 20,
            'linewidth': 2.5,
            'markersize': 8
        })

    # Create a single figure with 2x3 subplot layout
    fig = plt.figure(figsize=(20, 12), dpi=300)
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Set modern style
    plt.style.use('default')
    # Set custom style parameters
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'axes.axisbelow': True,
    })
    
    for i, metric in enumerate(['Loss/train', 'Loss/val']):
        for j, lattice in enumerate(lattice_types):
            ax = fig.add_subplot(gs[i, j])
            
            max_x = 0
            for config in all_results[lattice]:
                data = all_results[lattice][config].get(metric, [])
                
                if data:
                    x = [point[0] for point in data]
                    y = [point[1] for point in data]
                    
                    if len(x) > 0:
                        ax.plot(x, y, **styles[config])
                        max_x = max(max_x, max(x))
            
            # Customize each subplot
            ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
            ax.set_ylabel('Loss' if j == 0 else '', fontsize=12, fontweight='bold')
            ax.set_title(f'{"Training" if i == 0 else "Validation"} Loss - {lattice}',
                        fontsize=14, fontweight='bold', pad=10)
            
            # Scientific notation for y-axis with fixed precision
            ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            
            # Enhance grid and ticks
            ax.tick_params(axis='both', which='major', labelsize=10)
            
            # Set consistent y-axis limits across rows
            if j > 0:
                ax.set_ylim(ax.get_ylim())

    # Add single legend for the entire figure
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels,
              loc='center right',
              bbox_to_anchor=(0.98, 0.5),
              fontsize=12,
              frameon=True,
              fancybox=True,
              shadow=True)

    # Save the combined plot
    filename = os.path.join(save_path, f'combined_loss_curves_{metric.replace("/", "_")}.png')
    plt.savefig(filename,
                bbox_inches='tight',
                dpi=300,
                facecolor='white',
                edgecolor='none')
    plt.close()

def create_performance_heatmap(all_results):
    """Generate heatmap of final performance metrics."""
    metrics = ['Loss/train', 'Loss/val', 'Reconstruction_MSE', 'Reconstruction_Similarity']
    lattice_types = ['Cubic', 'BCC', 'FCC']
    
    # Map config names to actual directory patterns
    config_dir_map = {
        'baseline': 'lr_0.001_bs_1024_epochs_100_patches_3000',
        'lower_lr': 'lr_0.0001_bs_1024_epochs_100_patches_3000',
        'small_batch': 'lr_0.001_bs_256_epochs_100_patches_3000',
        'large_batch': 'lr_0.002_bs_2048_epochs_100_patches_3000',
        'more_patches': 'lr_0.001_bs_1024_epochs_100_patches_4900',
        'extended_training': 'lr_0.001_bs_1024_epochs_200_patches_3000',
        'lr_decay': 'lr_0.001_bs_1024_epochs_100_patches_3000'  # This might need adjustment
    }
    
    for metric in metrics:
        final_values = []
        for lattice in lattice_types:
            row = []
            for config_name, dir_pattern in config_dir_map.items():
                # Find the matching directory
                matching_dirs = [d for d in os.listdir(f'runs/autoencoder_experiment/{lattice}') 
                               if d.startswith(dir_pattern)]
                if matching_dirs:
                    # Use the first matching directory
                    dir_path = os.path.join('runs/autoencoder_experiment', lattice, matching_dirs[0])
                    try:
                        ea = event_accumulator.EventAccumulator(dir_path)
                        ea.Reload()
                        if metric in ea.Tags()['scalars']:
                            events = ea.Scalars(metric)
                            row.append(events[-1].value)  # Get the last value
                        else:
                            row.append(None)
                    except Exception as e:
                        print(f"Error loading data for {lattice} - {config_name}: {str(e)}")
                        row.append(None)
                else:
                    row.append(None)
            if any(v is not None for v in row):  # Only add row if it has any valid values
                final_values.append(row)
        
        if final_values:  # Only create heatmap if we have data
            df = pd.DataFrame(
                final_values,
                index=[lt for lt in lattice_types if any(row for row in final_values)],
                columns=[cfg for cfg in config_dir_map.keys()]
            )
            
            # Drop any columns that are all None
            df = df.dropna(axis=1, how='all')
            
            if not df.empty:
                plt.figure(figsize=(15, 8))
                sns.heatmap(df, annot=True, cmap='YlOrRd', fmt='.4f')
                plt.title(f'Final {metric.replace("/", " ")} Across Configurations')
                plt.tight_layout()
                plt.savefig(f'analysis/figures/heatmap_{metric.replace("/", "_")}.png')
                plt.close()

def analyze_results():
    """Analyze results and create visualizations."""
    lattice_types = ['Cubic', 'BCC', 'FCC']
    all_results = {}
    
    # Map config names to actual directory patterns
    config_dir_map = {
        'baseline': 'lr_0.001_bs_1024_epochs_100_patches_3000',
        'lower_lr': 'lr_0.0001_bs_1024_epochs_100_patches_3000',
        'small_batch': 'lr_0.001_bs_256_epochs_100_patches_3000',
        'large_batch': 'lr_0.002_bs_2048_epochs_100_patches_3000',
        'more_patches': 'lr_0.001_bs_1024_epochs_100_patches_4900',
        'extended_training': 'lr_0.001_bs_1024_epochs_200_patches_3000',
        'lr_decay': 'lr_0.001_bs_1024_epochs_100_patches_3000'
    }
    
    # Collect all results
    for lattice in lattice_types:
        all_results[lattice] = {}
        for config_name, dir_pattern in config_dir_map.items():
            try:
                matching_dirs = [d for d in os.listdir(f'runs/autoencoder_experiment/{lattice}') 
                               if d.startswith(dir_pattern)]
                if matching_dirs:
                    log_path = os.path.join('runs/autoencoder_experiment', lattice, matching_dirs[0])
                    try:
                        print(f"Loading data from: {log_path}")
                        data = extract_tensorboard_data(log_path)
                        # Keep the original metric names with prefixes
                        all_results[lattice][config_name] = data
                    except Exception as e:
                        print(f"Error loading data for {lattice} - {config_name}: {str(e)}")
                        all_results[lattice][config_name] = {}
                else:
                    print(f"No matching directory found for {lattice} - {config_name} with pattern {dir_pattern}")
            except Exception as e:
                print(f"Error accessing directory for {lattice}: {str(e)}")
    
    # Create plots
    save_path = 'analysis/figures/'
    os.makedirs(save_path, exist_ok=True)
    
    metrics = ['Loss/train', 'Loss/val', 'Reconstruction_MSE', 'Reconstruction_Similarity']
    for metric in metrics:
        try:
            # Pass save_path explicitly
            create_training_plots(all_results, save_path=save_path)
            print(f"Created training plot for {metric}")
        except Exception as e:
            print(f"Error creating training plot for {metric}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Create heatmaps
    try:
        create_performance_heatmap(all_results)
        print("Created performance heatmap")
    except Exception as e:
        print(f"Error creating performance heatmap: {str(e)}")
    
    return all_results

def run_all_configs(resume_training=False):
    """Run all configurations for each lattice type."""
    lattice_types = ['Cubic', 'BCC', 'FCC']
    
    for lattice_type in lattice_types:
        print(f"\nStarting lattice type: {lattice_type}")
        for config in configs:
            print(f"\nStarting configuration: {config['name']}")
            
            train_autoencoder(
                learning_rate=config['learning_rate'],
                batch_size=config['batch_size'],
                num_epochs=config['epochs'],  # Changed from 'epochs' to 'num_epochs'
                patches_per_image=config['patches_per_image'],
                lattice_type=lattice_type,
                resume_training=resume_training
            )
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print(f"Completed configuration: {config['name']}")
        print(f"Completed lattice type: {lattice_type}\n")

def save_metrics(lattice_type, config_name, metrics):
    """Save metrics to a JSON file in a structured format."""
    results_file = 'analysis/metrics/performance_metrics.json'
    try:
        with open(results_file, 'r') as f:
            all_results = json.load(f)
    except FileNotFoundError:
        all_results = defaultdict(lambda: defaultdict(dict))
    
    # Update results
    if lattice_type not in all_results:
        all_results[lattice_type] = {}
    if config_name not in all_results[lattice_type]:
        all_results[lattice_type][config_name] = {}
    
    all_results[lattice_type][config_name].update(metrics)
    
    # Save updated results
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=4)

def generate_latex_table():
    """Generate a LaTeX table from saved metrics."""
    results_file = 'analysis/metrics/performance_metrics.json'
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    latex_table = []
    latex_table.append(r'\begin{table}[h]')
    latex_table.append(r'\centering')
    latex_table.append(r'\begin{tabular}{|c|c|c|c|}')
    latex_table.append(r'\hline')
    latex_table.append(r'Lattice Type & Configuration & Average MSE & Average SSIM \\')
    latex_table.append(r'\hline')
    
    for lattice_type in ['Cubic', 'BCC', 'FCC']:
        for config_name, metrics in results[lattice_type].items():
            row = f"{lattice_type} & {config_name} & {metrics['final_mse']:.4f} & {metrics['final_ssim']:.4f} \\\\"
            latex_table.append(row)
            latex_table.append(r'\hline')
    
    latex_table.append(r'\end{tabular}')
    latex_table.append(r'\caption{Performance metrics across different lattice types and configurations}')
    latex_table.append(r'\label{tab:performance_metrics}')
    latex_table.append(r'\end{table}')
    
    # Save the LaTeX table
    with open('analysis/metrics/performance_table.tex', 'w') as f:
        f.write('\n'.join(latex_table))
        
if __name__ == "__main__":
    choice = input("Do you want to (1) run all configurations or (2) analyze existing results? ")
    
    if choice == "1":
        run_all_configs()
        print("All configurations completed. Now analyzing results...")
        analyze_results()
    elif choice == "2":
        analyze_results()
    else:
        print("Invalid choice. Please enter 1 or 2.")