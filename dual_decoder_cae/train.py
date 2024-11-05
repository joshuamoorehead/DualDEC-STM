import torch
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import time
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from models import DualDecoderAE
from losses import DualDecoderLoss
from datasets import load_data_automated
from utils.metrics import DualMetrics
from utils.visualization import DualVisualizer
from utils.latent_analysis import LatentAnalyzer
from config.hyperparameters import default_config
import os
import torchvision
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
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
    print(f"GPU {device}: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB / "
          f"{torch.cuda.get_device_properties(device).total_memory / 1024**3:.2f} GB")

def log_reconstructed_patches(model, data_loader, writer, visualizer, epoch, lattice_type, device, num_patches=3):
    model.eval()
    metrics = DualMetrics(device)
    
    try:
        # Update file paths to use the new directory structure
        tracking_dir = os.path.join('tracking', lattice_type)
        os.makedirs(tracking_dir, exist_ok=True)
        tracked_patches_file = os.path.join(tracking_dir, f'tracked_patches.pt')
        
        # Get or create tracked patches
        if epoch == 0 or not os.path.exists(tracked_patches_file):
            batch = next(iter(data_loader))
            if isinstance(batch, tuple):
                original_patches, centered_patches = batch
            else:
                original_patches = batch
            total_patches = original_patches.size(0)
            tracked_indices = torch.randperm(total_patches)[:num_patches]
            tracked_patches = original_patches[tracked_indices]
            torch.save(tracked_patches, tracked_patches_file)
        else:
            tracked_patches = torch.load(tracked_patches_file)
        
        original_patches = tracked_patches.to(device)
        
        with torch.no_grad():
            noise_output, center_output, latent = model(original_patches, return_latent=True)
        
        # Calculate metrics
        noise_metrics = metrics.compute_noise_metrics(noise_output, original_patches)
        center_metrics = metrics.compute_centering_metrics(center_output, original_patches)
        
        # Log metrics
        for name, value in noise_metrics.items():
            writer.add_scalar(f'{lattice_type}/Noise_{name}', value, epoch)
        for name, value in center_metrics.items():
            writer.add_scalar(f'{lattice_type}/Center_{name}', value, epoch)
        
        # Visualize reconstructions
        visualizer.visualize_batch(original_patches, noise_output, center_output, epoch, lattice_type)
        
        # Log latent space visualization if it's a visualization epoch
        if epoch % 10 == 0:
            visualizer.visualize_latent_space(latent, epoch, lattice_type)
        
        return {**noise_metrics, **center_metrics}
        
    except Exception as e:
        print(f"Error in log_reconstructed_patches: {str(e)}")
        return {}

def train_autoencoder(learning_rate, batch_size, num_epochs, patches_per_image, lattice_type, resume_training=False):
    start_time = time.time()
    
    # Load data
    train_loader, val_loader, test_loader, _ = load_data_automated(
        data_root="../Data",
        batch_size=batch_size,
        transform=transform,
        patches_per_image=patches_per_image,
        lattice_type=lattice_type
    )
    
    # Initialize model, loss, and optimizer
    model = DualDecoderAE().to(device)
    criterion = DualDecoderLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Setup directories
    weights_dir = os.path.join('models', 'weights', lattice_type)
    os.makedirs(weights_dir, exist_ok=True)
    run_name = f'lr_{learning_rate}_bs_{batch_size}_epochs_{num_epochs}_patches_{patches_per_image}'
    log_dir = os.path.join('runs', 'dual_decoder_experiment', lattice_type, run_name)
    
    # Initialize tracking
    writer = SummaryWriter(log_dir=log_dir)
    visualizer = DualVisualizer(writer)
    metrics_tracker = DualMetrics(device)
    latent_analyzer = LatentAnalyzer(writer)
    
    # Resume training if requested
    start_epoch = 0
    best_loss = float('inf')
    checkpoint_path = os.path.join(weights_dir, 'checkpoint.pt')
    
    if resume_training and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['loss']
        print(f"Resuming training from epoch {start_epoch}")
    
    # Learning rate scheduler
    scheduler = None
    if run_name.startswith('lr_decay'):
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        if resume_training and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print("Using learning rate decay scheduler")
    
    print(f"Starting training for {num_epochs} epochs")
    
    try:
        # Training loop
        for epoch in range(start_epoch, num_epochs):
            epoch_start_time = time.time()
            model.train()
            train_loss = 0.0
            noise_loss_sum = 0.0
            center_loss_sum = 0.0
            
            # Training step
            for batch_idx, (original_patches, centered_patches) in enumerate(train_loader):
                original_patches = original_patches.to(device)
                centered_patches = centered_patches.to(device)
                
                # Forward pass
                noise_output, center_output = model(original_patches)
                
                # Compute losses
                loss, noise_loss, center_loss = criterion(noise_output, center_output, original_patches)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Track losses
                train_loss += loss.item() * original_patches.size(0)
                noise_loss_sum += noise_loss.item() * original_patches.size(0)
                center_loss_sum += center_loss.item() * original_patches.size(0)
                
                if batch_idx % 10 == 0:
                    print(f'Epoch: {epoch}/{num_epochs} [{batch_idx}/{len(train_loader)}] '
                          f'Loss: {loss.item():.6f}')
            
            # Calculate average losses
            train_loss = train_loss / len(train_loader.dataset)
            avg_noise_loss = noise_loss_sum / len(train_loader.dataset)
            avg_center_loss = center_loss_sum / len(train_loader.dataset)
            
            # Log training metrics
            writer.add_scalar(f'{lattice_type}/Loss/train/total', train_loss, epoch)
            writer.add_scalar(f'{lattice_type}/Loss/train/noise', avg_noise_loss, epoch)
            writer.add_scalar(f'{lattice_type}/Loss/train/center', avg_center_loss, epoch)
            
            # Step scheduler if using lr_decay
            if scheduler is not None:
                scheduler.step()
                writer.add_scalar('Learning_Rate', scheduler.get_last_lr()[0], epoch)
            
            # Validation step
            if val_loader is not None:
                model.eval()
                val_metrics = validate(model, val_loader, criterion, device)
                
                # Log validation metrics
                for name, value in val_metrics.items():
                    writer.add_scalar(f'{lattice_type}/Loss/val/{name}', value, epoch)
                
                print(f'Epoch: {epoch+1}/{num_epochs}')
                print(f'Training Loss: {train_loss:.6f}')
                print(f'Validation Loss: {val_metrics["total"]:.6f}')
                print(f'Time: {time.time() - epoch_start_time:.2f}s')
                
                # Save best model
                if val_metrics["total"] < best_loss:
                    best_loss = val_metrics["total"]
                    print(f"New best loss: {best_loss:.6f}")
                    save_checkpoint(model, optimizer, epoch, best_loss, 
                                  scheduler,
                                  weights_dir, {
                                      'lr': learning_rate,
                                      'bs': batch_size,
                                      'patches': patches_per_image
                                  })
            
            # Visualization and analysis (every 10 epochs)
            if epoch % 10 == 0:
                metrics = log_reconstructed_patches(
                    model, train_loader, writer, visualizer,
                    epoch, lattice_type, device
                )
                latent_analyzer.run_full_analysis(
                    model, train_loader, device, epoch, lattice_type
                )
            
            # Print epoch summary
            epoch_time = time.time() - epoch_start_time
            print(f'Epoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f}s')
            print(f'Train Loss: {train_loss:.6f}')
            print(f'Noise Loss: {avg_noise_loss:.6f}')
            print(f'Center Loss: {avg_center_loss:.6f}')
            print('-' * 80)
            
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise
    finally:
        # Final evaluation
        print("Running final evaluation...")
        try:
            final_metrics = evaluate_model(model, test_loader, criterion, metrics_tracker, device)
            save_metrics(lattice_type, run_name, final_metrics)
            
            print("\nFinal Metrics:")
            for metric_name, value in final_metrics.items():
                print(f"{metric_name}: {value:.6f}")
            
        except Exception as e:
            print(f"Error during final evaluation: {str(e)}")
        
        writer.close()
        
    total_time = time.time() - start_time
    print(f'\nTraining completed in {total_time/3600:.2f} hours')
    
    return final_metrics

def validate(model, val_loader, criterion, device):
    model.eval()
    val_metrics = defaultdict(float)
    
    with torch.no_grad():
        for data in val_loader:
            if isinstance(data, tuple):
                patches, centered_patches = data
            else:
                patches = data
                centered_patches = data
                
            patches = patches.to(device)
            centered_patches = centered_patches.to(device)
            
            noise_output, center_output = model(patches)
            loss, noise_loss, center_loss = criterion(noise_output, center_output, patches)
            
            val_metrics["total"] += loss.item() * patches.size(0)
            val_metrics["noise"] += noise_loss.item() * patches.size(0)
            val_metrics["center"] += center_loss.item() * patches.size(0)
    
    # Calculate averages
    for key in val_metrics:
        val_metrics[key] /= len(val_loader.dataset)
    
    return val_metrics

def save_checkpoint(model, optimizer, epoch, loss, scheduler, weights_dir, hyperparams):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
        'hyperparameters': hyperparams
    }
    
    checkpoint_path = os.path.join(weights_dir, 'checkpoint.pt')
    torch.save(checkpoint, checkpoint_path)
    
    # Also save best model weights separately
    weights_path = os.path.join(weights_dir, 'best_model_weights.pth')
    torch.save(model.state_dict(), weights_path)

def evaluate_model(model, test_loader, criterion, metrics_tracker, device):
    model.eval()
    test_metrics = defaultdict(float)
    all_metrics = {}
    
    with torch.no_grad():
        for data in test_loader:
            if isinstance(data, tuple):
                patches, centered_patches = data
            else:
                patches = data
                centered_patches = data
                
            patches = patches.to(device)
            centered_patches = centered_patches.to(device)
            
            noise_output, center_output = model(patches)
            
            # Get all metrics
            noise_metrics = metrics_tracker.compute_noise_metrics(noise_output, patches)
            center_metrics = metrics_tracker.compute_centering_metrics(center_output, patches)
            
            # Update metrics
            for key, value in {**noise_metrics, **center_metrics}.items():
                test_metrics[key] += value * patches.size(0)
    
    # Calculate averages
    for key in test_metrics:
        test_metrics[key] /= len(test_loader.dataset)
    
    return dict(test_metrics)

def save_metrics(lattice_type, run_name, metrics):
    results_file = 'analysis/metrics/dual_decoder_performance.json'
    os.makedirs('analysis/metrics', exist_ok=True)
    
    try:
        with open(results_file, 'r') as f:
            all_results = json.load(f)
    except FileNotFoundError:
        all_results = {}
    
    if lattice_type not in all_results:
        all_results[lattice_type] = {}
    
    all_results[lattice_type][run_name] = metrics
    
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=4)

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
                num_epochs=config['epochs'],
                patches_per_image=config['patches_per_image'],
                lattice_type=lattice_type,
                resume_training=resume_training
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(f"Completed configuration: {config['name']}")
        print(f"Completed lattice type: {lattice_type}\n")

def analyze_results():
    """Analyze results and create visualizations."""
    lattice_types = ['Cubic', 'BCC', 'FCC']
    results = defaultdict(dict)

    metrics_file = 'analysis/metrics/dual_decoder_performance.json'
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            results = json.load(f)

        # Create visualizations using DualVisualizer
        writer = SummaryWriter('analysis/summary')
        visualizer = DualVisualizer(writer)

        for lattice_type in lattice_types:
            if lattice_type in results:
                lattice_results = results[lattice_type]
                visualizer.plot_comparative_results(lattice_results, lattice_type)
                visualizer.create_performance_heatmap(lattice_results, lattice_type)

        writer.close()
    else:
        print("No results found to analyze. Run training first.")

if __name__ == "__main__":
    print("DualDEC-STM Training Interface")
    print("------------------------------")
    print("1. Run single configuration")
    print("2. Run all configurations")
    print("3. Analyze existing results")
    print("4. Resume interrupted training")
    print("5. Run evaluation only")

    choice = input("\nEnter your choice (1-5): ")

    if choice == "1":
        print("\nSelect lattice type:")
        print("1. Cubic")
        print("2. BCC")
        print("3. FCC")
        lattice_choice = input("Enter choice (1-3): ")
        lattice_map = {"1": "Cubic", "2": "BCC", "3": "FCC"}
        lattice_type = lattice_map.get(lattice_choice)

        if lattice_type:
            print("\nSelect configuration:")
            for i, config in enumerate(configs, 1):
                print(f"{i}. {config['name']}")
            config_choice = input("Enter choice (1-7): ")

            try:
                config = configs[int(config_choice) - 1]
                train_autoencoder(
                    learning_rate=config['learning_rate'],
                    batch_size=config['batch_size'],
                    num_epochs=config['epochs'],
                    patches_per_image=config['patches_per_image'],
                    lattice_type=lattice_type
                )
                print("Training complete. Running analysis...")
                analyze_results()
            except (IndexError, ValueError):
                print("Invalid configuration choice.")
        else:
            print("Invalid lattice type choice.")

    elif choice == "2":
        print("Running all configurations for all lattice types...")
        run_all_configs()
        print("Training complete. Running analysis...")
        analyze_results()

    elif choice == "3":
        print("Analyzing existing results...")
        analyze_results()

    elif choice == "4":
        print("\nSelect lattice type to resume:")
        print("1. Cubic")
        print("2. BCC")
        print("3. FCC")
        lattice_choice = input("Enter choice (1-3): ")
        lattice_map = {"1": "Cubic", "2": "BCC", "3": "FCC"}
        lattice_type = lattice_map.get(lattice_choice)

        if lattice_type:
            print("\nSelect configuration to resume:")
            for i, config in enumerate(configs, 1):
                print(f"{i}. {config['name']}")
            config_choice = input("Enter choice (1-7): ")

            try:
                config = configs[int(config_choice) - 1]
                train_autoencoder(
                    learning_rate=config['learning_rate'],
                    batch_size=config['batch_size'],
                    num_epochs=config['epochs'],
                    patches_per_image=config['patches_per_image'],
                    lattice_type=lattice_type,
                    resume_training=True
                )
                print("Training complete. Running analysis...")
                analyze_results()
            except (IndexError, ValueError):
                print("Invalid configuration choice.")
        else:
            print("Invalid lattice type choice.")

    elif choice == "5":
        print("\nSelect lattice type for evaluation:")
        print("1. Cubic")
        print("2. BCC")
        print("3. FCC")
        lattice_choice = input("Enter choice (1-3): ")
        lattice_map = {"1": "Cubic", "2": "BCC", "3": "FCC"}
        lattice_type = lattice_map.get(lattice_choice)

        if lattice_type:
            # Load best model and evaluate
            model = DualDecoderAE().to(device)
            weights_path = os.path.join('models', 'weights', lattice_type, 'best_model_weights.pth')

            if os.path.exists(weights_path):
                model.load_state_dict(torch.load(weights_path))
                _, _, test_loader, _ = load_data_automated(
                    data_root="../Data",
                    batch_size=64,
                    transform=transform,
                    patches_per_image=3000,
                    lattice_type=lattice_type
                )

                criterion = DualDecoderLoss()
                metrics_tracker = DualMetrics(device)

                print("Evaluating model...")
                metrics = evaluate_model(model, test_loader, criterion, metrics_tracker, device)
                print("\nEvaluation Results:")
                for metric, value in metrics.items():
                    print(f"{metric}: {value:.6f}")
            else:
                print("No trained model found. Run training first.")
        else:
            print("Invalid lattice type choice.")

    else:
        print("Invalid choice. Please enter a number between 1 and 5.")
