# Single Decoder STM Convolutional Autoencoder

This directory contains the implementation of a single decoder convolutional autoencoder for analyzing Scanning Tunneling Microscopy (STM) images. This architecture serves as the baseline model for our STM image analysis research.

## Directory Structure

```
single_decoder_cae/
├── analysis/              # Analysis results and visualizations
│   ├── figures/          # Generated plots and visualizations
│   └── metrics/          # Performance metrics data
├── models/               # Model weights and checkpoints
│   └── weights/          # Saved model weights for each lattice type
├── runs/                 # TensorBoard logs
├── tracking/            # Tracked patches for consistent evaluation
├── datasets.py          # Data loading and patch extraction
├── gputest.py          # GPU availability and performance testing
├── models.py           # Model architecture definition
├── train.py            # Main training script
└── utils.py            # Utility functions
```

## Key Components

### Core Files
- `models.py`: Implements the single decoder convolutional autoencoder architecture
  ```python
  class Autoencoder(nn.Module):
      # Encoder: 17x17 -> 10 latent dimensions
      # Decoder: 10 -> 17x17 reconstruction
  ```

- `datasets.py`: Handles data loading and patch extraction
  - Implements FlatImageDataset for loading STM images
  - Supports random patch extraction
  - Handles different lattice types (Cubic, BCC, FCC)

- `train.py`: Main training script with functions for:
  - Training single configurations
  - Running multiple configurations
  - Analyzing results
  - Generating visualizations
  - Performance metrics tracking

- `utils.py`: Utility functions for:
  - GPU management
  - Metric calculations
  - Visualization helpers

### Configuration Options
The following training configurations are available:
```python
configs = [
    "baseline":    {lr: 0.001, batch_size: 1024, epochs: 100},
    "lower_lr":    {lr: 0.0001, batch_size: 1024, epochs: 100},
    "small_batch": {lr: 0.001, batch_size: 256, epochs: 100},
    "large_batch": {lr: 0.002, batch_size: 2048, epochs: 100},
    "more_patches": {lr: 0.001, batch_size: 1024, epochs: 100, patches: 4900},
    "extended_training": {lr: 0.001, batch_size: 1024, epochs: 200},
    "lr_decay":    {lr: 0.001, batch_size: 1024, epochs: 100}
]
```

## Usage

1. Train a single configuration:
```bash
python train.py
# Select option 1 and follow prompts
```

2. Run all configurations:
```bash
python train.py
# Select option 2
```

3. Analyze existing results:
```bash
python train.py
# Select option 3
```

## Performance Monitoring
- Training progress is logged to TensorBoard in the `runs/` directory
- Model checkpoints are saved in `models/weights/`
- Analysis results are stored in `analysis/`
- Patch tracking for consistent evaluation in `tracking/`

## Key Features
- Supports multiple lattice types (Cubic, BCC, FCC)
- Implements patch-based training
- Uses MSE loss for reconstruction
- Includes SSIM for quality assessment
- Provides comprehensive visualization tools
- Supports training resumption from checkpoints

## Dependencies
```
torch
torchvision
numpy
matplotlib
seaborn
tensorboard
scikit-image
pandas
```

## Results Directory Structure
```
analysis/
├── figures/
│   ├── loss_curves/
│   ├── reconstructions/
│   └── heatmaps/
└── metrics/
    ├── performance_metrics.json
    └── performance_table.tex
```

## GPU Testing
Use `gputest.py` to verify GPU availability and performance:
```bash
python gputest.py
```

## Notes
- This implementation serves as the baseline for comparison with the dual decoder architecture
- Model architecture focuses on general reconstruction quality
- All results and analysis are reproducible through saved configurations
- Training progress can be monitored in real-time through TensorBoard

