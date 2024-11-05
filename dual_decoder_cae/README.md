# DualDEC-STM: Dual Decoder Enhanced Convolutional Auto-Encoder for STM Images

This project implements a dual decoder convolutional autoencoder specifically designed for analyzing Scanning Tunneling Microscopy (STM) images. The architecture features two specialized decoders: one for noise reduction and another for structural feature centering.

## Directory Structure

```
dual_decoder_cae/
├── models/
│   ├── __init__.py          # Model exports
│   ├── dual_decoder.py      # Main dual decoder architecture
│   ├── noise_decoder.py     # Specialized noise reduction decoder
│   ├── center_decoder.py    # Specialized centering decoder
│   └── activation_layers.py # Custom activation functions
├── losses/
│   ├── __init__.py          # Loss function exports
│   ├── noise_loss.py        # Noise reduction loss components
│   ├── center_loss.py       # Centering loss components
│   └── combined_loss.py     # Combined loss function
├── utils/
│   ├── __init__.py          # Utility exports
│   ├── metrics.py           # Performance metrics
│   ├── visualization.py     # Visualization tools
│   └── latent_analysis.py   # Latent space analysis tools
├── config/
│   └── hyperparameters.py   # Model and training configurations
├── datasets.py              # Data loading and preprocessing
└── train.py                 # Training script
```

## Key Components

### Models
- `dual_decoder.py`: Implements the main DualDecoderAE architecture combining both decoders
- `noise_decoder.py`: Specialized decoder for noise reduction using wavelet analysis
- `center_decoder.py`: Specialized decoder for feature centering with position encoding
- `activation_layers.py`: Collection of advanced activation functions

### Losses
- `noise_loss.py`: Implements frequency domain and gradient-based losses for noise reduction
- `center_loss.py`: Implements structural and position-based losses for centering
- `combined_loss.py`: Combines both loss functions with adaptive weighting

### Utils
- `metrics.py`: Evaluation metrics for both noise reduction and centering quality
- `visualization.py`: Tools for visualizing reconstructions and training progress
- `latent_analysis.py`: Tools for analyzing the learned latent space representations

### Configuration and Data
- `hyperparameters.py`: Configuration settings for model architecture and training
- `datasets.py`: Data loading pipeline with support for patch extraction

### Training
- `train.py`: Main training script with support for:
  - Single configuration training
  - Multi-configuration experiments
  - Training resumption
  - Model evaluation
  - Results analysis

## Usage

1. Single Configuration Training:
```bash
python train.py
# Select option 1 and follow the prompts
```

2. Run All Configurations:
```bash
python train.py
# Select option 2
```

3. Analyze Results:
```bash
python train.py
# Select option 3
```

4. Resume Training:
```bash
python train.py
# Select option 4 and follow the prompts
```

5. Evaluate Model:
```bash
python train.py
# Select option 5 and follow the prompts
```

## Configuration Options

The following configurations are available:
- baseline: Standard training settings
- lower_lr: Lower learning rate for stability
- small_batch: Smaller batch size for memory efficiency
- large_batch: Larger batch size for faster training
- more_patches: Increased patches per image
- extended_training: Longer training duration
- lr_decay: Learning rate decay scheduling

## Dependencies

- PyTorch >= 1.8.0
- torchvision
- numpy
- scikit-learn
- matplotlib
- seaborn
- tensorboard
- scipy

## Notes

- The dual decoder architecture is designed to simultaneously address noise reduction and feature centering in STM images
- Each decoder is specialized for its specific task while sharing a common encoder
- The training process balances both objectives through adaptive loss weighting
- Results are logged to TensorBoard for easy visualization
- Checkpointing allows for training resumption and model persistence

## Paper Reference

For more details on the methodology and results, please refer to our paper [STM Image Analysis using Autoencoders](link-to-paper).
