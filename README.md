# DualDEC-STM: Dual Decoder Enhanced Convolutional Auto-Encoder for STM Images

## Project Overview

DualDEC-STM is a convolutional autoencoder with a novel dual decoder architecture, designed for high-fidelity reconstruction of scanning tunneling microscopy (STM) images. It leverages advanced non-linear activations (e.g., Swish, Mish, GELU) in the encoder and separate decoders for general image reconstruction and feature-specific enhancement.

## Research Objectives

1. **Develop a Dual Decoder Autoencoder**
   - Integrate non-linear activations for improved image reconstruction

2. **Quantify Reconstruction Quality and Feature Extraction**
   - Compare with traditional single-decoder models using PSNR, SSIM, and atomic position accuracy

3. **Optimize Latent Space Compression**
   - Identify compression ratios that retain ≥95% reconstruction fidelity

## Methodology

### Architecture
- Encoder with advanced activations
- Two decoders for distinct reconstruction goals

### Training
- Custom loss functions
- Adam optimizer
- Regular validation

### Evaluation
- Benchmark against single-decoder models using metrics such as PSNR and SSIM

## Resources

### Dependencies
- PyTorch
- NumPy
- scikit-image

### Hardware
- University HPC cluster

### Mentorship
- Dr. Peter Binev, Department of Mathematics

## Usage

### 1. Clone the Repository

```bash
git clone https://github.com/joshuamoorehead/DualDEC-STM.git
cd DualDEC-STM
```

### 2. Install Dependencies

Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate STM_AE
```

### 3. Run Training or Analysis

Navigate to the conv_ae directory and run the training script:
```bash
cd conv_ae
python train.py
```

The script will prompt you to:
1. Run all configurations (Option 1)
   - Trains models for all lattice types with various hyperparameter configurations:
     - Baseline (lr=0.001, batch_size=1024, epochs=100)
     - Lower Learning Rate (lr=0.0001)
     - Small Batch Size (batch_size=256)
     - Large Batch Size (batch_size=2048)
     - More Patches (patches_per_image=4900)
     - Extended Training (epochs=200)
     - Learning Rate Decay

2. Analyze existing results (Option 2)
   - Generates visualization plots and performance metrics for trained models
   - Creates loss curves and reconstruction quality comparisons
   - Outputs performance metrics in LaTeX table format

Results and model checkpoints will be saved in:
- `runs/autoencoder_experiment/` - TensorBoard logs
- `models/weights/` - Model checkpoints
- `analysis/figures/` - Generated plots and visualizations
- `analysis/metrics/` - Performance metrics and analysis
## Project Significance

DualDEC-STM enhances STM image analysis, bridging deep learning and nanoscale imaging, with potential applications in nanotechnology, metallurgy, and quantum computing.

## Contributors

- **Joshua Moorehead**
  - Computer Engineering
  - Business Administration

- **Rori Pumphrey**
  - Mechanical Engineering
  - Mathematics

- **Mentor**: Dr. Peter Binev
  - Department of Mathematics

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

