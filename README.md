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
git clone https://github.com/yourusername/DualDEC-STM.git
cd DualDEC-STM
```

### 2. Install Dependencies

Ensure Python and required libraries are installed:

```bash
pip install -r requirements.txt
```

### 3. Train the Model

Run training on your STM dataset:

```bash
python train.py --config configs/dual_decoder.yaml
```

### 4. Evaluate the Model

Use evaluation scripts for reconstruction quality:

```bash
python evaluate.py --model_path checkpoints/model_best.pth
```

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

## Acknowledgments

Special thanks to the University HPC cluster and the AI Institute for their support.
