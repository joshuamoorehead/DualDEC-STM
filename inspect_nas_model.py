import torch
import numpy as np
import matplotlib.pyplot as plt

# Path to your saved model
model_path = "runs/nas_experiment/Cubic/lr_0.001_bs_64_nas_epochs_20_patches_500/best_nas_model.pt"

# Load the checkpoint
checkpoint = torch.load(model_path)

# Define operation names for each mixed operation type
conv_ops = ["3x3 Conv", "5x5 Conv", "3x3 Dilated Conv", "Depthwise Sep Conv"]
act_ops = ["ReLU", "LeakyReLU"]
up_ops = ["Transposed Conv", "Bilinear Upsampling", "Nearest Upsampling", "Pixel Shuffle"]

# Function to map alpha parameter name to operation names
def get_operation_names(param_name):
    if "mixed_op" in param_name:
        return conv_ops
    elif "mixed_act" in param_name:
        return act_ops
    elif "up" in param_name:
        return up_ops
    return [f"Option {i}" for i in range(len(param))]  # Fallback

# Extract and interpret architecture parameters
print("\n=== Selected Architecture Choices ===\n")
arch_choices = {}
selected_ops = {}

for name, param in checkpoint['model_state_dict'].items():
    if 'alpha' in name:
        # Get the corresponding operation names
        op_names = get_operation_names(name)
        
        # Convert to numpy for easier handling
        weights = param.cpu().numpy()
        
        # Print weights for each option
        component = name.replace(".alpha", "")
        print(f"\n{component}:")
        for i, weight in enumerate(weights):
            print(f"  {op_names[i]}: {weight:.4f}")
        
        # Get the selected operation (highest weight)
        max_op_idx = np.argmax(weights)
        selected_op = op_names[max_op_idx]
        print(f"  → Selected: {selected_op} (weight: {weights[max_op_idx]:.4f})")
        
        # Store for summary
        selected_ops[component] = selected_op

# Print a clean summary of architecture choices
print("\n\n=== ARCHITECTURE SUMMARY ===\n")

print("ENCODER:")
for name, op in selected_ops.items():
    if "encoder" in name:
        print(f"  {name}: {op}")

print("\nCENTER DECODER:")
for name, op in selected_ops.items():
    if "center_decoder" in name:
        print(f"  {name}: {op}")

print("\nNOISE DECODER:")
for name, op in selected_ops.items():
    if "noise_decoder" in name:
        print(f"  {name}: {op}")

# Print performance metrics
if 'metrics' in checkpoint:
    metrics = checkpoint['metrics']
    print("\nBEST PERFORMANCE:")
    print(f"  Validation Loss: {min(metrics['val_loss']):.4f}")
    print(f"  SSIM: {max(metrics['ssim']):.4f}")
    print(f"  PSNR: {max(metrics['psnr']):.2f} dB")
