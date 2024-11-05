import os
import torch
import subprocess

def run_command(command):
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
        return result.stdout
    except Exception as e:
        return f"Error running command: {e}"

print("=== GPU Availability ===")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")

print("\n=== Environment Variables ===")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
print(f"NVIDIA_VISIBLE_DEVICES: {os.environ.get('NVIDIA_VISIBLE_DEVICES', 'Not set')}")

print("\n=== GPU Information ===")
for i in range(torch.cuda.device_count()):
    print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"  Compute Capability: {torch.cuda.get_device_capability(i)}")
    total_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
    allocated_memory = torch.cuda.memory_allocated(i) / 1e9
    cached_memory = torch.cuda.memory_reserved(i) / 1e9
    free_memory = total_memory - (allocated_memory + cached_memory)
    print(f"  Total memory: {total_memory:.2f} GB")
    print(f"  Allocated memory: {allocated_memory:.2f} GB")
    print(f"  Cached memory: {cached_memory:.2f} GB")
    print(f"  Free memory: {free_memory:.2f} GB")

print("\n=== Memory Allocation Test ===")
for i in range(torch.cuda.device_count()):
    torch.cuda.set_device(i)
    try:
        # Try to allocate 1 GB
        tensor = torch.empty(1024 * 1024 * 256, device=f'cuda:{i}')
        print(f"Successfully allocated 1 GB on GPU {i}")
        del tensor
        torch.cuda.empty_cache()
    except RuntimeError as e:
        print(f"Failed to allocate 1 GB on GPU {i}: {e}")

print("\n=== NVIDIA-SMI Output ===")
print(run_command("nvidia-smi"))

print("\n=== PyTorch Build Information ===")
print(torch.__config__.show())

print("\n=== Current Process Information ===")
print(run_command("ps -p " + str(os.getpid()) + " -o pid,ppid,user,%cpu,%mem,cmd"))

print("\n=== System Information ===")
print(run_command("uname -a"))
print(run_command("lscpu | grep 'Model name'"))

print("\n=== Python and PyTorch Versions ===")
print(f"Python version: {os.sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"PyTorch debug build: {torch.version.debug}")