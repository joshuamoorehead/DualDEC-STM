import subprocess
import torch

def get_available_gpus(min_free_memory_gb=1):
    """
    This function checks for available GPUs and returns a list of device IDs of GPUs with sufficient free memory.
    If no GPUs are available or have enough memory, it returns ['cpu'].
    """
    available_gpus = []
    if torch.cuda.is_available():
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.free,memory.total', '--format=csv,nounits,noheader'], 
                                stdout=subprocess.PIPE, encoding='utf-8')
        memory_info = result.stdout.strip().split('\n')
        for i, info in enumerate(memory_info):
            free_memory, total_memory = map(int, info.split(','))
            free_memory_gb = free_memory / 1024
            print(f'GPU {i}: {free_memory_gb:.2f} GB free memory')
            if free_memory_gb >= min_free_memory_gb:
                available_gpus.append(torch.device(f'cuda:{i}'))
    
    return available_gpus if available_gpus else ['cpu']
