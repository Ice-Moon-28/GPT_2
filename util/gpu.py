import torch

def print_gpu_info(device):
    if device == 'cuda':
        print(f"当前显存占用: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"显存缓存: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    elif device == 'mps':
        print("MPS 后端当前无法获取显存信息。")
