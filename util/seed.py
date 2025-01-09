import random
import numpy as np
import torch


def set_random_seed(seed, deterministic=False):
    """
    固定 PyTorch、NumPy 和 Python 的随机种子，以确保结果的可重复性。
    
    :param seed: 随机种子值（整数）
    :param deterministic: 是否设置 PyTorch 的确定性行为（默认False）。
                          如果为 True，将启用确定性算法，可能会影响性能。
    """
    # Python 内置 random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多 GPU 时设置所有 GPU 的种子
    
    # 确保 PyTorch 的确定性行为（可选）
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    print(f"Random seed set to {seed} (deterministic={deterministic})")