import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Custom Dataset
class MemmapDataset(Dataset):
    def __init__(self, data_path, block_size):
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
    # 如果 idx 是一个 tensor，确保它是整数类型
        if isinstance(idx, torch.Tensor):
            idx = idx.long()  # 将 idx 转换为 long 类型（索引必须是 long）

        # 获取输入和目标块
        x = torch.stack([torch.from_numpy(self.data[i:i + self.block_size].astype(np.int64)) for i in idx])
        y = torch.stack([torch.from_numpy(self.data[i + 1:i + 1 + self.block_size].astype(np.int64)) for i in idx])
        
        return x, y

def create_dataset(split, data_dir, block_size):
    data_path = os.path.join(data_dir, f"{split}.bin")
    dataset = MemmapDataset(data_path, block_size)

    return dataset

# Function to create DataLoader
def create_dataloader(split, data_dir, block_size, batch_size, device_type, num_workers=2):
    
    dataset = create_dataset(split, data_dir, block_size)

    # DataLoader with pinned memory for CUDA
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=(device_type == 'cuda'),
        num_workers=num_workers,
    )
    return dataloader