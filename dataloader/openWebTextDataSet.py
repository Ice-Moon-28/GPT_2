import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from dataloader.MyRequesrDataset import MyIterableDataset

# Custom Dataset
class MemmapDataset(Dataset):
    def __init__(self, data_path, block_size):
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx:idx + self.block_size].astype(np.int64))
        y = torch.from_numpy(self.data[idx + 1:idx + 1 + self.block_size].astype(np.int64))
        return x, y

def create_dataset(split, data_dir, block_size, batch_size):
    data_path = os.path.join(data_dir, f"{split}.bin")
    dataset = MyIterableDataset(data_path, block_size, batch_size=batch_size)

    return dataset

# Function to create DataLoader
def create_dataloader(split, data_dir, block_size, batch_size, device_type, num_workers=2):
    
    dataset = create_dataset(split, data_dir, block_size, batch_size)

    # DataLoader with pinned memory for CUDA
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=(device_type == 'cuda'),
        num_workers=num_workers,
    )
    return iter(dataloader)