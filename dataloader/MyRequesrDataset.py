import numpy as np
import torch
from torch.utils.data import IterableDataset

class MyIterableDataset(IterableDataset):
    def __init__(self, data_path, block_size, batch_size):
        self.data_path = data_path
        self.block_size = block_size
        self.data = np.memmap(self.data_path, dtype=np.uint16, mode='r')

    def __iter__(self):
        # 假设 data 是一个文件，按需逐块读取
        

        total_length = len(self.data) - self.block_size
        
        while True:  # 无限生成数据块
            # 随机生成起始索引
            ix = torch.randint(0, total_length, (1,)).item()
            
            # 提取输入块 x 和目标块 y
            x = torch.from_numpy(self.data[ix:ix + self.block_size].astype(np.int64))
            y = torch.from_numpy(self.data[ix + 1:ix + 1 + self.block_size].astype(np.int64))
            
            # 使用 yield 返回结果

            yield x, y