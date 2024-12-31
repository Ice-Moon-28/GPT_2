

import os

from tqdm import tqdm
from dataloader.openWebTextDataSet import create_dataloader

dataset = 'openwebtext'
data_dir = os.path.join('data', dataset)
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
device = 'mps'
num_workers = 0


if __name__ == '__main__':
    train_loader = create_dataloader('train', data_dir, block_size, batch_size, device, num_workers=num_workers)

    print(train_loader, '1')

    for x, y in tqdm(train_loader, desc="Loading data"):
        print(train_loader, '2')
        print(x.shape, y.shape)
        break