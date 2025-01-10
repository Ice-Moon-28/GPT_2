# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

import time


wandb_log = True
wandb_key = '1c1fa66d79864363e5f33bb705a768da6cf094e5'
wandb_project = 'GPT2'
wandb_run_name= 'run' + str(time.time())

device = 'cuda'

data_dir = 'data/openwebtext/1'
compile = True
# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 16
block_size = 1024
gradient_accumulation_steps = 10

# this makes total number of tokens be 300B
max_iters = 600000
lr_decay_iters = 600000

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1
num_workers = 8
enable_dataloader = True # 是否开启 dataloader 来加载数据集