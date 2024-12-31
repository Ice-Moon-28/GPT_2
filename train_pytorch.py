"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import wandb

import math
import pickle
from contextlib import nullcontext

from dataloader.openWebTextDataSet import create_dataloader
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT
from model_gpt2.gpt2 import get_model
from model_gpt2.scheduler import CustomLRScheduler
from util.gpu import print_gpu_info
from transformers import GPT2LMHeadModel, GPT2Config

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
wandb_key = None
# data
dataset = 'openwebtext'
data_dir = os.path.join('data', dataset)
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
num_workers = 1
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster

config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging

meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

iter_num = 0
best_val_loss = 1e9

def init_model(init_from='scratch'):
    if init_from == 'scratch':
        model_args = GPT2Config(
            vocab_size=meta_vocab_size if meta_vocab_size is not None else 50304,
            n_positions=1024,  # 输入序列的最大长度
            n_ctx=1024,  # 上下文窗口大小
            n_embd=768,  # 嵌入层大小
            n_layer=12,  # Transformer 层数
            n_head=12,  # 自注意力头数
            dropout=0.1,  # Dropout 比例
        )

        model = get_model(model_args)

    elif init_from == 'resume':
        print(f"Resuming training from {out_dir}")
        # resume training from a checkpoint.
        ckpt_path = os.path.join(out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        checkpoint_model_args = checkpoint['model_args']
        
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = checkpoint_model_args[k]
        # create the model
        gptconf = GPT2Config(**model_args)
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        global iter_num
        iter_num = checkpoint['iter_num']
        global best_val_loss
        best_val_loss = checkpoint['best_val_loss']

    return model


def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
    """
    Configure the optimizer for the model.

    Parameters:
        weight_decay (float): Weight decay for the optimizer.
        learning_rate (float): Learning rate for the optimizer.
        betas (tuple): Tuple of (beta1, beta2) for the Adam optimizer.
        device_type (str): The type of device to use (e.g., 'cuda' or 'cpu').

    Returns:
        torch.optim.AdamW: The AdamW optimizer.
    """
    optimizer = torch.optim.AdamW(
        self.parameters(), 
        lr=learning_rate, 
        betas=betas, 
        weight_decay=weight_decay
    )

    return optimizer

def init_wandb(wandb_log, wandb_project, wandb_run_name, config, master_process):
    if wandb_log and master_process:
        if wandb_key:
            wandb.login(key=wandb_key)
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

@torch.no_grad()
def estimate_loss(model, train_dataloader, eval_dataloader, eval_iters=200, device_type='cuda'):
    out = {}
    model.eval()

    losses = torch.zeros(eval_iters)

    for batch_idx in range(eval_iters):
        if batch_idx <= eval_iters:
            x, y = get_batch(train_dataloader, device_type=device_type)
            loss = model(x, y)
            loss = loss.item()
            losses[batch_idx] = loss.item()


    out['train'] = losses.mean()
    
    for batch_idx in range(eval_iters):
        if batch_idx <= eval_iters:
            x, y = get_batch(eval_dataloader, device_type=device_type)
            loss = model(x, y)
            loss = loss.item()
            losses[batch_idx] = loss.item()

    out['val'] = losses.mean()

    model.train()

    return out

def get_batch(dataloader, device_type):
    x, y = next(iter(dataloader))
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
  
    return x, y

def train(
        model,
        optimizer,
        scheduler,
        train_dataloader,
        val_dataloader,
        scaler,
        num_iterions,
        log_interval,
        eval_interval,
        gradient_accumulation_steps,
        device,
        ddp=False,
        wandb_log=False,
        model_args=None,
        master_process=False,
        ctx=None,
        device_type='cuda'
    ):
    """
    Training loop for the model.
    """
    global iter_num
    global best_val_loss
    
    local_iter_num = 0  # number of iterations in the lifetime of this process
    running_mfu = -1.0
    t0 = time.time()

    print_gpu_info(device)

    raw_model = model.module if ddp else model

    for batch_idx in range(num_iterions):
    
        for micro_step in range(gradient_accumulation_steps):
            if ddp:
                # in DDP training we only need to sync gradients at the last micro step.
                # the official way to do this is with model.no_sync() context manager, but
                # I really dislike that this bloats the code and forces us to repeat code
                # looking at the source of that context manager, it just toggles this variable
                model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
            with ctx:
                X, Y = get_batch(train_dataloader, device_type)

                output = model(X, labels=Y) 

                loss = output.loss

                loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation

            
                scaler.scale(loss).backward()
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        scheduler.step()

        if iter_num % eval_interval == 0 and master_process:
            losses = estimate_loss(model=model, train_dataloader=train_dataloader, eval_dataloader=val_dataloader, device_type=device_type)
            
            if device == 'cuda':
                torch.cuda.empty_cache()
            elif device == 'mps':
                torch.mps.empty_cache()

            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if wandb_log:
                wandb.log({
                    "iter": iter_num,
                    "train/loss": losses['train'],
                    "val/loss": losses['val'],
                    "lr": optimizer.param_groups[0]['lr'],
                    "mfu": running_mfu * 100,  # convert to percentage
                })

            if losses['val'] < best_val_loss or always_save_checkpoint:
                best_val_loss = losses['val']
                if iter_num > 0:
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': config,
                    }
                    print(f"saving checkpoint to {out_dir}")
                    torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

        
        optimizer.zero_grad(set_to_none=True)

        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % log_interval == 0 and master_process:
            # get loss as float. note: this is a CPU-GPU sync point
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
            lossf = loss.item() * gradient_accumulation_steps
            if local_iter_num >= 5: # let the training loop settle a bit
                mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        iter_num += 1
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
        local_iter_num += 1
    if ddp:
        destroy_process_group()

    

def main():

    global gradient_accumulation_steps
    global device

    # various inits, derived attributes, I/O setup
    ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
    print(f"DDP RUNNING: {ddp}")
    if ddp:
        init_process_group(backend=backend)
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
        seed_offset = ddp_rank # each process gets a different seed
        # world_size number of processes will be training simultaneously, so we can scale
        # down the desired gradient accumulation iterations per process proportionally
        assert gradient_accumulation_steps % ddp_world_size == 0
        gradient_accumulation_steps //= ddp_world_size
    else:
        # if not ddp, we are running on a single gpu, and one process
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
    tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
    print(f"tokens per iteration will be: {tokens_per_iter:,}")

    train_loader = create_dataloader('train', data_dir, block_size, batch_size, device, num_workers=num_workers)
    val_loader = create_dataloader('val', data_dir, block_size, batch_size, device, num_workers=num_workers)

    if master_process:
        os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)


    model = init_model(init_from=init_from)

    if compile and torch.backends.mps.is_available():
        print("Compiling the model is skipped due to MPS backend issues.")
    elif compile:
        print("compiling the model... (takes a ~minute)")
        model = torch.compile(model, backend='inductor') # requires PyTorch 2.0

    model.to(device)


    optimizer = configure_optimizers(model, weight_decay, learning_rate, [beta1, beta2], device_type)

    scheduler = CustomLRScheduler(optimizer, warmup_iters=warmup_iters, lr_decay_iters=lr_decay_iters, 
                              learning_rate=learning_rate, min_lr=min_lr)
    
    init_wandb(wandb_log, wandb_project, wandb_run_name, config=config, master_process=master_process)

    print_gpu_info(device)

    train(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        eval_interval=eval_interval,
        log_interval=log_interval,
        gradient_accumulation_steps=gradient_accumulation_steps,
        ddp=ddp,
        wandb_log=wandb_log,
        model_args=init_from,
        device=device,
        num_iterions=max_iters,
        scaler=scaler,
        master_process=master_process,
        ctx=ctx,
        device_type=device_type
    )


if __name__ == '__main__':
    main() 
    

    