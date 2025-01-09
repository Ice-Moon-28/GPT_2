import argparse

def get_args():
    parser = argparse.ArgumentParser(description="GPT-2 Training Configuration")

    # I/O
    parser.add_argument("--out_dir", type=str, default="out", help="Output directory")
    parser.add_argument("--eval_interval", type=int, default=2000, help="Evaluation interval")
    parser.add_argument("--log_interval", type=int, default=1, help="Log interval")
    parser.add_argument("--eval_iters", type=int, default=200, help="Number of evaluation iterations")
    parser.add_argument("--eval_only", action="store_true", help="Exit after first evaluation if set")
    parser.add_argument("--always_save_checkpoint", action="store_true", help="Always save checkpoint after evaluation")
    parser.add_argument("--init_from", type=str, choices=["scratch", "resume", "gpt2"], default="scratch", help="Initialization strategy")

    # wandb logging
    parser.add_argument("--wandb_log", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="owt", help="wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default="gpt2", help="wandb run name")
    parser.add_argument("--wandb_key", type=str, default=None, help="wandb API key")

    # data
    parser.add_argument("--dataset", type=str, default="openwebtext", help="Dataset name")
    parser.add_argument("--data_dir", type=str, default=None, help="Data directory (overrides default path)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=40, help="Gradient accumulation steps")
    parser.add_argument("--batch_size", type=int, default=12, help="Micro-batch size")
    parser.add_argument("--block_size", type=int, default=1024, help="Block size for token sequences")

    # model
    parser.add_argument("--n_layer", type=int, default=12, help="Number of layers")
    parser.add_argument("--n_head", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--n_embd", type=int, default=768, help="Embedding size")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")
    parser.add_argument("--bias", action="store_true", help="Use bias in LayerNorm and Linear layers")

    # adamw optimizer
    parser.add_argument("--learning_rate", type=float, default=6e-4, help="Maximum learning rate")
    parser.add_argument("--max_iters", type=int, default=600000, help="Maximum training iterations")
    parser.add_argument("--weight_decay", type=float, default=1e-1, help="Weight decay rate")
    parser.add_argument("--beta1", type=float, default=0.9, help="Beta1 for AdamW optimizer")
    parser.add_argument("--beta2", type=float, default=0.95, help="Beta2 for AdamW optimizer")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping value")

    # learning rate decay settings
    parser.add_argument("--decay_lr", action="store_true", help="Enable learning rate decay")
    parser.add_argument("--warmup_iters", type=int, default=2000, help="Warmup iterations")
    parser.add_argument("--lr_decay_iters", type=int, default=600000, help="LR decay iterations")
    parser.add_argument("--min_lr", type=float, default=6e-5, help="Minimum learning rate")

    # DDP settings
    parser.add_argument("--backend", type=str, choices=["nccl", "gloo"], default="nccl", help="DDP backend")

    # system
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float32", "bfloat16", "float16"], help="Data type")
    parser.add_argument("--compile", action="store_true", help="Enable PyTorch 2.0 model compilation")

    args = parser.parse_args()
    return args