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

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.nn.functional as F

from GPT_VL_mae import GPTConfig, GPT

from torch.utils.data import DataLoader, Dataset
from dataset import ClusterDataset, collate_fn
import sys

import copy
from tqdm import tqdm

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'ckpt/MAE'
eval_interval = 2000
log_interval = 100
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
resume_ckpt = ""
# data
gradient_accumulation_steps = 1 # used to simulate larger batch sizes
batch_size = 128 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1300
max_seq_length = 25
out_seq_num = None
decay_ratio = 0.
use_SVLN = True
# model
n_layer = 4
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
pe2 = 13
# adamw optimizer
learning_rate = 1e-5 # max learning rate
max_iters = 200000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
target_update_tau = 0.005
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 1000 # how many steps to warm up for
lr_decay_iters = max_iters # should be ~= max_iters per Chinchilla
min_lr = learning_rate/10 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
# dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
dtype = 'float32'
# compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('config.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------
print(time.time(), config)

master_process = True
seed_offset = 0
ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, )
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.cuda.amp.autocast(dtype=ptdtype)



train_dataset = ClusterDataset(max_seq_length, out_seq_num, decay_ratio, 'train', use_SVLN=use_SVLN)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, collate_fn=collate_fn)
val1_dataset = ClusterDataset(max_seq_length, out_seq_num, decay_ratio, 'train1', use_SVLN=use_SVLN)
val1_dataloader = DataLoader(val1_dataset, batch_size=batch_size*2, shuffle=True, num_workers=8, collate_fn=collate_fn)
val2_dataset = ClusterDataset(max_seq_length, out_seq_num, decay_ratio, 'unseen', use_SVLN=use_SVLN)
val2_dataloader = DataLoader(val2_dataset, batch_size=batch_size*2, shuffle=True, num_workers=8, collate_fn=collate_fn)
val_dataloaders = [val1_dataloader, val2_dataloader]



# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, dropout=dropout, pe2=pe2) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {resume_ckpt}")
    # resume training from a checkpoint.
    ckpt_path = resume_ckpt # os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'pe2', ]:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model = model.cuda()
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss'].cuda()
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', ]:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    assert False
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memor

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    
    for iv, val_loader in enumerate(val_dataloaders):
        num = 0
        gt_loss = 0
        n_gt = 0
        losses = torch.zeros(len(val_loader))
        n_end = 0
        for k, (in_vI, in_vT, in_a, in_cand, num_cand, gt) in enumerate(tqdm(val_loader)):
            with ctx:
                loss2gt = model(in_vI, in_vT, in_a, )
                gt_loss += loss2gt
                n_gt += torch.norm(gt, dim=1).mean()
                num += 1
                
            losses[k] = loss2gt.item()
        if iv == 0:
            out["train"] = losses.mean()
            print("TRAIN CENT LOSS:", gt_loss/num, n_gt/num, "end ratio:", n_end/num)
        else:
            assert iv == 1
            out["val"] = losses.mean()
            print("VAL CENT LOSS:", gt_loss/num, n_gt/num, "end ratio:", n_end/num)
    model.train()
    
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# training loop
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model # unwrap DDP container if needed
running_mfu = -1.0
data_iter = iter(train_dataloader)
while True:
    try:
        # Get the next batch of data
        in_vI, in_vT, in_a, in_cand, num_cand, gt = next(data_iter)
    except StopIteration:
        # Reset the iterator when it reaches the end of the dataset
        data_iter = iter(train_dataloader)
    
    
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if init_from == "resume":
            best_val_loss = best_val_loss.to(losses['val'].device)
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
                torch.save(checkpoint, os.path.join(out_dir, f'ckpt{iter_num}.pt'))
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            loss2gt = model(in_vI, in_vT, in_a, )
            loss = loss2gt / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        
        try:
            # Get the next batch of data
            in_vI, in_vT, in_a, in_cand, num_cand, gt = next(data_iter)
        except StopIteration:
            # Reset the iterator when it reaches the end of the dataset
            data_iter = iter(train_dataloader)
        
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    
    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms") #, mfu {running_mfu*100:.2f}%
    iter_num += 1
    local_iter_num += 1
    sys.stdout.flush()
    # termination conditions
    if iter_num > max_iters:
        break
