"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

import time

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # # causal mask to ensure that attention is only applied to the left in the input sequence
            # self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
            #                             .view(1, 1, config.block_size, config.block_size))

    def forward(self, x, padding_mask=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            assert False
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            
            if padding_mask is not None:
                padding_mask = padding_mask.unsqueeze(1)  # (B, 1, T, T)
                assert len(padding_mask.shape) == 4 and padding_mask.shape[1] == 1 and padding_mask.shape[2] == padding_mask.shape[3]
                att = att.masked_fill(padding_mask == 1, float('-inf'))
            
            
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, padding_mask=None):
        x = x + self.attn(self.ln_1(x), padding_mask=padding_mask)
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    pe2: int = 0

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.block_size is not None
        self.config = config

        self.pe2 = config.pe2
        self.transformer = nn.ModuleDict(dict(
            wte_viewI = nn.Linear(512, config.n_embd),
            wte_viewT = nn.Linear(512, config.n_embd),
            wte_act = nn.Linear(4, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            wpe2 = nn.Embedding(config.pe2, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, 512, bias=False)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
            n_params -= self.transformer.wpe2.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, viewsI, viewsT, acts, cand_acts, targets=None):
        # TODO: debug padding
        device = "cuda"
        view_lens = [x.shape[0] for x in viewsI]
        B = len(view_lens)
        num_cands = [x.shape[0] for x in cand_acts]
        
        L_max = max([x*13-1+y for (x, y) in zip(view_lens, num_cands)])
        assert max(view_lens)*13 <= self.config.block_size, f"Cannot forward sequence of length {max(view_lens)*13}, block size is only {self.config.block_size}"
        pos = torch.arange(0, max(view_lens)*13, dtype=torch.long, device=device) # shape (t)
        
        padding_mask = torch.zeros(
            B, L_max, L_max, device=device).bool()
        pred_idx = []
        for b, (_view_len, _num_cand) in enumerate(zip(view_lens, num_cands)):
            padding_mask[b, :, _view_len*13+_num_cand-1:] = True
            
            pred_idx.append(torch.arange(_view_len*13-1, _view_len*13-1+_num_cand))
        
        padded_viewsI = pad_tensors(viewsI).cuda()
        assert list(padded_viewsI.shape) == [B, max(view_lens), 12, 512], (padded_viewsI.shape, B, max(view_lens))
        padded_viewsT = pad_tensors(viewsT).cuda()
        assert list(padded_viewsT.shape) == [B, max(view_lens), 12, 512], (padded_viewsT.shape, B, max(view_lens))
        padded_acts = pad_tensors(acts).cuda()
        assert list(padded_acts.shape) == [B, max(view_lens), 4]
        padded_cands = pad_tensors(cand_acts).cuda()
        assert list(padded_cands.shape) == [B, max(num_cands), 4]

        view_tok_emb = self.transformer.wte_viewI(padded_viewsI) + self.transformer.wte_viewT(padded_viewsT)
        # B x max(N_step) x 12 x 768
        act_tok_emb = self.transformer.wte_act(padded_acts).unsqueeze(2) # B x max(N_step) x 1 x 768
        cand_acts_tok_emb = self.transformer.wte_act(padded_cands) # B x max(N_cand) x 768
        
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        pos_emb2 = self.transformer.wpe2(torch.arange(0, self.pe2, dtype=torch.long, device=device))
        
        pad_all_tokens = []
        for b, (_view_len, _num_cand) in enumerate(zip(view_lens, num_cands)):
            view_ft = view_tok_emb[b][:_view_len]
            act_ft = act_tok_emb[b][:_view_len]
            view_and_act = torch.cat([view_ft, act_ft], dim=1)
            assert list(view_and_act.shape) == [_view_len, 13, self.config.n_embd] and list(pos_emb2.shape) == [13, self.config.n_embd]
            view_and_act += pos_emb2
            view_and_act = view_and_act.reshape(_view_len*13, self.config.n_embd)
            assert list(pos_emb[:_view_len*13].shape) == [_view_len*13, self.config.n_embd]
            view_and_act += pos_emb[:_view_len*13]
            cand_ft = cand_acts_tok_emb[b][:_num_cand] + pos_emb2[12] + pos_emb[_view_len*13 - 1]
            view_and_act_and_cand = torch.cat([view_and_act[:-1], cand_ft], dim=0)
            pad_all_tokens.append(view_and_act_and_cand)
        pad_all_tokens = pad_tensors_wgrad(pad_all_tokens).cuda()
        assert list(pad_all_tokens.shape) == [B, L_max, self.config.n_embd], (pad_all_tokens.shape, B, L_max, max(num_cands))
        
        
        x = self.transformer.drop(pad_all_tokens)
        for block in self.transformer.h:
            x = block(x, padding_mask)
        x = self.transformer.ln_f(x)

        logits = self.lm_head(x)
        pred_ft = torch.cat([logits[x, y] for x, y in enumerate(pred_idx)], dim=0)
        if targets is not None:
            loss = F.mse_loss(pred_ft, targets, reduction='none').sum(dim=-1).mean()
        else:
            loss = None

        return pred_ft, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer
 
def pad_tensors(tensors, lens=None, pad=0):
    """B x [T, ...] torch tensors"""
    if lens is None:
        lens = [t.size(0) for t in tensors]
    max_len = max(lens)
    bs = len(tensors)
    hid = list(tensors[0].size()[1:])
    size = [bs, max_len] + hid

    dtype = tensors[0].dtype
    output = torch.zeros(*size, dtype=dtype)
    if pad:
        output.data.fill_(pad)
    for i, (t, l) in enumerate(zip(tensors, lens)):
        output.data[i, :l, ...] = t.data
    return output  

def pad_tensors_wgrad(tensors, lens=None):
    """B x [T, ...] torch tensors"""
    if lens is None:
        lens = [t.size(0) for t in tensors]
    max_len = max(lens)
    batch_size = len(tensors)
    hid = list(tensors[0].size()[1:])

    device = tensors[0].device
    dtype = tensors[0].dtype

    output = []
    for i in range(batch_size):
        if lens[i] < max_len:
            tmp = torch.cat(
                [tensors[i], torch.zeros([max_len-lens[i]]+hid, dtype=dtype).to(device)],
                dim=0
            )
        else:
            tmp = tensors[i]
        output.append(tmp)
    output = torch.stack(output, 0)
    return output


 