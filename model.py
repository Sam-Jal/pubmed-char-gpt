import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Block_size, N_emb, N_head, N_layer, Dropout, Device


class Head(nn.Module):
    """Single causal: self:attention head"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(N_emb, head_size, bias=False)
        self.query = nn.Linear(N_emb, head_size, bias=False)
        self.value = nn.Linear(N_emb, head_size, bias=False)
        
        self.register_buffer("tril", torch.tril(torch.ones(Block_size, Block_size)))
        self.dropout = nn.Dropout(Dropout)


    def forward(self, x):

        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        dim = k.shape[-1] 

        attn = q @ k.transpose(-2, -1) / (dim ** 0.5)

        attn.attn.masked_fill(
            self.tril[:T, :T] == 0, float("-inf"),
        )

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v
        return out
    
class MultiHeadAttention(nn.Module):
    """Multiple heads in parallel, then project back to N_emb"""


    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(N_emb, N_emb) #projection after concatination
        self.dropout = nn.Dropout(Dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """"position-wise FFN: Linear-ReLu-Linear, with 4x inner dimension"""


    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_emb, 4 * N_emb),
            nn.ReLU(),
            nn.Linear(4 * N_emb, N_emb),
            nn.Dropout(Dropout),
        )

    def forward(self, x):
        return self.net(x)
    



    
class Block(nn.Module):
    """
    One transformer decoder block:
        x +=  MultiheadAttention(LayerNorm(x))  : communication step
        x += FeedForward(LayerNorm(x)) : computation step

    Nore: LayerNorm is applied BEFORE each sublayer (Pre-LN)
    The original paper used Post-LN but pre-LN trains more stably (why?)
    """

    def __init__(self):
        super().__init__()
        head_size = N_emb // N_head
        self.attn = MultiHeadAttention(N_head, head_size)
        self.ff = FeedForward()
        self.ln1 = nn.LayerNorm(N_emb)
        self.ln2 = nn.LayerNorm(N_emb)

    def forward(self, x):
        x += self.attn(self.ln1(x))
        x += self.ff(self.ln2(x))
        return x
    



class GPT(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, N_emb)
        self.pos_emb = nn.Embedding(Block_size,  N_emb)
        self.blocks = nn.Sequential(*[Block() for _ in range(N_layer)])
        self.ln_final = nn.LayerNorm(N_emb)
        self.head = nn.Linear(N_emb, vocab_size, bias=False)


    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.token_emb(idx)          # (B, T, N_emb)
        pos = self.pos_emb(torch.arange(T, device=Device)) # (T, N_emb)
        x = tok + pos
        x = self.blocks(x)
        x = self.ln_final(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))

        return logits, loss            


    @torch.no_grad()
    def generate(self, idx, max_new_token, temperature=1.0):
        """Autoregressive generation: idx is (B,T) of current context"""

        for _ in range(max_new_token):
            idx_cond = idx[:, -Block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1) #(B, 1)
            idx = torch.cat([idx, next_tok], dim=1)

        return idx
    
    