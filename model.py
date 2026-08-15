import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Block_size, Dropout, N_emb, N_head, N_layer


class Head(nn.Module):
    """One causal self-attention head."""

    def __init__(self, n_emb, head_size, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_emb, head_size, bias=False)
        self.query = nn.Linear(n_emb, head_size, bias=False)
        self.value = nn.Linear(n_emb, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        _, time_steps, _ = x.shape
        key = self.key(x)
        query = self.query(x)
        value = self.value(x)

        attention = query @ key.transpose(-2, -1) / (key.shape[-1] ** 0.5)
        attention = attention.masked_fill(
            self.tril[:time_steps, :time_steps] == 0,
            float("-inf"),
        )
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)
        return attention @ value


class MultiHeadAttention(nn.Module):
    """Run several attention heads in parallel and project their outputs."""

    def __init__(self, n_emb, num_heads, block_size, dropout):
        super().__init__()
        head_size = n_emb // num_heads
        self.heads = nn.ModuleList(
            [Head(n_emb, head_size, block_size, dropout) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(n_emb, n_emb)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        combined = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.dropout(self.proj(combined))


class FeedForward(nn.Module):
    """Position-wise feed-forward network with a 4x hidden dimension."""

    def __init__(self, n_emb, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_emb, 4 * n_emb),
            nn.ReLU(),
            nn.Linear(4 * n_emb, n_emb),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """A pre-LayerNorm transformer decoder block."""

    def __init__(self, n_emb, n_head, block_size, dropout):
        super().__init__()
        self.attn = MultiHeadAttention(n_emb, n_head, block_size, dropout)
        self.ff = FeedForward(n_emb, dropout)
        self.ln1 = nn.LayerNorm(n_emb)
        self.ln2 = nn.LayerNorm(n_emb)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        block_size=Block_size,
        n_emb=N_emb,
        n_head=N_head,
        n_layer=N_layer,
        dropout=Dropout,
    ):
        super().__init__()
        if n_emb % n_head != 0:
            raise ValueError("n_emb must be divisible by n_head")

        self.block_size = block_size
        self.model_config = {
            "block_size": block_size,
            "n_emb": n_emb,
            "n_head": n_head,
            "n_layer": n_layer,
            "dropout": dropout,
        }
        self.token_emb = nn.Embedding(vocab_size, n_emb)
        self.pos_emb = nn.Embedding(block_size, n_emb)
        self.blocks = nn.Sequential(
            *[
                Block(n_emb, n_head, block_size, dropout)
                for _ in range(n_layer)
            ]
        )
        self.ln_final = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        batch_size, time_steps = idx.shape
        if time_steps == 0:
            raise ValueError("idx must contain at least one token")
        if time_steps > self.block_size:
            raise ValueError(
                f"sequence length {time_steps} exceeds block size {self.block_size}"
            )

        token_embeddings = self.token_emb(idx)
        positions = torch.arange(time_steps, device=idx.device)
        position_embeddings = self.pos_emb(positions)
        x = token_embeddings + position_embeddings
        x = self.blocks(x)
        logits = self.head(self.ln_final(x))

        loss = None
        if targets is not None:
            _, _, vocab_size = logits.shape
            loss = F.cross_entropy(
                logits.reshape(batch_size * time_steps, vocab_size),
                targets.reshape(batch_size * time_steps),
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        """Generate tokens autoregressively from a non-empty context."""
        if temperature <= 0:
            raise ValueError("temperature must be greater than zero")
        if idx.shape[1] == 0:
            raise ValueError("generation context must contain at least one token")

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probabilities = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
        return idx
