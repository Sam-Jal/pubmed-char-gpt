import torch
from config import *
from model import GPT

# ------------------------------------------------------------------
# Load corpus and build vocabulary
# ------------------------------------------------------------------
with open(Data_path, "r", encoding="utf-8") as f:
    text = f.read()

chars     = sorted(set(text))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}   # char → int
itos = {i: ch for i, ch in enumerate(chars)}   # int → char

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

# ------------------------------------------------------------------
# Train / val split
# ------------------------------------------------------------------
data  = torch.tensor(encode(text), dtype=torch.long)
n     = int(Train_split * len(data))
train_data = data[:n]
val_data   = data[n:]

# ------------------------------------------------------------------
# Batch sampling
# ------------------------------------------------------------------
def get_batch(split):
    data = train_data if split == "train" else val_data
    # Sample BATCH_SIZE random starting positions
    ix = torch.randint(len(data) - Block_size, (Batch_size,))
    x  = torch.stack([data[i     : i + Block_size    ] for i in ix])
    y  = torch.stack([data[i + 1 : i + Block_size + 1] for i in ix])
    return x.to(Device), y.to(Device)

# ------------------------------------------------------------------
# Loss estimation (averaged over EVAL_ITERS batches)
# ------------------------------------------------------------------
@torch.no_grad()
def estimate_loss(model):
    model.eval()
    losses = {}
    for split in ("train", "val"):
        L = torch.zeros(Eval_iters)
        for k in range(Eval_iters):
            x, y  = get_batch(split)
            _, loss = model(x, y)
            L[k]  = loss.item()
        losses[split] = L.mean().item()
    model.train()
    return losses

# ------------------------------------------------------------------
# Build model and optimizer
# ------------------------------------------------------------------
model = GPT(vocab_size).to(Device)
print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
print(f"Vocabulary: {vocab_size} characters")
print(f"Device: {Device}")

optimizer = torch.optim.AdamW(model.parameters(), lr=Learning_rate)

# ------------------------------------------------------------------
# Training loop
# ------------------------------------------------------------------
for step in range(Max_iters):

    if step % Eval_interval == 0:
        losses = estimate_loss(model)
        print(f"step {step:5d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")

    x, y = get_batch("train")
    logits, loss = model(x, y)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), Grad_clip)
    optimizer.step()

# Save checkpoint
torch.save({
    "model_state": model.state_dict(),
    "stoi"       : stoi,
    "itos"       : itos,
    "vocab_size" : vocab_size,
}, "checkpoint.pt")
print("Checkpoint saved.")