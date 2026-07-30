import torch

# Data
Data_path = "data/corpus.txt"
Train_split = 0.9

# Model
Block_size = 256
N_emb = 384
N_head = 6
N_layer = 6
Dropout = 0.2


assert N_emb % N_head == 0, (
    f"n_emb ({N_emb}) must be divisible by n_head ({N_head}) ")

# Training
Batch_size = 64
Max_iters = 5000
Eval_interval = 500
Eval_iters = 200
Learning_rate = 3e-4
Grad_clip = 1.0

#device
Device = "cuda" if torch.cuda.is_available() else "cpu"