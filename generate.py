import torch
from config import Block_size, Device
from model import GPT

def load_model(checkpoint_path="checkpoint.pt"):
    ckpt = torch.load(checkpoint_path, map_location=Device)
    model = GPT(ckpt["vocab_size"]).to(Device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["stoi"], ckpt["itos"]

def generate(prompt="", max_new_tokens=500, temperature=0.8):
    model, stoi, itos = load_model()

    encode = lambda s: [stoi[c] for c in s if c in stoi]
    decode = lambda l: "".join([itos[i] for i in l])

    if prompt:
        context = torch.tensor(encode(prompt), dtype=torch.long, device=Device).unsqueeze(0)
    else:
        # Start from newline token — cold start
        context = torch.zeros((1, 1), dtype=torch.long, device=Device)

    output = model.generate(context, max_new_tokens=max_new_tokens, temperature=temperature)
    print(decode(output[0].tolist()))

if __name__ == "__main__":
    generate(prompt="The patient presented with", max_new_tokens=500, temperature=0.8)