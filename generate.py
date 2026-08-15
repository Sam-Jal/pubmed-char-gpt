import torch
from config import Block_size, Device
from model import GPT

def load_model(checkpoint_path="checkpoint.pt"):
    ckpt = torch.load(checkpoint_path, map_location=Device)
    # Older checkpoints did not record model_config, so retain compatibility
    # with the default architecture in config.py.
    model = GPT(ckpt["vocab_size"], **ckpt.get("model_config", {})).to(Device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["stoi"], ckpt["itos"]

def generate(prompt="", max_new_tokens=500, temperature=0.8):
    model, stoi, itos = load_model()

    encode = lambda s: [stoi[c] for c in s if c in stoi]
    decode = lambda values: "".join([itos[i] for i in values])

    prompt_tokens = encode(prompt)
    if not prompt_tokens:
        prompt_tokens = [stoi.get("\n", 0)]
    context = torch.tensor(
        prompt_tokens,
        dtype=torch.long,
        device=Device,
    ).unsqueeze(0)

    output = model.generate(context, max_new_tokens=max_new_tokens, temperature=temperature)
    print(decode(output[0].tolist()))

if __name__ == "__main__":
    generate(prompt="The patient presented with", max_new_tokens=500, temperature=0.8)
