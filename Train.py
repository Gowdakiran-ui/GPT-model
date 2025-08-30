import torch
from torch.utils.data import DataLoader
from model import GPT, GPTConfig
from data import CharDataset
from tqdm import tqdm

# Load dataset
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

dataset = CharDataset(text, block_size=128)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Model
config = GPTConfig(vocab_size=dataset.vocab_size)
model = GPT(config)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# Training loop
for epoch in range(5):
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    for xb, yb in pbar:
        logits, loss = model(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.set_postfix(loss=loss.item())

# Generate text
context = torch.zeros((1, 1), dtype=torch.long)
print("Generated:", dataset.itos[i] for i in model.generate(context, max_new_tokens=100)[0].tolist())
