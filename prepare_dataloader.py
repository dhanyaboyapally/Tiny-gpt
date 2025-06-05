from datasets import load_dataset
from transformers import GPT2Tokenizer
import torch
from torch.utils.data import DataLoader

# Load Shakespeare dataset
dataset = load_dataset("tiny_shakespeare", split="train", trust_remote_code=True)

# Load GPT2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Concatenate all text into one string
all_text = " ".join(dataset['text'])

# Tokenize the whole text
tokens = tokenizer(all_text, return_tensors='pt')['input_ids'][0]

# Define sequence length
seq_len = 128

# Chunk the token stream
chunks = [tokens[i:i + seq_len] for i in range(0, len(tokens) - seq_len, seq_len)]

print("✅ Total chunks created:", len(chunks))

# PyTorch Dataset class
class GPTDataset(torch.utils.data.Dataset):
    def __init__(self, chunks):
        self.chunks = chunks

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        input_ids = self.chunks[idx]
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

# Create Dataset and DataLoader
gpt_dataset = GPTDataset(chunks)
data_loader = DataLoader(gpt_dataset, batch_size=12, shuffle=True)

# Show a full batch
for batch in data_loader:
    print("🎯 Input IDs shape:", batch['input_ids'].shape)
    print("🎯 Labels shape:", batch['labels'].shape)
    break
