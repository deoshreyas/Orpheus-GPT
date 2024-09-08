import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.onnx

# Hyperparameters
batch_size = 64
chunk_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

# torch.manual_seed(1234) # for reproducibility

# Dataset
# Courtesy of the Internet Classics Archive (shortened version, full text available online at http://classics.mit.edu//Homer/iliad.html)
with open("input.txt", "r") as f:
    text = f.read()

# Preparing data for tokenization
chars = sorted(list(set(text)))
vocabulary_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i, ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder
decode = lambda l: ''.join([itos[i] for i in l]) # decoder

# Split data into training and validation sets
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% is training dataset
training_data = data[:n]
validation_data = data[n:]

# Get a random batch for training and testing
def get_batch(split):
    data = training_data if split=='train' else validation_data
    ix = torch.randint(len(data)-chunk_size, (batch_size,))
    x = torch.stack([data[i:i+chunk_size] for i in ix])
    y = torch.stack([data[i+1:i+chunk_size+1] for i in ix])
    x,y = x.to(device), y.to(device)
    return x,y

# Loss estimation function
@torch.no_grad
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Attention is all you need
class Head(nn.Module):
    """single-head self-attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(chunk_size, chunk_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        # compute affinities
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T]==0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        # weighted aggregation
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    """multiple-heads of self-attention running in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.projection = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.projection(out))
        return out

# Feed forward layer for transformer
class FeedForward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

# Block class (Each block has 8 chars)
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.lnm1 = nn.LayerNorm(n_embd)
        self.lnm2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.lnm1(x))
        x = x + self.ffwd(self.lnm2(x))
        return x

# Bigram Language Model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocabulary_size, n_embd)
        self.position_embedding_table = nn.Embedding(chunk_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.lmnf = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocabulary_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.lmnf(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets) # loss - how well are we predicting the next character

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # getting predictions
            idx_condition = idx[:, -chunk_size:]
            logits, loss = self(idx_condition)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# Define encoding and decoding functions
chars = ['\n', ' ', '!', '"', '&', "'", '(', ')', ',', '-', '.', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return ''.join([itos[i] for i in l])

# Prepare input
prompt = "My name is Shreyas, and I am the greatest"
encoded_prompt = encode(prompt)

# Load the entire model
model = torch.load('model.pth', map_location=torch.device('cpu'))
model.eval()

# Generate text
tokens_to_generate = 100
context = torch.tensor((encoded_prompt, encoded_prompt), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=tokens_to_generate)[0].tolist()))