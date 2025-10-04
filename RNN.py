import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------------- Text Dataset ----------------
class TextDataset(Dataset):
    """Character-level text dataset"""
    def __init__(self, text, seq_length):
        self.seq_length = seq_length
        self.chars = sorted(list(set(text)))
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.data = [self.char_to_idx[ch] for ch in text]

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_length]
        y = self.data[idx+1:idx+self.seq_length+1]
        return torch.tensor(x), torch.tensor(y)

# ---------------- Model ----------------
class SimpleRNN(nn.Module):
    """Simple RNN for character-level text generation"""
    def __init__(self, vocab_size, hidden_size=256, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embed = nn.Embedding(vocab_size, vocab_size)  # replaces one-hot
        self.rnn = nn.RNN(vocab_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embed(x)
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size, device):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)

# ---------------- Training ----------------
def train(model, dataloader, epochs=50, lr=0.001, device='cpu'):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs+1):
        total_loss = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            hidden = model.init_hidden(x.size(0), device)

            output, _ = model(x, hidden)
            loss = criterion(output.transpose(1,2), y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

# ---------------- Text Generation ----------------
def generate_text(model, start_text, char_to_idx, idx_to_char, length=100, device='cpu'):
    model.eval()
    model.to(device)
    input_seq = [char_to_idx.get(ch, 0) for ch in start_text.lower()]
    generated = start_text
    hidden = model.init_hidden(1, device)

    with torch.no_grad():
        # Feed start text
        for idx in input_seq[:-1]:
            x = torch.tensor([[idx]], device=device)
            _, hidden = model(x, hidden)

        x = torch.tensor([[input_seq[-1]]], device=device)
        for _ in range(length):
            out, hidden = model(x, hidden)
            probs = torch.softmax(out[0, -1], dim=0)
            idx = torch.multinomial(probs, 1).item()
            generated += idx_to_char[idx]
            x = torch.tensor([[idx]], device=device)

    return generated

# ---------------- Main ----------------
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load data
    try:
        with open('timemachine.txt', 'r', encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        text = ("The time traveller ... flushed and animated " * 100).lower()

    seq_length = 35
    batch_size = 32
    dataset = TextDataset(text, seq_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    vocab_size = len(dataset.chars)
    model = SimpleRNN(vocab_size, hidden_size=256)

    # Train
    train(model, dataloader, epochs=50, lr=0.001, device=device)

    # Generate
    print(generate_text(model, "the time", dataset.char_to_idx, dataset.idx_to_char, length=200, device=device))
