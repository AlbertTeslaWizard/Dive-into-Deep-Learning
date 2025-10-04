import torch
import torch.nn as nn
import torch.nn.functional as F
import math, time, os, re, random, requests, collections

# =========================
# 1. Dataset & Vocabulary
# =========================

def download_text(url):
    """Download and return the dataset as a string."""
    filename = url.split('/')[-1]
    if not os.path.exists(filename):
        print(f"Downloading {filename} ...")
        text = requests.get(url).text
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(text)
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read()

class Vocab:
    """Vocabulary for mapping between tokens and indices."""
    def __init__(self, tokens, min_freq=0):
        if isinstance(tokens[0], list):
            tokens = [t for line in tokens for t in line]
        counter = collections.Counter(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        self.idx_to_token = ['<unk>'] + [t for t, f in self.token_freqs if f >= min_freq]
        self.token_to_idx = {t: i for i, t in enumerate(self.idx_to_token)}

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, 0)
        return [self.__getitem__(t) for t in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[i] for i in indices]


def data_iterator_seq(corpus, batch_size, num_steps, device):
    """Sequential mini-batch iterator."""
    offset = random.randint(0, num_steps - 1)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    if num_tokens <= 0:
        raise ValueError(f"Corpus too small for batch_size={batch_size}, num_steps={num_steps}")

    X = torch.tensor(corpus[offset: offset + num_tokens], device=device)
    Y = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens], device=device)

    X, Y = X.reshape(batch_size, -1), Y.reshape(batch_size, -1)
    num_batches = max(1, X.shape[1] // num_steps)

    for i in range(0, num_batches * num_steps, num_steps):
        yield X[:, i:i + num_steps], Y[:, i:i + num_steps]


# =========================
# 2. RNN Model (from scratch)
# =========================

class RNNModelScratch(nn.Module):
    """Simple RNN implemented from scratch."""
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.W_xh = nn.Parameter(torch.randn(vocab_size, hidden_size) * 0.01)
        self.W_hh = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.b_h = nn.Parameter(torch.zeros(hidden_size))
        self.W_hq = nn.Parameter(torch.randn(hidden_size, vocab_size) * 0.01)
        self.b_q = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, X, state):
        # X: (batch_size, num_steps)
        X = F.one_hot(X.T, self.vocab_size).float()  # (num_steps, batch_size, vocab_size)
        H, = state
        outputs = []
        for x in X:
            H = torch.tanh(x @ self.W_xh + H @ self.W_hh + self.b_h)
            outputs.append(H @ self.W_hq + self.b_q)
        return torch.cat(outputs, dim=0), (H,)

    def init_state(self, batch_size, device):
        return (torch.zeros((batch_size, self.hidden_size), device=device),)


# =========================
# 3. Text Generation
# =========================

def generate_text(prefix, num_preds, model, vocab, device):
    """Generate text sequence given a prefix."""
    state = model.init_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))

    # Warm up with prefix
    for ch in prefix[1:]:
        _, state = model(get_input(), state)
        outputs.append(vocab[ch])

    # Predict next tokens
    for _ in range(num_preds):
        y, state = model(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))

    return ''.join(vocab.to_tokens(outputs))


# =========================
# 4. Training Loop
# =========================

def train(model, corpus, vocab, lr, num_epochs, device, batch_size, num_steps):
    """Train the RNN model."""
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    model.to(device)

    for epoch in range(1, num_epochs + 1):
        train_iter = data_iterator_seq(corpus, batch_size, num_steps, device)
        state = None
        total_loss, total_tokens = 0, 0
        start = time.time()

        for X, Y in train_iter:
            state = model.init_state(X.shape[0], device) if state is None else tuple(s.detach() for s in state)
            y_hat, state = model(X, state)
            l = loss_fn(y_hat, Y.T.reshape(-1).long())

            optimizer.zero_grad()
            l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += l.item() * Y.numel()
            total_tokens += Y.numel()

        if epoch % 10 == 0:
            ppl = math.exp(total_loss / total_tokens)
            speed = total_tokens / (time.time() - start)
            print(f"Epoch {epoch}/{num_epochs} | Perplexity: {ppl:.1f} | Speed: {speed:.1f} tokens/s")
            print(generate_text("time traveller", 50, model, vocab, device))


# =========================
# 5. Main Execution
# =========================

if __name__ == "__main__":
    BATCH_SIZE = 16
    NUM_STEPS = 20
    HIDDEN_SIZE = 512
    NUM_EPOCHS = 100
    LR = 1.0
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    url = "http://d2l-data.s3-accelerate.amazonaws.com/timemachine.txt"
    text = download_text(url)
    cleaned = [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in text.split('\n')]
    tokens = [list(line) for line in cleaned if line]

    vocab = Vocab(tokens)
    corpus = vocab[[t for line in tokens for t in line]]

    min_tokens = BATCH_SIZE * NUM_STEPS + 1
    if len(corpus) < min_tokens:
        raise ValueError(f"Corpus too small ({len(corpus)} tokens), need ≥ {min_tokens}")

    model = RNNModelScratch(len(vocab), HIDDEN_SIZE)
    print(f"Training on {DEVICE} ...")
    train(model, corpus, vocab, LR, NUM_EPOCHS, DEVICE, BATCH_SIZE, NUM_STEPS)
