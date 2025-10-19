import os
import requests
import zipfile
import torch
from torch import nn
from d2l import torch as d2l
from gensim.models import Word2Vec

# =========================
# 1. Download PTB dataset
# =========================
def download_ptb(data_dir='data'):
    url = 'https://d2l.ai/d2l-en-data/ptb.zip'
    fname = os.path.join(data_dir, 'ptb.zip')
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(fname):
        print(f'Downloading {url} ...')
        response = requests.get(url, stream=True)
        with open(fname, 'wb') as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
        print('Downloaded PTB dataset.')
    
    ptb_dir = os.path.join(data_dir, 'ptb')
    if not os.path.exists(ptb_dir):
        with zipfile.ZipFile(fname, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        print('PTB dataset extracted.')
    
    return ptb_dir

ptb_dir = download_ptb()

# =========================
# 2. Read training sentences
# =========================
train_file = os.path.join(ptb_dir, 'ptb.train.txt')
sentences = []
with open(train_file, 'r', encoding='utf-8') as f:
    for line in f:
        tokens = line.strip().split()
        if tokens:
            sentences.append(tokens)

print(f"Number of sentences: {len(sentences)}")

# =========================
# 3. PyTorch Skip-Gram (manual)
# =========================
batch_size, max_window_size, num_noise_words = 512, 5, 5
data_iter, vocab = d2l.load_data_ptb(batch_size, max_window_size, num_noise_words)

embed_size = 100
net = nn.Sequential(
    nn.Embedding(num_embeddings=len(vocab), embedding_dim=embed_size),
    nn.Embedding(num_embeddings=len(vocab), embedding_dim=embed_size)
)

def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    """Compute the skip-gram prediction using embeddings"""
    v = embed_v(center)
    u = embed_u(contexts_and_negatives)
    pred = torch.bmm(v, u.permute(0, 2, 1))
    return pred

class SigmoidBCELoss(nn.Module):
    """Binary cross entropy loss with mask"""
    def forward(self, inputs, target, mask=None):
        out = nn.functional.binary_cross_entropy_with_logits(
            inputs, target, weight=mask, reduction="none")
        return out.mean(dim=1)

loss = SigmoidBCELoss()

# Train function (simplified)
def train(net, data_iter, lr, num_epochs, device=d2l.try_gpu()):
    def init_weights(m):
        if isinstance(m, nn.Embedding):
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    metric = d2l.Accumulator(2)
    for epoch in range(num_epochs):
        for batch in data_iter:
            optimizer.zero_grad()
            center, context_negative, mask, label = [x.to(device) for x in batch]
            pred = skip_gram(center, context_negative, net[0], net[1])
            l = (loss(pred.reshape(label.shape).float(), label.float(), mask)
                 / mask.sum(axis=1) * mask.shape[1])
            l.sum().backward()
            optimizer.step()
            metric.add(l.sum(), l.numel())
    print(f'Final loss {metric[0]/metric[1]:.3f}')

# Train manually
lr, num_epochs = 0.002, 3
train(net, data_iter, lr, num_epochs)

# Get similar words (manual embeddings)
def get_similar_tokens(query_token, k, embed):
    W = embed.weight.data
    x = W[vocab[query_token]]
    cos = torch.mv(W, x) / torch.sqrt(torch.sum(W*W, dim=1) * torch.sum(x*x) + 1e-9)
    topk = torch.topk(cos, k=k+1)[1].cpu().numpy()
    print(f"Top {k} words similar to '{query_token}' (manual):")
    for i in topk[1:]:
        print(vocab.to_tokens(i))

get_similar_tokens('chip', 3, net[0])

# =========================
# 4. Gensim Skip-Gram (ready-made)
# =========================
model = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=1,
    sg=1,          # sg=1 means Skip-Gram
    negative=5,
    workers=4
)

query_word = 'computer'
top_k = 5
print(f"\nTop {top_k} words similar to '{query_word}' (gensim):")
for word, sim in model.wv.most_similar(query_word, topn=top_k):
    print(f"{word}: {sim:.3f}")
