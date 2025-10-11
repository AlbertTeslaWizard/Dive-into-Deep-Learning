import torch
from torch import nn
import torch.nn.functional as F
import math
import collections
import time
from torch.utils.data import DataLoader, TensorDataset
from torchtext.data.metrics import bleu_score
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# =============================================================================
# 0. Configuration and Device Setup
# =============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# 1. Data Preparation (Self-Contained)
# =============================================================================

def tokenize(lines, token='word'):
    """Splits text lines into word or character tokens."""
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        raise ValueError('ERROR: unknown token type: ' + token)

class Vocab:
    """Vocabulary for text processing."""
    def __init__(self, tokens=[], min_freq=0, reserved_tokens=[]):
        # Flatten a 2D list of tokens into a 1D list and count token frequencies
        counter = collections.Counter([token for line in tokens for token in line])
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        # The list of unique tokens
        self.idx_to_token = list(sorted(set(['<unk>'] + reserved_tokens + [
            token for token, freq in self.token_freqs if freq >= min_freq])))
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):  # Index for the unknown token
        return self.token_to_idx['<unk>']

def build_data(batch_size, num_steps):
    """Builds vocabulary and data iterator from a small sample dataset."""
    # Sample data
    raw_text = """go .	va !
    i lost .	j'ai perdu .
    he's calm .	il est calme .
    i'm home .	je suis chez moi ."""
    
    data = [line.split('\t') for line in raw_text.split('\n')]
    src, tgt = zip(*data)
    
    # Tokenization
    src_tokens = tokenize(src, 'word')
    tgt_tokens = tokenize(tgt, 'word')
    
    # Build vocabularies
    reserved_tokens = ['<pad>', '<bos>', '<eos>']
    src_vocab = Vocab(src_tokens, min_freq=0, reserved_tokens=reserved_tokens)
    tgt_vocab = Vocab(tgt_tokens, min_freq=0, reserved_tokens=reserved_tokens)
    
    # Convert tokens to indices and add <eos>
    src_indices = [src_vocab[line] + [src_vocab['<eos>']] for line in src_tokens]
    tgt_indices = [tgt_vocab[line] + [tgt_vocab['<eos>']] for line in tgt_tokens]
    
    # Pad sequences to num_steps
    def pad_sequences(indices, pad_value):
        # Truncate if longer than num_steps
        return torch.tensor([line[:num_steps] + [pad_value] * (num_steps - len(line)) for line in indices])
    
    src_tensor = pad_sequences(src_indices, src_vocab['<pad>'])
    tgt_tensor = pad_sequences(tgt_indices, tgt_vocab['<pad>'])
    
    # Create DataLoader
    dataset = TensorDataset(src_tensor, tgt_tensor)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return data_loader, src_vocab, tgt_vocab


# =============================================================================
# 2. Model Components (Encoder, Decoder, Attention)
# =============================================================================

def masked_softmax(X, valid_lens):
    """Performs softmax on the last dimension by masking out padding elements."""
    if valid_lens is None:
        return F.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        
        mask = torch.arange(shape[-1], device=X.device)[None, :] >= valid_lens[:, None]
        X_masked = X.clone()
        X_masked[mask] = -1e9
        return F.softmax(X_masked.reshape(shape), dim=-1)

class AdditiveAttention(nn.Module):
    """Additive Attention."""
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.attention_weights = None

    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys)
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)

class Seq2SeqEncoder(nn.Module):
    """A standard sequence-to-sequence encoder using GRU."""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers, dropout=dropout)

    def forward(self, X, *args):
        embedded = self.embedding(X).permute(1, 0, 2)
        outputs, hidden_state = self.rnn(embedded)
        return outputs, hidden_state

class Seq2SeqAttentionDecoder(nn.Module):
    """The decoder for a sequence-to-sequence model with attention."""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs):
        super(Seq2SeqAttentionDecoder, self).__init__(**kwargs)
        self.attention = AdditiveAttention(num_hiddens, num_hiddens, num_hiddens, dropout)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers, dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)
        self._attention_weights_storage = []

    def init_state(self, enc_outputs_and_state):
        enc_outputs, enc_hidden_state = enc_outputs_and_state
        return (enc_outputs, enc_hidden_state)

    # ############################ START OF THE FIX ############################
    def forward(self, X, state):
        # Unpack the state, using a name for the original encoder outputs
        # to emphasize that it should not be modified.
        original_enc_outputs, hidden_state = state
        
        # Permute the encoder outputs for attention calculation, but store it
        # in a new variable, leaving the original untouched.
        enc_outputs_for_attention = original_enc_outputs.permute(1, 0, 2)
        
        embedded = self.embedding(X).permute(1, 0, 2)
        
        outputs, self._attention_weights_storage = [], []
        for x in embedded:
            query = torch.unsqueeze(hidden_state[-1], dim=1)
            
            # Use the permuted version for attention
            context = self.attention(query, enc_outputs_for_attention, 
                                     enc_outputs_for_attention, None)
            
            x = torch.unsqueeze(x, dim=1)
            x_context_concat = torch.cat((context, x), dim=-1)
            out, hidden_state = self.rnn(x_context_concat.permute(1, 0, 2), hidden_state)
            outputs.append(out)
            self._attention_weights_storage.append(self.attention.attention_weights)
            
        outputs = self.dense(torch.cat(outputs, dim=0))
        
        # CRITICAL: Return the *original* encoder outputs along with the
        # *updated* hidden state.
        return outputs.permute(1, 0, 2), [original_enc_outputs, hidden_state]
    # ############################# END OF THE FIX #############################
    
    @property
    def attention_weights(self):
        return self._attention_weights_storage

class EncoderDecoder(nn.Module):
    """A wrapper for the encoder-decoder architecture."""
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_outputs_and_state = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs_and_state)
        return self.decoder(dec_X, dec_state)

# =============================================================================
# 3. Training Loop
# =============================================================================

class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """The softmax cross-entropy loss with masks."""
    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label)
        mask = torch.arange(label.shape[1], device=label.device)[None, :] >= valid_len[:, None]
        weights[mask] = 0
        
        self.reduction = 'none'
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(
            pred.permute(0, 2, 1), label)
        
        weighted_loss = (unweighted_loss * weights).sum()
        non_pad_elements = weights.sum()
        
        return weighted_loss / non_pad_elements if non_pad_elements > 0 else 0.0

def train(net, data_iter, lr, num_epochs, tgt_vocab, device):
    """Trains a sequence-to-sequence model."""
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])
    
    net.apply(xavier_init_weights)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = MaskedSoftmaxCELoss()
    
    net.train()
    print("Starting training...")
    for epoch in range(num_epochs):
        start_time = time.time()
        total_loss = 0
        num_batches = 0
        for batch in data_iter:
            optimizer.zero_grad()
            X, Y = [x.to(device) for x in batch]
            
            valid_len = (Y != tgt_vocab['<pad>']).type(torch.int32).sum(1)
            
            dec_input = torch.cat([torch.tensor(
                [tgt_vocab['<bos>']] * Y.shape[0], device=device).reshape(-1, 1),
                Y[:, :-1]], 1)
            
            Y_hat, _ = net(X, dec_input)
            l = loss(Y_hat, Y, valid_len)
            l.backward()
            
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
            optimizer.step()
            
            total_loss += l.item()
            num_batches += 1
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / num_batches:.4f}, Time: {time.time() - start_time:.2f}s')
    print("Training finished.")

# =============================================================================
# 4. Prediction/Inference
# =============================================================================
def translate(net, src_sentence, src_vocab, tgt_vocab, num_steps, device):
    """Translates a source sentence into the target language."""
    net.eval()
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [src_vocab['<eos>']]
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens = src_tokens + [src_vocab['<pad>']] * (num_steps - len(src_tokens))
    
    enc_X = torch.unsqueeze(torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs)
    
    dec_X = torch.unsqueeze(torch.tensor([tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)
    
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state)
        dec_X = Y.argmax(dim=2)
        pred_token_idx = dec_X.squeeze(dim=0).item()
        
        if hasattr(net.decoder, '_attention_weights_storage') and net.decoder._attention_weights_storage:
            attention_weight_seq.append(net.decoder._attention_weights_storage[0])
        
        if pred_token_idx == tgt_vocab['<eos>']:
            break
        output_seq.append(pred_token_idx)
    
    translation = ' '.join(tgt_vocab.to_tokens(output_seq))
    return translation, attention_weight_seq

# =============================================================================
# 5. Visualization
# =============================================================================
def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(5, 5), cmap='Reds'):
    """Displays heatmaps of attention matrices."""
    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.matshow(matrices.numpy(), cmap=cmap)
    fig.colorbar(cax)

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
        
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    if titles:
        ax.set_xticklabels([''] + titles[0])
        ax.set_yticklabels([''] + titles[1])

    plt.show()

# =============================================================================
# 6. Main Execution Block
# =============================================================================

if __name__ == '__main__':
    # --- Hyperparameters ---
    embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
    batch_size, num_steps = 64, 10
    lr, num_epochs = 0.005, 250
    
    # --- Data Loading ---
    train_iter, src_vocab, tgt_vocab = build_data(batch_size, num_steps)
    
    # --- Model Initialization ---
    encoder = Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
    decoder = Seq2SeqAttentionDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
    net = EncoderDecoder(encoder, decoder)
    
    # --- Training ---
    train(net, train_iter, lr, num_epochs, tgt_vocab, device)
    
    # --- Evaluation and Visualization ---
    engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
    fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
    
    for eng, fra in zip(engs, fras):
        translation, dec_attention_weight_seq = translate(net, eng, src_vocab, tgt_vocab, num_steps, device)
        candidate = [translation.split()]
        reference = [[fra.split()]]
        score = bleu_score(candidate, reference)
        print(f'{eng} => {translation}, bleu {score:.3f}')

    # Visualize attention for the last sentence
    eng_sentence_to_viz = engs[-1]
    translation, attention_weights = translate(net, eng_sentence_to_viz, src_vocab, tgt_vocab, num_steps, device)
    
    if attention_weights:
        num_output_steps = len(translation.split())
        num_input_tokens = len(eng_sentence_to_viz.split()) + 1
        
        attention_tensor = torch.cat([step[0].detach().cpu() for step in attention_weights[:num_output_steps]], 0)
        
        show_heatmaps(attention_tensor,
                      xlabel='Source Tokens', ylabel='Target Tokens',
                      titles=[eng_sentence_to_viz.split() + ['<eos>'], translation.split()])
    else:
        print("No attention weights to visualize.")