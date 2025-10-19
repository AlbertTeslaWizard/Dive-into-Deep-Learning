import math
import os
import random
from collections import Counter
import torch
from d2l import torch as d2l

# Download the PTB dataset
d2l.DATA_HUB['ptb'] = (d2l.DATA_URL + 'ptb.zip',
                        '319d85e578af0cdc590547f26231e4e31cdf1e42')

# ------------------ Data Processing Functions ------------------
def read_ptb():
    """Load the PTB dataset into a list of text lines"""
    data_dir = d2l.download_extract('ptb')
    with open(os.path.join(data_dir, 'ptb.train.txt')) as f:
        raw_text = f.read()
    return [line.split() for line in raw_text.split('\n')]

def count_corpus(tokens):
    """Count token frequencies"""
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # Flatten a list of lists of tokens into a single list
        tokens = [token for line in tokens for token in line]
    return Counter(tokens)

def subsample(sentences, vocab):
    """Subsample high-frequency words"""
    # Remove unknown tokens
    sentences = [[token for token in line if vocab[token] != vocab.unk]
                   for line in sentences]
    counter = count_corpus(sentences)
    num_tokens = sum(counter.values())

    def keep(token):
        # Subsampling probability calculation
        return random.uniform(0, 1) < math.sqrt(1e-4 / counter[token] * num_tokens)

    # Apply subsampling
    return [[token for token in line if keep(token)] for line in sentences], counter

def get_centers_and_contexts(corpus, max_window_size):
    """Return center words and context words for the Skip-Gram model"""
    centers, contexts = [], []
    for line in corpus:
        if len(line) < 2:
            continue
        centers += line
        for i in range(len(line)):
            # Randomly select a window size
            window_size = random.randint(1, max_window_size)
            # Find indices for context words
            indices = list(range(max(0, i - window_size),
                                 min(len(line), i + 1 + window_size)))
            # Exclude the center word itself
            indices.remove(i)
            contexts.append([line[idx] for idx in indices])
    return centers, contexts

# ------------------ Random Generator ------------------
class RandomGenerator:
    """Draw samples from {1,...,n} based on n sampling weights"""
    def __init__(self, sampling_weights):
        # Population is token indices (1 to len(vocab)-1, excluding <unk>)
        self.population = list(range(1, len(sampling_weights) + 1))
        self.sampling_weights = sampling_weights
        self.candidates = []
        self.i = 0

    def draw(self):
        # Refill the candidates list when exhausted
        if self.i == len(self.candidates):
            self.candidates = random.choices(
                self.population, self.sampling_weights, k=10000)
            self.i = 0
        self.i += 1
        return self.candidates[self.i - 1]

# ------------------ Negative Sampling ------------------
def get_negatives(all_contexts, vocab, counter, K):
    """Return noise words for negative sampling"""
    # Calculate sampling weights (power of 0.75 for unigram distribution)
    sampling_weights = [counter[vocab.to_tokens(i)]**0.75
                        for i in range(1, len(vocab))]
    all_negatives, generator = [], RandomGenerator(sampling_weights)
    for contexts in all_contexts:
        negatives = []
        # Generate K times more negative samples than positive context words
        while len(negatives) < len(contexts) * K:
            neg = generator.draw()
            # Ensure the negative sample is not a positive context word
            if neg not in contexts:
                negatives.append(neg)
        all_negatives.append(negatives)
    return all_negatives

# ------------------ Minibatch Collation ------------------
def batchify(data):
    """Return a minibatch of samples for Skip-Gram with negative sampling"""
    # Find the maximum sequence length (context + negative samples)
    max_len = max(len(c) + len(n) for _, c, n in data)
    centers, contexts_negatives, masks, labels = [], [], [], []
    for center, context, negative in data:
        cur_len = len(context) + len(negative)
        centers += [center]
        # Pad context and negative samples
        contexts_negatives += [context + negative + [0] * (max_len - cur_len)]
        # Create mask: 1 for real tokens, 0 for padding
        masks += [[1] * cur_len + [0] * (max_len - cur_len)]
        # Create labels: 1 for positive context, 0 for negative samples/padding
        labels += [[1] * len(context) + [0] * (max_len - len(context))]
        
    return (torch.tensor(centers).reshape((-1, 1)),
            torch.tensor(contexts_negatives),
            torch.tensor(masks),
            torch.tensor(labels))

# ------------------ Dataset Class ------------------
class PTBDataset(torch.utils.data.Dataset):
    """PTB Dataset wrapper"""
    def __init__(self, centers, contexts, negatives):
        assert len(centers) == len(contexts) == len(negatives)
        self.centers = centers
        self.contexts = contexts
        self.negatives = negatives

    def __getitem__(self, index):
        return (self.centers[index], self.contexts[index], self.negatives[index])

    def __len__(self):
        return len(self.centers)

# ------------------ Load Data ------------------
def load_data_ptb(batch_size, max_window_size, num_noise_words):
    """Download the PTB dataset and return the DataLoader and vocabulary"""
    # The num_workers will be set to 0 to avoid the multiprocessing error on Windows 
    # as advised in the previous response, or it can be obtained via d2l.get_dataloader_workers() 
    # if the code is structured correctly with `if __name__ == '__main__':`.
    # For this translated code, I'll keep the user's provided 'num_workers=0' in the DataLoader call
    # which is often necessary on Windows when the entry point is not wrapped.
    num_workers = d2l.get_dataloader_workers() # This line is kept but its output is ignored later.
    
    print("Starting to read PTB dataset...")
    sentences = read_ptb()
    print(f"Reading complete. Total {len(sentences)} lines.")
    
    vocab = d2l.Vocab(sentences, min_freq=10)
    print(f"Vocabulary built. Vocabulary size: {len(vocab)}")
    
    subsampled, counter = subsample(sentences, vocab)
    print(f"Subsampling complete. Remaining tokens: {sum(counter.values())}")
    
    corpus = [vocab[line] for line in subsampled]
    print("Converted to token indices.")
    
    print("Starting to generate center words and contexts...")
    all_centers, all_contexts = get_centers_and_contexts(corpus, max_window_size)
    print(f"Center words and contexts generated. Total {len(all_centers)} center words.")
    
    print("Starting to generate negative samples (this step will be very slow, please wait patiently)...")
    all_negatives = get_negatives(all_contexts, vocab, counter, num_noise_words)
    print("Negative sampling complete.")
    
    dataset = PTBDataset(all_centers, all_contexts, all_negatives)
    print("Dataset created.")
    
    # NOTE: num_workers is explicitly set to 0 here to prevent the RuntimeError on Windows,
    # which was the issue mentioned in your previous message.
    data_iter = torch.utils.data.DataLoader(
        dataset, batch_size, shuffle=True,
        collate_fn=batchify, num_workers=0) 
    
    print("DataLoader created. Ready to start iteration.")
    return data_iter, vocab

# ------------------ Test ------------------
names = ['centers', 'contexts_negatives', 'masks', 'labels']

data_iter, vocab = load_data_ptb(batch_size=512, max_window_size=5, num_noise_words=5)

# Iteration start (The part where the original error occurred if num_workers > 0 and no __main__ guard)
for batch in data_iter:
    for name, data in zip(names, batch):
        print(name, 'shape:', data.shape)
    break # Only take one batch for testing