import collections
import re
import random
import torch
from d2l import torch as d2l # Import d2l library
import os
import sys

# Set path to ensure read_time_machine can find the file
# Note: You need to ensure the file path '../data/timemachine.txt' is correct.
if not os.path.exists('../data/timemachine.txt'):
    # Temporarily download the file if not already present
    d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt',
                                    '090b5e7e70c295757f55df93cb0a180b9691891a')
    d2l.download('time_machine', root='../data')

def read_time_machine():  #@save
    """Load the time machine dataset into a list of text lines"""
    # Ensure the file exists and use the correct path
    try:
        with open('../data/timemachine.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print("Error: File not found. Please ensure timemachine.txt is in the ../data/ directory.")
        sys.exit()

    # Clean the text: remove non-alphabetic characters, strip whitespace, and convert to lowercase
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

def count_corpus(tokens):  #@save
    """Count token frequencies"""
    # tokens can be a 1D list or a 2D list
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # Flatten the list of tokens into a single list
        tokens = [token for line in tokens for token in line]
    # tokens can be a list of strings (unigrams) or a list of tuples (bigrams/trigrams)
    return collections.Counter(tokens)

# Use a local Vocab class to avoid potential str/tuple sorting errors in the d2l library
class LocalVocab:
    """Local Vocabulary implementation, compatible with str and tuple tokens"""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
            
        # 1. Count and sort by frequency
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)
        
        # 2. Initialize the vocabulary: '<unk>' and reserved tokens are strings
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}
        
        # 3. Add tokens to the vocabulary by frequency (no need to sort again)
        for token, freq in self._token_freqs:
            # token can be str (unigram) or tuple (n-gram)
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    @property
    def token_freqs(self):
        return self._token_freqs
    
    def __len__(self):
        return len(self.idx_to_token)
    
# -------------------- Execution Code --------------------

# 1. Prepare unigram data
tokens = d2l.tokenize(read_time_machine(), token='word')
corpus = [token for line in tokens for token in line] # Flatten into a 1D list

# 2. Count and print unigram frequencies (using LocalVocab)
vocab = LocalVocab(corpus)
print(f"Total number of unigram tokens: {len(corpus)}")
print(f"Unigram vocabulary size: {len(vocab)}")
print('--- Unigram Frequencies TOP 10 ---')
print(vocab.token_freqs[:10])

# 3. Generate Bigram tokens
# Bigram: pairs of adjacent tokens
bigram_tokens = [pair for pair in zip(corpus[:-1], corpus[1:])]

# 4. Count and print Bigram frequencies (using LocalVocab)
bigram_vocab = LocalVocab(bigram_tokens)

print('\n--- Bigram Frequencies TOP 10 ---')
# Here, the token is a tuple, e.g., ('of', 'the')
print(bigram_vocab.token_freqs[:10])

# 5. Generate Trigram tokens
# Trigram: triples of adjacent tokens
trigram_tokens = [triple for triple in zip(corpus[:-2], corpus[1:-1], corpus[2:])]

# 6. Count and print Trigram frequencies (using LocalVocab)
trigram_vocab = LocalVocab(trigram_tokens)
print('\n--- Trigram Frequencies TOP 10 ---')
print(trigram_vocab.token_freqs[:10])

# 7. Plotting the frequency distribution (Zipf's Law)
freqs = [freq for token, freq in vocab.token_freqs]
bigram_freqs = [freq for token, freq in bigram_vocab.token_freqs]
trigram_freqs = [freq for token, freq in trigram_vocab.token_freqs]

# Plotting with log-log scale
d2l.plot([freqs, bigram_freqs, trigram_freqs], xlabel='token: x',
         ylabel='frequency: n(x)', xscale='log', yscale='log',
         legend=['unigram', 'bigram', 'trigram'])
d2l.plt.show() # Display the plot