import collections

# Initialize the symbol vocabulary with all lowercase letters and special tokens
symbols = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
           'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
           '_', '[UNK]']  # '_' represents word end, '[UNK]' for unknown symbols

# Raw token frequencies in our training corpus
raw_token_freqs = {'fast_': 4, 'faster_': 3, 'tall_': 5, 'taller_': 4}

# Convert tokens to space-separated character sequences for processing
# e.g., 'fast_' becomes 'f a s t _'
token_freqs = {}
for token, freq in raw_token_freqs.items():
    token_freqs[' '.join(list(token))] = raw_token_freqs[token]

print("Initial token frequencies:", token_freqs)


def get_max_freq_pair(token_freqs):
    """
    Find the most frequent pair of adjacent symbols in the token frequencies.
    
    Args:
        token_freqs: Dictionary with tokens as keys and frequencies as values
        
    Returns:
        The most frequent pair of adjacent symbols as a tuple
    """
    pair_frequencies = collections.defaultdict(int)
    
    # Count frequencies of all adjacent symbol pairs
    for token, freq in token_freqs.items():
        symbols_in_token = token.split()  # Split token into individual symbols
        for i in range(len(symbols_in_token) - 1):
            # Create a tuple of adjacent symbols and add their frequency
            current_pair = (symbols_in_token[i], symbols_in_token[i + 1])
            pair_frequencies[current_pair] += freq
    
    # Return the pair with the highest frequency
    return max(pair_frequencies, key=pair_frequencies.get)


def merge_symbol_pair(pair_to_merge, token_freqs, symbols):
    """
    Merge a pair of symbols into a new symbol and update the token frequencies.
    
    Args:
        pair_to_merge: Tuple of two symbols to merge
        token_freqs: Current token frequencies dictionary
        symbols: Current symbol vocabulary
        
    Returns:
        Updated token frequencies after merging the pair
    """
    # Create new symbol by joining the pair
    new_symbol = ''.join(pair_to_merge)
    symbols.append(new_symbol)
    
    # Create updated token frequencies with merged symbols
    updated_token_freqs = {}
    
    for token, freq in token_freqs.items():
        # Replace the space-separated pair with the merged symbol
        # e.g., 't a' becomes 'ta'
        updated_token = token.replace(' '.join(pair_to_merge), new_symbol)
        updated_token_freqs[updated_token] = freq
    
    return updated_token_freqs


# Perform Byte Pair Encoding (BPE) for specified number of merges
num_merges = 10
print("\nPerforming BPE merges:")
for i in range(num_merges):
    # Find and merge the most frequent symbol pair
    most_frequent_pair = get_max_freq_pair(token_freqs)
    token_freqs = merge_symbol_pair(most_frequent_pair, token_freqs, symbols)
    print(f'Merge #{i+1}: {most_frequent_pair} -> {"".join(most_frequent_pair)}')

print("\nFinal symbol vocabulary:", symbols)
print("Final token representations:", list(token_freqs.keys()))


def segment_with_bpe(tokens, symbols):
    """
    Segment tokens using the BPE-learned symbol vocabulary.
    Uses greedy longest-match-first approach.
    
    Args:
        tokens: List of tokens to segment
        symbols: BPE-learned symbol vocabulary
        
    Returns:
        List of segmented tokens (as space-separated symbols)
    """
    segmented_results = []
    
    for token in tokens:
        start_pos, end_pos = 0, len(token)
        current_segmentation = []
        
        # Greedy segmentation: always try to match the longest possible symbol first
        while start_pos < len(token) and start_pos < end_pos:
            current_segment = token[start_pos:end_pos]
            
            if current_segment in symbols:
                # Found a matching symbol, add to result and move forward
                current_segmentation.append(current_segment)
                start_pos = end_pos  # Move start to after the matched symbol
                end_pos = len(token)  # Reset end to token end
            else:
                # No match, try a shorter segment
                end_pos -= 1
        
        # Handle any remaining characters that couldn't be matched
        if start_pos < len(token):
            current_segmentation.append('[UNK]')  # Unknown symbol
        
        segmented_results.append(' '.join(current_segmentation))
    
    return segmented_results


# Test BPE segmentation on new tokens
test_tokens = ['tallest_', 'fatter_']
print(f"\nSegmenting test tokens {test_tokens}:")
segmented = segment_with_bpe(test_tokens, symbols)
print("Segmentation results:", segmented)