import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE  # TSNE is imported but not used, kept for reference

# --- 1. Data Loading and Preprocessing ---
# Load the dataset
dataset = pd.read_csv('Corona_NLP_train.csv', encoding='latin1')

# List to hold cleaned and tokenized text
cleaned_texts = [] 
# The number of rows to process
num_rows = len(dataset)

# Clean up each tweet
for i in range(num_rows):
    # Remove all characters except letters (keeping only a-z, A-Z)
    text = re.sub('[^a-zA-Z]', ' ', dataset['OriginalTweet'][i])
    # Convert to lowercase
    text = text.lower()
    # Tokenize the text
    tokens = text.split()
    
    # --- URL and 'https' cleanup logic ---
    # The original logic intended to remove 'https' and subsequent tokens.
    # A cleaner approach is to filter out all URL-related tokens.
    # However, keeping the original simple 'https' removal logic as requested for modification.
    try:
        # Find index of 'https' if it exists
        https_index = tokens.index('https')
        # Truncate the list at the first occurrence of 'https'
        tokens = tokens[:https_index]
    except ValueError:
        # 'https' not found, do nothing
        pass
    
    # Filter out any remaining 'https' tokens (if they were not at the start)
    final_tokens = [t for t in tokens if t != 'https']
    
    # Store the list of tokens for Word2Vec training
    cleaned_texts.append(final_tokens)

# --- 2. Word2Vec Model Training ---
# Use the cleaned_texts (list of token lists) directly
sentences = cleaned_texts
w2v = Word2Vec(
    sentences, 
    vector_size=100,  # Dimensionality of the word vectors
    window=5,         # Maximum distance between the current and predicted word
    workers=4,        # Use 4 threads for training (parallel processing)
    epochs=10,        # Number of epochs (iterations) over the corpus (replaces 'iter')
    min_count=5       # Ignore all words with total frequency lower than 5
)

# --- 3. Model Query/Test ---
# Get the list of all words in the vocabulary (Gensim 4.0+ compatible)
vocabulary = w2v.wv.index_to_key

# Function to safely print word vector
def safe_print_vector(model, word):
    if word in model.wv:
        print(f"Vector for '{word}':\n{model.wv[word][:5]}...") # Print first 5 dimensions
    else:
        print(f"'{word}' is not in the vocabulary (min_count was likely too high).")

# Perform queries
safe_print_vector(w2v, 'computer')
print("\nMost similar words to 'pay':")
print(w2v.wv.most_similar('pay'))
print("\nAnalogy: 'russian' - 'arab' + 'russia' = ?")
print(w2v.wv.most_similar(positive=['russia', 'russian'], negative=['arab']))


# --- 4. PCA Visualization Function (Gensim 4.0+ Compatible) ---
def display_pca_scatterplot(model, words_to_plot=None, sample_size=0):
    """
    Performs PCA on word vectors and plots the results.

    Args:
        model (Word2Vec): The trained Word2Vec model.
        words_to_plot (list, optional): Specific words to include in the plot.
        sample_size (int): Number of random words to sample if words_to_plot is None.
    """
    wv = model.wv # Alias for KeyedVectors
    
    # Determine the words to be plotted
    if words_to_plot is None:
        # If no specific words are provided, sample randomly from the entire vocabulary
        if sample_size > 0:
            words_to_plot = np.random.choice(wv.index_to_key, sample_size, replace=False)
        else:
            # If sample_size is 0, plot all words (might be too slow/cluttered)
            words_to_plot = wv.index_to_key
            
    # Filter for words that actually exist in the model's vocabulary
    valid_words = [w for w in words_to_plot if w in wv]
    
    if not valid_words:
        print("No valid words found in the model's vocabulary for plotting.")
        return

    # Extract word vectors for the valid words
    word_vectors = np.array([wv[w] for w in valid_words])

    # Perform PCA to reduce dimensionality to 2
    twodim = PCA(n_components=2).fit_transform(word_vectors)
    
    # Create the scatter plot
    plt.figure(figsize=(10,10)) # Increased size for better readability 
    plt.scatter(twodim[:,0], twodim[:,1], edgecolors='k', c='r', s=50) # Increased marker size
    
    # Annotate points with word labels
    for word, (x,y) in zip(valid_words, twodim):
        plt.text(x + 0.05, y + 0.05, word, fontsize=9)
    
    plt.title("Word2Vec PCA Visualization", fontsize=16)
    plt.xlabel("Principal Component 1 (PC1)")
    plt.ylabel("Principal Component 2 (PC2)")
    plt.grid(True)
    plt.tight_layout()
    plt.show() # Display the plot

# --- 5. Execution ---
words_of_interest = [
    'coronavirus', 'covid', 'virus', 'corona', 'disease', 
    'saudiarabia', 'doctor', 'hospital', 'pakistan', 'kenya',
    'pay', 'paying', 'paid', 'wages', 'raise', 'bills', 'rent', 'charge'
] 

display_pca_scatterplot(w2v, words_to_plot=words_of_interest)