import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE # Import TSNE for potential use (often better for non-linear structures)

# --- 1. Load GloVe Embeddings ---
def load_glove_embeddings(glove_file):
    """
    Loads pre-trained GloVe word vectors from a file.

    The file format is: word followed by its vector components (space-separated).
    """
    embeddings = {}
    # Use 'utf8' encoding for standard text files
    with open(glove_file, 'r', encoding='utf8') as f:
        for line in f:
            values = line.split()
            # The first element is the word
            word = values[0]
            # The rest are vector components, converted to float32 NumPy array
            vector = np.array(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

# NOTE: You must have the 'glove.6B.50d.txt' file in the same directory.
# Download URL: https://nlp.stanford.edu/projects/glove/
glove_file = 'glove.6B.50d.txt' 
try:
    embeddings = load_glove_embeddings(glove_file)
except FileNotFoundError:
    print(f"Error: GloVe file '{glove_file}' not found. Please download it first.")
    # Use a dummy dictionary if file is missing to prevent crash
    embeddings = {'king': np.zeros(50), 'queen': np.zeros(50), 'man': np.zeros(50)}
    

# --- 2. Query and Similarity Example ---
print(f"Total words loaded: {len(embeddings)}")
# Check the vector for "king"
if 'king' in embeddings:
    print(f"\nVector for 'king' (first 5 dims):\n{embeddings['king'][:5]}...")
else:
    print("\n'king' not found in embeddings (check file integrity or case).")

# Calculate similarity example
if 'king' in embeddings and 'queen' in embeddings:
    vec1 = embeddings['king'].reshape(1, -1)
    vec2 = embeddings['queen'].reshape(1, -1)
    similarity = cosine_similarity(vec1, vec2)
    print(f"\nCosine similarity (king vs queen): {similarity[0][0]:.4f}")
else:
    print("\nCannot perform similarity calculation: one or more words missing.")


# --- 3. Visualization Function (using PCA) ---
def display_pca_scatterplot(embeddings_dict, words_to_plot):
    """
    Reduces word vectors to 2D using PCA and plots the scatterplot.
    
    Args:
        embeddings_dict (dict): Dictionary mapping words to their vectors.
        words_to_plot (list): List of words to include in the plot.
    """
    # 1. Filter out words not present in the embeddings dictionary
    valid_words = [w for w in words_to_plot if w in embeddings_dict]
    
    if not valid_words:
        print("\nNo valid words found for plotting.")
        return

    # 2. Extract vectors
    word_vectors = np.array([embeddings_dict[w] for w in valid_words])
    
    # 3. Apply PCA for dimensionality reduction (to 2 components)
    # n_components=2 specifies reduction to 2 dimensions
    twodim = PCA(n_components=2).fit_transform(word_vectors)
    
    # 4. Plotting
    plt.figure(figsize=(10, 8)) 
    plt.scatter(twodim[:, 0], twodim[:, 1], edgecolors='k', c='red', s=80)
    
    # Annotate points with word labels
    for word, (x, y) in zip(valid_words, twodim):
        # Annotate slightly offset for better visibility
        plt.text(x + 0.05, y + 0.05, word, fontsize=10)
    
    plt.title("GloVe Embeddings Visualization (PCA)", fontsize=16)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

# --- 4. Execution of Visualization ---

# Define a set of words that are interesting or semantically related
words_for_plot = [
    'king', 'queen', 'man', 'woman', 'prince', 'princess', 
    'apple', 'microsoft', 'google', 'facebook', 'company',
    'car', 'bus', 'train', 'airplane', 'transportation',
    'love', 'hate', 'joy', 'sadness', 'emotion'
]

display_pca_scatterplot(embeddings, words_for_plot)