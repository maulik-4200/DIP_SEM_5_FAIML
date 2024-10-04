import gensim.downloader as api

# Load pre-trained GloVe word vectors (50-dimensional vectors in this example)
glove_vectors = api.load("glove-wiki-gigaword-50")

# Check the vocabulary size and vector size
vocab_size = len(glove_vectors.key_to_index)
vector_size = glove_vectors.vector_size
print(f"Vocabulary Size: {vocab_size}")
print(f"Vector Size: {vector_size}")

# Get the word vector for a specific word
word = "apple"
if word in glove_vectors:
    word_vector = glove_vectors[word]
    print(f"Vector for '{word}': {word_vector}")

# Find similar words to a given word
similar_words = glove_vectors.most_similar(word, topn=5)
print(f"Words similar to '{word}': {similar_words}")
