from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec
import numpy as np

# Sample corpus
corpus = [
    "I love Python",
    "Python is a great language",
    "I enjoy solving problems in Python"
]

# Bag of Words (BoW)
def bag_of_words(corpus):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)
    return X.toarray(), vectorizer.get_feature_names_out()

# TF-IDF
def tf_idf(corpus):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    return X.toarray(), vectorizer.get_feature_names_out()

# Word2Vec
def word2vec(corpus):
    tokenized_corpus = [sentence.lower().split() for sentence in corpus]
    
    # Train Word2Vec model
    model = Word2Vec(sentences=tokenized_corpus, vector_size=100, window=5, min_count=1, workers=4)
    
    # Create sentence embeddings by averaging word vectors
    sentence_embeddings = []
    for sentence in tokenized_corpus:
        sentence_vector = np.mean([model.wv[word] for word in sentence if word in model.wv], axis=0)
        sentence_embeddings.append(sentence_vector)
        
    return sentence_embeddings

# Running BoW
bow_matrix, bow_features = bag_of_words(corpus)
print("Features (subset):\n", bow_features)
print("Bag of Words:\n", bow_matrix)

# Running TF-IDF
tfidf_matrix, tfidf_features = tf_idf(corpus)
print("\nTF-IDF:\n", tfidf_matrix)

# Running Word2Vec
word2vec_embeddings = word2vec(corpus)
print("\nWord2Vec Embeddings (first sentence):\n", word2vec_embeddings[:1])
