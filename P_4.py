import nltk

# ------------ Download nltk resources ------------
# nltk.download('punkt_tab')
# nltk.download('averaged_perceptron_tagger_eng') 
# nltk.download('stopwords')
# nltk.download('words')

# ------------ a ------------
from nltk.tokenize import word_tokenize, sent_tokenize

text = "Natural Language Processing (NLP) is a field of artificial intelligence that focuses on the interaction between computers and hum"

# Tokenize sentence-wise
sentences = sent_tokenize(text)
print("Sentences:", sentences)
print()
words_sentence_wise = [word_tokenize(sentence) for sentence in sentences]
print("Tokenize sentence-wise:", words_sentence_wise)
print()


# ------------ b ------------
from nltk.stem import PorterStemmer

# Initialize PorterStemmer
stemmer = PorterStemmer()

# Stem each token
stemmed_words = [[stemmer.stem(word) for word in sentence] for sentence in words_sentence_wise]
print("Stemmed words:", stemmed_words)
print()


# ------------ c ------------
# Perform POS tagging
pos_tags = [nltk.pos_tag(sentence) for sentence in words_sentence_wise]
print("POS Tags:", pos_tags)
print()


# ------------ d ------------
from nltk.corpus import stopwords

# Get the set of stopwords for English
stop_words = set(stopwords.words('english'))

# Filter out the stop words
filtered_sentence_wise = [
    [word for word in sentence if word.lower() not in stop_words]
    for sentence in words_sentence_wise
]
for i, filtered_words in enumerate(filtered_sentence_wise):
    print(f"Filtered sentence {i+1}: {filtered_words}")
print()


# ------------ e ------------
from nltk.corpus import words
from nltk.metrics import edit_distance

valid_words = set(words.words())
text = "Ths is an exampel of a simple speall chcker."
misspelled_words = [word for sentence in [word_tokenize(sentence) for sentence in sent_tokenize(text)] for word in sentence if word.lower() not in valid_words]


def correct_spelling(word):
    suggestions = [w for w in valid_words if edit_distance(word, w) <= 1]
    return suggestions


for word in misspelled_words:
    suggestions = correct_spelling(word)
    if suggestions:
        print(f"Word: {word}, Suggestions: {', '.join(suggestions)}")
