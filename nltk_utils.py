import nltk
nltk.download('punkt')
nltk.download('stopwords')
import numpy as np
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

factory = StemmerFactory()
stemmer = factory.create_stemmer()

indonesian_stopwords = nltk.corpus.stopwords.words('indonesian')

def tokenize(sentence):
    words = nltk.word_tokenize(sentence)
    words = [w for w in words if w not in indonesian_stopwords]
    return words

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    """
    tokenized_sentence = ["info", "wisata", "kawah", "putih", "kabupaten", "bandung"]
    all_words = ["info", "wisata", "kawah", "putih", "di", "kabupaten", "bandung", "yang"]
    bag = [1. 1. 1. 1. 0. 1. 1. 0.]
    """
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0

    return bag

# Testing Tokenize
# sentence = "Beritahu Saya dan Infokan Wisata Kawah Putih di Kabupaten Bandung!"
# words = tokenize(sentence)
# print(words)
# Output: ['Beritahu', 'Saya', 'Infokan', 'Wisata', 'Kawah', 'Putih', 'Kabupaten', 'Bandung', '!']

# Testing Stopwords
# tokenize_word = ["Beritahu", "Saya", "dan", "Infokan", "Wisata", "Kawah", "Putih", "di", "Kabupaten", "Bandung", "!"]
# stopwords = [w for w in tokenize_word if w not in indonesian_stopwords]
# print(stopwords)
# Output: ['Beritahu', 'Infokan', 'Wisata', 'Kawah', 'Putih', 'Kabupaten', 'Bandung', '!']

# Testing Case Folding
# word = ['Beritahu', 'Saya', 'Infokan', 'Wisata', 'Kawah', 'Putih', 'Kabupaten', 'Bandung', '!']
# case_folding = [w.lower() for w in word]
# print(case_folding)
# Output: ['beritahu', 'saya', 'infokan', 'wisata', 'kawah', 'putih', 'kabupaten', 'bandung', '!']

# Testing Stemming
# words = ['beritahu', 'saya', 'infokan', 'wisata', 'kawah', 'putih', 'kabupaten', 'bandung', '!']
# stemming = [stemmer.stem(w) for w in words]
# print(stemming)

# Testing Bag of Words
# tokenized_sentence = ["info", "wisata", "kawah", "putih", "kabupaten", "bandung"]
# all_words = ["info", "wisata", "kawah", "putih", "di", "kabupaten", "bandung", "yang"]
# bag = bag_of_words(tokenized_sentence, all_words)
# print(bag)
# Output: [1. 1. 1. 1. 0. 1. 1. 0.]


