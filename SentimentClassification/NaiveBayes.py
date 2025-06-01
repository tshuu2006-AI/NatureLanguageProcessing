from datasets import load_dataset
import nltk
import string

from envs.Main.Lib.warnings import deprecated
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import numpy as np
from collections import defaultdict

#Download stopwords and punctuation
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

#Get data
def import_dataset(path):
    first_half = load_dataset(path, split = "train")
    second_half = load_dataset(path, split = "test")
    text = first_half['text'] + second_half['text']
    label = first_half["label"] + second_half['label']
    return text, label

#Remove stopwords, punctuation and tokenize the text
def preprocessing_data(texts, labels):
    stop_words = set(stopwords.words('english'))
    tokenized_sentences = []
    for text in texts:
        text = text.lower()
        tokens = nltk.tokenize.word_tokenize(text)
        tokens = [token for token in tokens if token not in stop_words and token not in string.punctuation]
        tokenized_sentences.append(tokens)
    dataset = train_test_split(tokenized_sentences, labels, stratify = labels, test_size=0.25)
    return dataset

@deprecated
def preprocessing_texts(texts):
    stop_words = set(stopwords.words('english'))

    tokenized_sentences = []
    for text in texts:
        text = text.lower()
        tokens = nltk.tokenize.word_tokenize(text)
        tokens = [token for token in tokens if token not in stop_words and token not in string.punctuation]
        tokenized_sentences.append(tokens)
    return tokenized_sentences

class NaiveBayesModel:
    def __init__(self):
        self.vocab = set()
        self.positive = defaultdict(int)
        self.negative = defaultdict(int)
        self.pos_probs = defaultdict(float)
        self.neg_probs = defaultdict(float)
        self.total_pos = 0
        self.total_neg = 0

    def fit(self, train, labels):

        if len(train) != len(labels):
            raise ValueError("x and y must have the same size")

        if len(set(labels)) != 2:
            raise ValueError("y must have only 2 values 0 and 1")

        for i in range(len(train)):
            tokens = train[i]
            label = labels[i]

        #Count the positive word and negative word
            if label == 1:
                for token in tokens:
                    self.vocab.add(token)
                    if token not in self.positive:
                        self.positive[token] = 0
                    self.positive[token] += 1
                    self.total_pos += 1

            else:
                for token in tokens:
                    self.vocab.add(token)
                    if token not in self.negative:
                        self.negative[token] = 0
                    self.negative[token] += 1
                    self.total_neg += 1



    def predict(self, texts):
        preds = []
        if type(texts[0]) == str:
            tokenized_sentences = preprocessing_texts(texts)
        else:
            tokenized_sentences = texts

        #Get the log-probabilities of the sentence using laplacian smoothing
        for tokens in tokenized_sentences:
            log_pos_probs = 0
            log_neg_probs = 0

            for token in tokens:
                pos_probs = (self.positive.get(token, 0) + 1) / (self.total_pos + len(self.vocab))
                neg_probs = (self.negative.get(token, 0) + 1) / (self.total_neg + len(self.vocab))
                log_pos_probs += np.log(pos_probs)
                log_neg_probs += np.log(neg_probs)

            if log_pos_probs > log_neg_probs:
                preds.append(1)
            else:
                preds.append(0)

        return preds

#Metrics
def accuracy(preds, true):
    preds = np.array(preds)
    true = np.array(true)
    total_correct = np.sum(preds == true)
    score = total_correct / len(true)
    return score


if __name__ == "__main__":
    x, y = import_dataset("imdb")
    x_train, x_test, y_train, y_test = preprocessing_data(x, y)
    model = NaiveBayesModel()
    model.fit(x_train, y_train)
    y_preds = model.predict(x_test)
    print(accuracy(y_preds, y_test))




