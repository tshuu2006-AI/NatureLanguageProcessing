from datasets import load_dataset
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
def load_data():
    dataset = load_dataset("conll2003")
    x = []
    y = []
    for cat in ('train', 'test', 'validation'):
        x += dataset[cat]['tokens']
        y += dataset[cat]['ner_tags']

    x = [" ".join(Tokens) for Tokens in x]
    tags_list = dataset["train"].features["ner_tags"].feature.names
    return x, y, tags_list


def get_max_seq_length(y) -> int:
    max_length = 0
    for tokens in y:
        max_length = max(max_length, len(tokens))
    return max_length

def label_preprocessing(y):
    return tf.keras.utils.pad_sequences(y, padding='post', value = -1, dtype='int64')

def data_preprocessing(x, y, vectorizer):
    """
    Preprocess data before feeding to network

    :param x: sentences.
    :param y: tags indices

    :return split data
    """
    y = label_preprocessing(y)
    #Split data
    x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.3, random_state=42)
    x_test, x_valid, y_test, y_valid = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

    x_train = vectorizer(x_train)
    x_valid = vectorizer(x_valid)
    return x_train, y_train, x_valid, y_valid, x_test, y_test


def text_vectorizer(x, max_length, max_tokens = 500):
    vectorizer = tf.keras.layers.TextVectorization(max_tokens = max_tokens, standardize = None, output_mode = 'int', output_sequence_length=max_length)
    vectorizer.adapt(x)
    return vectorizer

class LstmModel:

    def __init__(self, num_tags, embedding_size = 500, bidirectional = True, text_vectorizer = None):
        if text_vectorizer == None:
            raise ValueError("missing argument 'text_vectorizer'")
        self.vectorizer = text_vectorizer
        vocab_size = text_vectorizer.vocabulary_size()

        self.model = tf.keras.Sequential(name = "ner_model")
        self.model.add(tf.keras.layers.Embedding(input_dim = vocab_size + 1, output_dim = embedding_size, mask_zero = True))
        if bidirectional:
            self.model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units = embedding_size, return_sequences = True)))
        else:
            self.model.add(tf.keras.layers.LSTM(units = embedding_size, return_sequences = True))
        self.model.add(tf.keras.layers.Dense(units = num_tags))

    def compile(self, optimizer = 'adam', lr = 0.005):
        if optimizer == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate = lr)
        self.model.compile(optimizer = optimizer, loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True, ignore_class = -1))

    def fit(self, x_train, y_train, x_valid, y_valid, epochs = 2, batch_size = 32):
        self.model.fit(x_train, y_train, validation_data = (x_valid, y_valid), epochs = epochs, batch_size = batch_size)

    def predict(self, x):
        vectorized_x = self.vectorizer(x)
        y_pred = self.model.predict(vectorized_x)
        return y_pred

    def accuracy(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.math.argmax(y_pred, axis=-1)
        y_pred = tf.cast(y_pred, y_true.dtype)
        mask = tf.math.not_equal(y_true, -1)
        mask = tf.cast(mask, y_true.dtype)

        y_pred_class = y_pred * mask
        matches_classes = y_pred_class == y_true
        matches_classes = tf.cast(matches_classes, y_true.dtype)

        acc = tf.reduce_sum(matches_classes) / tf.reduce_sum(mask)
        return acc


if __name__ == "__main__":
    x, y, tags = load_data()
    max_length = get_max_seq_length(y)
    vectorizer = text_vectorizer(x, max_length=max_length)
    x_train, y_train, x_valid, y_valid, x_test, y_test = data_preprocessing(x, y, vectorizer)
    lstm = LstmModel(num_tags=len(tags), bidirectional=False, text_vectorizer=vectorizer)
    lstm.compile()
    lstm.fit(x_train, y_train, x_valid, y_valid)
    y_preds1 = lstm.predict(x_test)
    print(lstm.accuracy(y_test, y_preds1))
    biLSTM = LstmModel(num_tags=len(tags), bidirectional=True, text_vectorizer=vectorizer)
    biLSTM.compile()
    biLSTM.fit(x_train, y_train, x_valid, y_valid)
    y_preds1 = biLSTM.predict(x_test)
    print(biLSTM.accuracy(y_test, y_preds1))