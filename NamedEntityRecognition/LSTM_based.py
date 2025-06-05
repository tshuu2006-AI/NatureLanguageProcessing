from datasets import load_dataset
import tensorflow as tf

def load_data():
    dataset = load_dataset("conll2003")
    x = []
    y = []
    for cat in ('train', 'test', 'validate'):
        x += dataset[cat]['tokens']
        y += dataset[cat]['ner_tags']

    tags_list = dataset["train"].features["ner_tags"].feature.names
    return x, y, tags_list


def get_max_seq_length(x) -> int:
    max_length = 0
    for tokens in x:
        max_length = max(max_length, len(tokens))
    return max_length

def text_vectorizer(x):
    
def data_preprocessing(x, y, tags_list):
    for