import numpy as np
import re
import itertools
from collections import Counter
import xml.etree.ElementTree
import word2vec
from nltk.tokenize import sent_tokenize




def load_data_and_labels():
    
    root = xml.etree.ElementTree.parse('corpus.xml').getroot()
    corpus = XmlDictConfig(root)
    root = xml.etree.ElementTree.parse('annotations.xml').getroot()
    annotations = XmlDictConfig(root)

    #for each listno, make a list of sentences, label some as summaries, others as not.


    return [x_text, y]

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def embed(sentences):
    model = word2vec.load('~/word2vec_models/GoogleNews-vectors-negative300.bin')
    embedded_sentences = []
    tokenized_sentences = []

    max_len = 0
    for sentence in sentences:
        tokenized_sentence = sent_tokenize(sentence)
        tokenized_sentences.append(tokenized_sentence)
        if len(tokenized_sentence) > max_len:
            max_len = len(tokenized_sentence)


    for sentence in sentences:
        tokenized_sentence = sent_tokenize(sentence)
        embedded_words = []
        
        for word in tokenized_sentence:
            try:
                word = model['word']
            except:
                word = np.zeros(300)
            embedded_words.append(word)

        #padding    
        for i in range(max_len - len(embedded_words)):
            embedded_words.append(np.zeros(300))

        embedded_sentences.append(embedded_words)

    embedded_sentences = np.array(embedded_sentences)

    return embedded_sentences







def load_data():
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels = load_data_and_labels()
    sentences_padded = pad_sentences(sentences)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    return [x, y, vocabulary, vocabulary_inv]


def batch_iter(data, batch_size, num_epochs):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
