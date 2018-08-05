# -*- coding:utf-8 -*-
"""construct vocab and extract embedding from glove"""

import collections
import pickle
from tqdm import tqdm
import gzip
import numpy as np


def construct_vocab(save_path, src_path):
    PADDING = "<PAD>"
    UNK = "<UNK>"
    word_counter = collections.Counter()
    f = open(src_path, "rb")
    data_set = pickle.load(f)
    f.close()
    for key in data_set.keys():
        marker_type = key
        type_data = data_set[key]
        for item in tqdm(type_data):
            word_counter.update(item[0])
            word_counter.update(item[1])
    vocabulary = set([word for word in word_counter])
    vocabulary = list(vocabulary)
    vocabulary = [PADDING, UNK] + vocabulary
    f = gzip.open(save_path, "wb")
    pickle.dump(vocabulary, f)
    f.close()


def save_embedding(glove_text_path, vocab_path, save_path, divident=1.0):
    f = gzip.open(vocab_path, "rb")
    vocab = pickle.load(f)
    f.close()
    emb_dim = 300
    n = len(vocab)
    emb = np.empty((n, emb_dim), dtype=np.float32)
    emb[:, :] = np.random.normal(size=(n, emb_dim)) / divident
    # Explicitly assign embedding of <PAD> to be zeros.
    emb[0, :] = np.zeros((1, emb_dim), dtype=np.float32)
    word_indices = dict(zip(vocab, range(len(vocab))))

    with open(glove_text_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            s = line.split()
            if s[0] in word_indices:
                try:
                    emb[word_indices[s[0]], :] = np.asarray(s[1:])
                except ValueError:
                    print(s[0])
                    continue
    f = gzip.open(save_path, "wb")
    pickle.dump(emb, f)
    f.close()


def load_vocab(vocab_path):
    f = gzip.open(vocab_path, "rb")
    vocab = pickle.load(f)
    f.close()
    return vocab


def load_word_indices(vocab_path):
    vocab = load_vocab(vocab_path)
    word_indices = dict(zip(vocab, range(len(vocab))))
    return word_indices


def load_emb(emb_path):
    f = gzip.open(emb_path, "rb")
    emb = pickle.load(f)
    f.close()
    return emb


if __name__ == '__main__':
    data_set_path = "../bookcorpus/all_sentence_pairs.pkl"
    vocab_path = "../bookcorpus/vocab.pkl.gz"
    emb_path = "../bookcorpus/embedding.pkl.gz"
    glove_text_path = "glove.840B.300d.txt"
    print("+++++ construct vocab")
    construct_vocab(vocab_path, data_set_path)
    print("+++++ extract embedding")
    save_embedding(glove_text_path, vocab_path, emb_path)
