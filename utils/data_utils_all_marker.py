# -*- coding:utf-8 -*-

from utils import vocab
import gzip
import pickle
from tqdm import tqdm
import random
import numpy as np


Markers = ['because', 'though', 'for example', 'after', 'when', 'however', 'but', 'if', 'while', 'so', 'although', 'meanwhile', 'still', 'before']
Markers_map = dict(zip(Markers, range(len(Markers))))


def convert_to_ids(sentence, word_indices):
    result = []
    for word in sentence:
        if word not in word_indices:  # unk word
            result.append(1)
        else:
            result.append(word_indices[word])
    return result


def shuffle_data_set(src_path, save_path, word_indices):
    f = open(src_path, "rb")
    data_set = pickle.load(f)
    f.close()
    new_data_set = []
    for key in data_set.keys():
        marker_type = key
        marker_id = Markers_map[marker_type]
        type_data = data_set[key]
        for item in tqdm(type_data):
            new_item = {
                "marker_id": marker_id,
                "sen_1": item[0],
                "sen_2": item[1],
                "sen_1_ids": convert_to_ids(item[0], word_indices),
                "sen_2_ids": convert_to_ids(item[1], word_indices)
            }
            new_data_set.append(new_item)

    random.shuffle(new_data_set)
    random.shuffle(new_data_set)
    random.shuffle(new_data_set)

    f = gzip.open(save_path, "wb")
    pickle.dump(new_data_set, f)
    f.close()


def load_all_data(emb_path, converted_data_set_path):
    embedding = vocab.load_emb(emb_path)
    f = gzip.open(converted_data_set_path, "rb")
    data_set = pickle.load(f)
    f.close()
    test_set = data_set[0: 10000]
    dev_set = data_set[10000: 10926]
    train_set = data_set[10926:]
    return embedding, train_set, dev_set, test_set


def fix_data_len(data, max_sen_len, np_dtype=np.int32):
    new_data = [x for x in data]
    len_sen_1 = len(new_data)
    if len_sen_1 > max_sen_len:
        len_sen_1 = max_sen_len
        new_data = new_data[0: len_sen_1]
    else:
        while len(new_data) < max_sen_len:
            new_data.append(0)
    return len_sen_1, np.array(new_data, dtype=np_dtype)


def next_batch(step, batch_size, data_set, total_num, max_sen_len):
    """
    :param step: >= 0
    :param batch_size:
    :param data_set:
    :param total_num:
    :return:
    """
    start_ids = step * batch_size % total_num
    ids = [x % total_num for x in range(start_ids, start_ids + batch_size)]

    sen_1_ids = []
    sen_2_ids = []
    sen_1_lens = []
    sen_2_lens = []
    marker_ids = []
    for idx in ids:
        item = data_set[idx]
        marker_ids.append(item["marker_id"])
        sen_1_id = item["sen_1_ids"]
        sen_2_id = item["sen_2_ids"]
        len_sen_1, sen_1_id = fix_data_len(sen_1_id, max_sen_len, np_dtype=np.int32)
        len_sen_2, sen_2_id = fix_data_len(sen_2_id, max_sen_len, np_dtype=np.int32)

        sen_1_ids.append(sen_1_id)
        sen_2_ids.append(sen_2_id)
        sen_1_lens.append(len_sen_1)
        sen_2_lens.append(len_sen_2)
    batch_data = {
        "marker_ids": np.array(marker_ids, dtype=np.int32),
        "sen_1_ids": np.array(sen_1_ids, dtype=np.int32),
        "sen_2_ids": np.array(sen_2_ids, dtype=np.int32),
        "sen_1_lens": np.array(sen_1_lens, dtype=np.int32),
        "sen_2_lens": np.array(sen_2_lens, dtype=np.int32)
    }
    return batch_data


if __name__ == '__main__':
    data_set_path = "../bookcorpus/all_sentence_pairs.pkl"
    vocab_path = "../bookcorpus/vocab.pkl.gz"
    converted_data_set_path = "../bookcorpus/converted_data_set.pkl.gz"
    emb_path = "../bookcorpus/embedding.pkl.gz"
    # word_indices = vocab.load_word_indices(vocab_path)
    # shuffle_data_set(data_set_path, converted_data_set_path, word_indices)
    load_all_data(emb_path, converted_data_set_path)



