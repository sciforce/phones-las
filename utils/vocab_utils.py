import tensorflow as tf
import pickle

__all__ = [
    'create_vocab_table',
    'load_vocab',
    'UNK',
    'SOS',
    'EOS',
    'UNK_ID',
    'SOS_ID',
    'EOS_ID',
]


UNK = '<unk>'
SOS = '<s>'
EOS = '</s>'
UNK_ID = 0
SOS_ID = 1
EOS_ID = 2


def load_vocab(filename):
    if not '.pickle' in filename:
        with open(filename, 'r') as f:
            vocab_list = [vocab.strip('\r\n') for vocab in f]
            vocab_list = [UNK, SOS, EOS] + vocab_list
    else:
        with open(filename, 'rb') as f:
            vocab_list = pickle.load(f)
            vocab_list = [UNK, SOS, EOS] + vocab_list

    return vocab_list


def create_vocab_table(filename):
    vocab_list = load_vocab(filename)

    return tf.contrib.lookup.index_table_from_tensor(
        tf.constant(vocab_list), num_oov_buckets=0, default_value=UNK_ID)
