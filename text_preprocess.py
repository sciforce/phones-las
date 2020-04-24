from argparse import ArgumentParser
import os
import re
from tqdm import tqdm
import tensorflow as tf
import numpy as np

from preprocess_all import make_example


def process_line(args, line):
    try:
        text, transcript = line.split(args.delimiter)
        inputs = {
            'text': text,
            'transcript': transcript.strip().split()
        }
        text = ' '.join(text.strip().split())
        for p in ',.:;?!-_':
            text = text.replace(p, '')
        inputs['text'] = list(text.lower())
        if args.strip_stress:
            inputs['transcript'] = list(map(lambda x: re.sub(r'\d', '', x), inputs['transcript']))
    except ValueError:
        print(f'Problem with line "{line}"')
        return None
    return inputs


def encode_text(vocab, inputs):
    features = np.zeros([len(inputs['text']), len(vocab)], dtype=np.float32)
    for i, elem in enumerate(inputs['text']):
        ind = vocab.index(elem)
        features[i, ind] = 1
    inputs['features'] = features
    return inputs


if __name__ == "__main__":
    parser = ArgumentParser(description='Runs one-hot encoding for lexicon dataset files.')
    parser.add_argument('--input_file', help='File with audio paths and texts.', required=True)
    parser.add_argument('--output_file', help='Target TFRecord file name.', required=True)
    parser.add_argument('--delimiter', help='CSV delimiter', type=str, default=',')
    parser.add_argument('--strip_stress', help='Removes stress markers from phones', action='store_true')
    args = parser.parse_args()

    output_dir = os.path.dirname(args.output_file)
    in_vocab_path = os.path.join(output_dir, 'in_vocab.txt')
    vocab_path = os.path.join(output_dir, 'vocab.txt')

    print('Processing text dataset from file {}.'.format(args.input_file))
    lines = open(args.input_file, 'r').readlines()
    lines = [process_line(args, line) for line in tqdm(lines, unit='word', desc='Reading')]
    lines = [x for x in lines if x is not None]
    if os.path.exists(in_vocab_path) and os.path.exists(vocab_path):
        print('Found vocabs, not collecting new.')
        in_vocab = [x.strip() for x in open(in_vocab_path, 'r')]
        out_vocab = [x.strip() for x in open(vocab_path, 'r')]
    else:
        in_vocab = set()
        out_vocab = set()
        for line in lines:
            in_vocab.update(line['text'])
            out_vocab.update(line['transcript'])
        in_vocab = sorted(in_vocab)
        out_vocab = sorted(out_vocab)
        with open(in_vocab_path, 'w') as f:
            for el in in_vocab:
                f.write(f'{el}\n')
        with open(vocab_path, 'w') as f:
            for el in out_vocab:
                f.write(f'{el}\n')
    lines = [encode_text(in_vocab, line) for line in tqdm(lines, unit='word', desc='Encoding')]
    with tf.io.TFRecordWriter(args.output_file) as writer:
        for x in tqdm(lines, unit='word', desc='Writing'):
            writer.write(make_example(x['features'], x['transcript']).SerializeToString())
