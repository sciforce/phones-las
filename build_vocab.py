import argparse
import tensorflow as tf
from tqdm import tqdm

from utils.dataset_utils import read_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--num_channels', type=int, required=True)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()
    vocab = set()
    dataset = read_dataset(args.data, args.num_channels)
    read_op = dataset.make_one_shot_iterator().get_next()
    max_frames = max_symbols = 0
    with tf.Session() as sess:
        handle = tqdm(None, unit='phrase')
        while True:
            try:
                features, label = sess.run(read_op)
                max_frames = max([max_frames, features.shape[0]])
                max_symbols = max([max_symbols, len(label)])
                for x in label:
                    vocab.add(x.decode('utf-8'))
                handle.update()
            except tf.errors.OutOfRangeError:
                break
    handle.close()
    print('Max frames: {}, max symbols: {}'.format(max_frames, max_symbols))
    if args.output is None:
        print('\n'.join(x for x in vocab))
    else:
        with open(args.output, 'w') as f:
            f.write('\n'.join(x for x in vocab))
