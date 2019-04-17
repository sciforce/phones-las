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
    with tf.Session() as sess:
        handle = tqdm(None, unit='phrase')
        while True:
            try:
                _, label = sess.run(read_op)
                for x in label:
                    vocab.add(x.decode('utf-8'))
                handle.update()
            except tf.errors.OutOfRangeError:
                break
    handle.close()
    if args.output is None:
        print('\n'.join(x for x in vocab))
    else:
        with open(args.output, 'w') as f:
            f.write('\n'.join(x for x in vocab))
