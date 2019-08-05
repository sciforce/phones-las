import argparse
import tensorflow as tf
from tqdm import tqdm

from utils.dataset_utils import read_dataset, read_dataset_t2t_format


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--num_channels', type=int, required=True)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--t2t_format', action='store_true')
    args = parser.parse_args()
    vocab = set()
    if args.t2t_format:
        dataset = read_dataset_t2t_format(args.data, 1, tf.estimator.ModeKeys.TRAIN, -1, -1)
    else:
        dataset = read_dataset(args.data, args.num_channels)
    read_op = dataset.make_one_shot_iterator().get_next()
    max_frames = max_symbols = 0
    with tf.Session() as sess:
        handle = tqdm(None, unit='phrase')
        while True:
            try:
                if args.t2t_format:
                    item = sess.run(read_op)
                    features, labels = item['inputs'], item['targets']
                    max_symbols = max([max_symbols, labels.shape[0]])
                else:
                    features, label = sess.run(read_op)
                    max_symbols = max([max_symbols, len(label)])
                max_frames = max([max_frames, features.shape[0]])
                if not args.t2t_format:
                    for x in label:
                        vocab.add(x.decode('utf-8'))
                handle.update()
            except tf.errors.OutOfRangeError:
                break
    handle.close()
    print('Max frames: {}, max symbols: {}'.format(max_frames, max_symbols))
    if not args.t2t_format:
        if args.output is None:
            print('\n'.join(x for x in vocab))
        else:
            with open(args.output, 'w') as f:
                f.write('\n'.join(x for x in vocab))
