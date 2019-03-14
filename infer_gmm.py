import argparse
import numpy as np
import tensorflow as tf
from joblib import load
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.dataset_utils import read_dataset
from preprocess_all import make_example


def input_fn(dataset, means, stds):
    sess = tf.Session()
    if means is not None:
        dataset = dataset.map(
            lambda inputs, labels: ((inputs - means) / stds, labels), num_parallel_calls=8)
    data_op = dataset.make_one_shot_iterator().get_next()
    while True:
        try:
            yield sess.run(data_op)
        except tf.errors.OutOfRangeError:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--norm', type=str, default=None)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--gmm_path', type=str, default='gmm')
    parser.add_argument('--num_channels', type=int, default=42)
    parser.add_argument('--num_mixtures', type=int, default=128)
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()
    if args.norm is not None:
        means, stds = load(args.norm)
    else:
        means = stds = None
    dataset = read_dataset(args.input, args.num_channels)
    data_in = tf.placeholder(tf.float32, [None, args.num_channels], 'features')
    loss, scores, assignments, train_op, init_op, _ = tf.contrib.factorization.gmm(
        data_in, 'random', args.num_mixtures, 0)
    saver = tf.train.Saver(max_to_keep=1)
    sess = tf.Session()
    saver.restore(sess, tf.train.latest_checkpoint(args.gmm_path))
    writer = tf.io.TFRecordWriter(args.output) if args.output is not None else None
    for features, label in tqdm(input_fn(dataset, means, stds), unit='file'):
        label = [x.decode('utf-8') for x in label]
        scores_val, assignments_val = sess.run([scores, assignments], {data_in: features})
        diagram = np.zeros((features.shape[0], args.num_mixtures))
        for i, j in enumerate(assignments_val[0][0]):
            diagram[i, j] = 1
        if writer is not None:
            writer.write(make_example(diagram, label).SerializeToString())
        if args.show:
            plt.subplot(311)
            plt.plot(scores_val)
            plt.title(' '.join(label))
            plt.axis([0, diagram.shape[0], scores_val.min(), scores_val.max()])
            plt.subplot(312)
            plt.imshow(diagram.T, aspect='auto')
            plt.subplot(313)
            plt.imshow(features[:, ::3].T, aspect='auto')
            plt.show()
    if writer is not None:
        writer.close()
