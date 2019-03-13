import numpy as np
import tensorflow as tf
from joblib import load
import matplotlib.pyplot as plt

from utils.dataset_utils import read_dataset


def input_fn(dataset, means, stds):
    sess = tf.Session()
    dataset = dataset.map(
        lambda inputs, labels: ((inputs - means) / stds, labels), num_parallel_calls=8)
    data_op = dataset.shuffle(100).make_one_shot_iterator().get_next()
    while True:
        try:
            yield sess.run(data_op)
        except tf.errors.OutOfRangeError:
            break


if __name__ == '__main__':
    num_channels = 42
    num_mixtures = 128
    means, stds = load('timit/norm.dmp')
    dataset = read_dataset('timit/train.record', num_channels)
    data_in = tf.placeholder(tf.float32, [None, num_channels], 'features')
    loss, scores, assignments, train_op, init_op, _ = tf.contrib.factorization.gmm(data_in, 'random', num_mixtures, 0)
    saver = tf.train.Saver(max_to_keep=1)
    sess = tf.Session()
    saver.restore(sess, tf.train.latest_checkpoint('gmm'))
    for d, l in input_fn(dataset, means, stds):
        scores_val, assignments_val = sess.run([scores, assignments], {data_in: d})
        plt.subplot(311)
        plt.plot(scores_val)
        plt.title(' '.join([x.decode('utf-8') for x in l]))
        diagram = np.zeros((d.shape[0], num_mixtures))
        for i, j in enumerate(assignments_val[0][0]):
            diagram[i, j] = 1
        plt.axis([0, diagram.shape[0], scores_val.min(), scores_val.max()])
        plt.subplot(312)
        plt.imshow(diagram.T, aspect='auto')
        plt.subplot(313)
        plt.imshow(d[:, ::3].T, aspect='auto')
        plt.show()
