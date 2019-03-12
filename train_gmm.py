import numpy as np
import tensorflow as tf
from joblib import load
from tqdm import tqdm
from collections import Counter

from utils.dataset_utils import read_dataset


def validate(loss, dataset, means, stds, sess, data_in):
    loss_avg = 0
    total = 0
    for d in input_fn(dataset, means, stds):
        loss_val = sess.run(loss, {data_in: d})
        loss_avg += loss_val
        total += 1
    scores_val, assignments_val = sess.run([scores, assignments], {data_in: d})
    print(f'Scores: {scores_val.min()} - {scores_val.max()}')
    counts = Counter(assignments_val[0][0])
    print(f'Top classes: {", ".join([f"{x[0]}: {x[1]}" for x in counts.most_common(5)])}')
    return loss_avg / total


def input_fn(dataset, means, stds):
    sess = tf.Session()
    dataset = dataset.map(
        lambda inputs, labels: (inputs - means) / (stds + 1e-5), num_parallel_calls=8)
    data_op = dataset.shuffle(100).make_one_shot_iterator().get_next()
    while True:
        try:
            yield sess.run(data_op)
        except tf.errors.OutOfRangeError:
            break


if __name__ == '__main__':
    num_channels = 42
    num_mixtures = 128
    dataset = read_dataset('timit/train.record', num_channels)
    means, stds = load('timit/norm.dmp')
    val_dataset = read_dataset('timit/val.record', num_channels)
    data_in = tf.placeholder(tf.float32, [None, num_channels], 'features')
    loss, scores, assignments, train_op, init_op, _ = tf.contrib.factorization.gmm(data_in, 'random', num_mixtures, 0)
    saver = tf.train.Saver(max_to_keep=3)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    initial_chunk = []
    for d in input_fn(dataset, means, stds):
        initial_chunk.append(d[::10, :])
    initial_chunk = np.concatenate(initial_chunk, axis=0)
    sess.run(init_op, {data_in: initial_chunk})
    print('Training loop')
    handle = tqdm()
    best_loss = loss_avg = validate(loss, val_dataset, means, stds, sess, data_in)
    handle.write(f'Initial: {loss_avg:2.2f}')
    for i in range(50):
        _, loss_val = sess.run([train_op, loss], {data_in: initial_chunk})
        handle.set_description(f'Loss: {loss_val:2.3f}')
        handle.update()
        loss_avg = validate(loss, val_dataset, means, stds, sess, data_in)
        handle.write(f'Epoch {i}: {loss_avg:2.2f}')
        if loss_avg > best_loss:
            best_loss = loss_avg
            saver.save(sess, 'gmm/model', i)
            handle.write('Best model saved!')
