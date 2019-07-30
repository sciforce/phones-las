import tensorflow as tf
import numpy as np

__all__ = [
    'read_dataset',
    'process_dataset',
]


def read_dataset(filename, num_channels=39, labels_shape=None, labels_dtype=tf.string):
    """Read data from tfrecord file."""
    if labels_shape is None:
        labels_shape = []

    def parse_fn(example_proto):
        """Parse function for reading single sequence example."""
        sequence_features = {
            'inputs': tf.io.FixedLenSequenceFeature(shape=[num_channels], dtype=tf.float32),
            'labels': tf.io.FixedLenSequenceFeature(labels_shape, labels_dtype)
        }

        context, sequence = tf.io.parse_single_sequence_example(
            serialized=example_proto,
            sequence_features=sequence_features
        )

        return sequence['inputs'], sequence['labels']
    # Multi-file training support.
    if filename.endswith('.txt'):
        filename = [x.strip() for x in open(filename, 'r').readlines()]
    dataset = tf.data.TFRecordDataset(filename, num_parallel_reads=4)
    dataset = dataset.map(parse_fn)
    dataset = dataset.cache()

    return dataset


def process_dataset(dataset, vocab_table, sos, eos, means=None, stds=None,
    batch_size=8, num_epochs=1, num_parallel_calls=32, is_infer=False,
    binary_targets=False, max_frames=-1, max_symbols=-1):

    eos_id = None
    try:
        use_labels = len(tf.compat.v1.data.get_output_shapes(dataset)) == 2
    except TypeError:
        use_labels = False

    if not use_labels:
        if means is not None and stds is not None:
            tf.compat.v1.logging.info('Applying normalization.')
            means_const = tf.constant(means, dtype=tf.float32)
            stds_const = tf.constant(stds, dtype=tf.float32)
            dataset = dataset.map(
                lambda inputs: (inputs - means_const) / stds_const,
                num_parallel_calls=num_parallel_calls)
        dataset = dataset.map(lambda inputs: {
            'encoder_inputs': tf.cast(inputs, tf.float32),
            'source_sequence_length': tf.shape(inputs)[0]
        })
        dataset = dataset.padded_batch(
            batch_size,
            padded_shapes=tf.compat.v1.data.get_output_shapes(dataset),
            padding_values={
                    'encoder_inputs': 0.0,
                    'source_sequence_length': 0,
                },
            drop_remainder=True)
    else:
        output_buffer_size = batch_size * 1000

        if max_frames > 0:
            dataset = dataset.filter(
                lambda inputs, labels: tf.logical_and(tf.shape(inputs)[0] <= max_frames,
                                                      tf.shape(labels)[0] <= max_symbols))

        if not binary_targets:
            sos_id = tf.cast(vocab_table.lookup(tf.constant(sos)), tf.int32)
            eos_id = tf.cast(vocab_table.lookup(tf.constant(eos)), tf.int32)

        dataset = dataset.repeat(num_epochs if num_epochs > 0 else None)

        if not is_infer:
            dataset = dataset.shuffle(output_buffer_size)

        if not binary_targets:
            dataset = dataset.map(
                lambda inputs, labels: (inputs,
                                        vocab_table.lookup(labels)),
                num_parallel_calls=num_parallel_calls)

        if means is not None and stds is not None:
            tf.compat.v1.logging.info('Applying normalization.')
            means_const = tf.constant(means, dtype=tf.float32)
            stds_const = tf.constant(stds, dtype=tf.float32)
            dataset = dataset.map(
                lambda inputs, labels: ((inputs - means_const) / stds_const,
                                        labels),
                num_parallel_calls=num_parallel_calls)

        dataset = dataset.map(
            lambda inputs, labels: (tf.cast(inputs, tf.float32),
                                    tf.cast(labels, tf.int32)),
            num_parallel_calls=num_parallel_calls)

        if binary_targets:
            sos_id = sos[np.newaxis, :]
            eos_id = eos[np.newaxis, :]

            dataset = dataset.map(
                lambda inputs, labels: (inputs,
                                        tf.concat((sos_id, labels), 0),
                                        tf.concat((labels, eos_id), 0),
                                        ),
                num_parallel_calls=num_parallel_calls)
        else:
            dataset = dataset.map(
                lambda inputs, labels: (inputs,
                                        tf.concat(([sos_id], labels), 0),
                                        tf.concat((labels, [eos_id]), 0)),
                num_parallel_calls=num_parallel_calls)

        dataset = dataset.map(
            lambda inputs, labels_in, labels_out: (inputs,
                                                   labels_in,
                                                   labels_out,
                                                   tf.shape(inputs)[0],
                                                   tf.shape(labels_in)[0] if binary_targets else tf.size(labels_in)),
            num_parallel_calls=num_parallel_calls)

        dataset = dataset.map(
            lambda inputs, labels_in, labels_out,
                   source_sequence_length, target_sequence_length: (
                {
                    'encoder_inputs': inputs,
                    'source_sequence_length': source_sequence_length,
                },
                {
                    'targets_inputs': labels_in,
                    'targets_outputs': labels_out,
                    'target_sequence_length': target_sequence_length
                }))

        out_shapes = tf.compat.v1.data.get_output_shapes(dataset)
        if max_frames > 0:
            padded_shapes = (
                {
                    'encoder_inputs': tf.TensorShape([max_frames, out_shapes[0]['encoder_inputs'][1].value]),
                    'source_sequence_length': out_shapes[0]['source_sequence_length']
                },
                {
                    'targets_inputs': tf.TensorShape([max_symbols]),
                    'targets_outputs': tf.TensorShape([max_symbols]),
                    'target_sequence_length': out_shapes[1]['target_sequence_length'],
                }
            )
        else:
            padded_shapes = out_shapes
        dataset = dataset.padded_batch(
            batch_size, padded_shapes=padded_shapes,
            padding_values=(
                {
                    'encoder_inputs': 0.0,
                    'source_sequence_length': 0,
                },
                {
                    'targets_inputs': 0 if binary_targets else eos_id,
                    'targets_outputs': 0 if binary_targets else eos_id,
                    'target_sequence_length': 0,
                }),
            drop_remainder=True)

    return dataset
