import tensorflow as tf
import tensorflow.contrib as tf_contrib
import numpy as np
from tensor2tensor.data_generators.speech_recognition import SpeechRecognitionProblem
from tensor2tensor.models.transformer import transformer_librispeech_tpu
import utils

__all__ = [
    'input_fn',
    'input_fn_t2t'
]

def read_dataset_t2t_format(data_dir, num_parallel_calls, mode, max_frames, max_symbols):
    problem = SpeechRecognitionProblem()
    problem.name = 'asr_ipa_precalc'
    speech_params = transformer_librispeech_tpu()
    speech_params.max_input_seq_length = max_frames
    speech_params.max_target_seq_length = max_symbols
    dataset = problem.dataset(mode, data_dir=data_dir, num_threads=num_parallel_calls, hparams=speech_params)
    return dataset

def process_dataset_t2t_format(dataset, sos_id, eos_id,
    batch_size=8, num_epochs=1, num_parallel_calls=32, is_infer=False,
    max_frames=-1, max_symbols=-1):
    output_buffer_size = batch_size * 1000

    channels_count = dataset.output_shapes['inputs'][1] * dataset.output_shapes['inputs'][2]

    dataset = dataset.map(
        lambda inputs: (
            tf.reshape(inputs['inputs'], (tf.shape(inputs['inputs'])[0], channels_count)),
            inputs['targets']),
        num_parallel_calls=num_parallel_calls
    )

    if max_frames > 0:
        dataset = dataset.filter(
            lambda inputs, labels: tf.logical_and(tf.shape(inputs)[0] <= max_frames,
                                                    tf.shape(labels)[0] <= max_symbols))

    dataset = dataset.repeat(num_epochs if num_epochs > 0 else None)

    if not is_infer:
        dataset = dataset.shuffle(output_buffer_size)

    dataset = dataset.map(
        lambda inputs, labels: (tf.cast(inputs, tf.float32),
                                tf.cast(labels, tf.int32)),
        num_parallel_calls=num_parallel_calls)

    dataset = dataset.map(
        lambda inputs, labels: (inputs,
                                tf.concat(([sos_id], labels[:-1]), 0),
                                labels),
        num_parallel_calls=num_parallel_calls)

    dataset = dataset.map(
        lambda inputs, labels_in, labels_out: (inputs,
                                            labels_in,
                                            labels_out,
                                            tf.shape(inputs)[0],
                                            tf.size(labels_in)),
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

    if max_frames > 0:
        padded_shapes = (
            {
                'encoder_inputs': tf.TensorShape([max_frames, dataset.output_shapes[0]['encoder_inputs'][1].value]),
                'source_sequence_length': dataset.output_shapes[0]['source_sequence_length']
            },
            {
                'targets_inputs': tf.TensorShape([max_symbols]),
                'targets_outputs': tf.TensorShape([max_symbols]),
                'target_sequence_length': dataset.output_shapes[1]['target_sequence_length'],
            }
        )
    else:
        padded_shapes = dataset.output_shapes
    dataset = dataset.padded_batch(
        batch_size,
        padded_shapes=padded_shapes,
        padding_values=(
            {
                'encoder_inputs': 0.0,
                'source_sequence_length': 0,
            },
            {
                'targets_inputs': eos_id,
                'targets_outputs': eos_id,
                'target_sequence_length': 0,
            }),
        drop_remainder=True)

    return dataset

def input_fn_t2t(data_dir, mode, hparams, batch_size=8, num_epochs=1,
    num_parallel_calls=32, max_frames=-1, max_symbols=-1, take=0):
    dataset = read_dataset_t2t_format(data_dir, num_parallel_calls, mode, max_frames, max_symbols)

    dataset = process_dataset_t2t_format(
        dataset, hparams.decoder.sos_id, hparams.decoder.eos_id, batch_size, num_epochs,
        num_parallel_calls=num_parallel_calls,
        max_frames=max_frames, max_symbols=max_symbols
    )

    if take > 0:
        dataset = dataset.take(take)

    return dataset

def read_dataset(filename, num_channels=39):
    """Read data from tfrecord file."""

    def parse_fn(example_proto):
        """Parse function for reading single sequence example."""
        sequence_features = {
            'inputs': tf.FixedLenSequenceFeature(shape=[num_channels], dtype=tf.float32),
            'labels': tf.FixedLenSequenceFeature([], tf.string)
        }

        context, sequence = tf.parse_single_sequence_example(
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
    max_frames=-1, max_symbols=-1):

    try:
        use_labels = len(dataset.output_classes) == 2
    except TypeError:
        use_labels = False

    if not use_labels:
        if means is not None and stds is not None:
            tf.logging.info('Applying normalization.')
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
            padded_shapes=dataset.output_shapes,
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

        sos_id = tf.cast(vocab_table.lookup(tf.constant(sos)), tf.int32)
        eos_id = tf.cast(vocab_table.lookup(tf.constant(eos)), tf.int32)

        dataset = dataset.repeat(num_epochs if num_epochs > 0 else None)

        if not is_infer:
            dataset = dataset.shuffle(output_buffer_size)

        dataset = dataset.map(
            lambda inputs, labels: (inputs,
                                    vocab_table.lookup(labels)),
            num_parallel_calls=num_parallel_calls)

        if means is not None and stds is not None:
            tf.logging.info('Applying normalization.')
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
                                                tf.size(labels_in)),
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

        if max_frames > 0:
            padded_shapes = (
                {
                    'encoder_inputs': tf.TensorShape([max_frames, dataset.output_shapes[0]['encoder_inputs'][1].value]),
                    'source_sequence_length': dataset.output_shapes[0]['source_sequence_length']
                },
                {
                    'targets_inputs': tf.TensorShape([max_symbols]),
                    'targets_outputs': tf.TensorShape([max_symbols]),
                    'target_sequence_length': dataset.output_shapes[1]['target_sequence_length'],
                }
            )
        else:
            padded_shapes = dataset.output_shapes
        dataset = dataset.padded_batch(
            batch_size,
            padded_shapes=padded_shapes,
            padding_values=(
                {
                    'encoder_inputs': 0.0,
                    'source_sequence_length': 0,
                },
                {
                    'targets_inputs': eos_id,
                    'targets_outputs': eos_id,
                    'target_sequence_length': 0,
                }),
            drop_remainder=True)

    return dataset

def input_fn(dataset_filename, vocab_filename, norm_filename=None, num_channels=39, batch_size=8, num_epochs=1,
    num_parallel_calls=32, max_frames=-1, max_symbols=-1, take=0, is_infer=False):
    dataset = read_dataset(dataset_filename, num_channels)
    vocab_table = utils.create_vocab_table(vocab_filename)

    if norm_filename is not None:
        means, stds = utils.load_normalization(norm_filename)
    else:
        means = stds = None

    sos = utils.SOS
    eos = utils.EOS

    dataset = process_dataset(
        dataset, vocab_table, sos, eos, means, stds, batch_size, num_epochs,
        num_parallel_calls=num_parallel_calls,
        max_frames=max_frames, max_symbols=max_symbols
    )

    if take > 0:
        dataset = dataset.take(take)

    return dataset