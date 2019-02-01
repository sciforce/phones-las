import argparse
import tensorflow as tf
import os

import utils

from model_helper import las_model_fn


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run model evaluation.')

    parser.add_argument('--data', type=str,
                        help='data in TFRecord format')
    parser.add_argument('--vocab', type=str, required=True,
                        help='vocabulary table, listing vocabulary line by line')
    parser.add_argument('--norm', type=str, default=None,
                        help='normalization params')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='path of saving model')

    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch size')
    parser.add_argument('--num_channels', type=int, default=39,
                        help='number of input channels')
    parser.add_argument('--binf_map', type=str, default='misc/binf_map.csv',
                        help='Path to CSV with phonemes to binary features map')

    return parser.parse_args()


def input_fn(dataset_filename, vocab_filename, norm_filename=None, num_channels=39, batch_size=8, take=0,
    binf2phone=None):
    binary_targets = binf2phone is not None
    labels_shape = [] if not binary_targets else len(binf2phone.index)
    labels_dtype = tf.string if not binary_targets else tf.float32
    dataset = utils.read_dataset(dataset_filename, num_channels, labels_shape=labels_shape,
        labels_dtype=labels_dtype)
    vocab_table = utils.create_vocab_table(vocab_filename)

    if norm_filename is not None:
        means, stds = utils.load_normalization(args.norm)
    else:
        means = stds = None

    sos = binf2phone[utils.SOS].values if binary_targets else utils.SOS
    eos = binf2phone[utils.EOS].values if binary_targets else utils.EOS

    dataset = utils.process_dataset(
        dataset, vocab_table, sos, eos, means, stds, batch_size, 1,
        binary_targets=binary_targets, labels_shape=labels_shape)

    return dataset


def main(args):
    eval_name = str(os.path.basename(args.data).split('.')[0])
    config = tf.estimator.RunConfig(model_dir=args.model_dir)
    hparams = utils.create_hparams(args)

    vocab_list = utils.load_vocab(args.vocab)
    binf2phone_np = None
    binf2phone = None
    if hparams.decoder.binary_outputs:
        binf2phone = utils.load_binf2phone(args.binf_map, vocab_list)
        binf2phone_np = binf2phone.values

    def model_fn(features, labels,
        mode, config, params):
        return las_model_fn(features, labels, mode, config, params,
                            binf2phone=binf2phone_np, run_name=eval_name)

    model = tf.estimator.Estimator(
        model_fn=model_fn,
        config=config,
        params=hparams)

    tf.logging.info('Evaluating on {}'.format(eval_name))
    model.evaluate(lambda: input_fn(
            args.data, args.vocab, args.norm, num_channels=args.num_channels,
            batch_size=args.batch_size, binf2phone=binf2phone), name=eval_name)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    args = parse_args()
    main(args)
