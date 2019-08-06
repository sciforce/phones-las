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
    parser.add_argument('--vocab', type=str,
                        help='vocabulary table, listing vocabulary line by line')
    parser.add_argument('--norm', type=str, default=None,
                        help='normalization params')
    parser.add_argument('--t2t_format', action='store_true',
                        help='Use dataset in the format of ASR problems of Tensor2Tensor framework. --train param should be directory')
    parser.add_argument('--mapping', type=str,
                        help='additional mapping when evaluation')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='path of saving model')

    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch size')
    parser.add_argument('--num_channels', type=int, default=39,
                        help='number of input channels')
    parser.add_argument('--binf_map', type=str, default='misc/binf_map.csv',
                        help='Path to CSV with phonemes to binary features map')

    return parser.parse_args()

def main(args):
    eval_name = str(os.path.basename(args.data).split('.')[0])
    config = tf.estimator.RunConfig(model_dir=args.model_dir)
    hparams = utils.create_hparams(args)

    vocab_name = args.vocab if not args.t2t_format else os.path.join(args.data, 'vocab.txt')
    vocab_list = utils.load_vocab(vocab_name)
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
    if args.t2t_format:
        input_fn = lambda: utils.input_fn_t2t(
            args.data, tf.estimator.ModeKeys.EVAL, hparams,
            batch_size=args.batch_size)
    else:
        input_fn = lambda: utils.input_fn(
            args.data, args.vocab, args.norm, num_channels=args.num_channels,
            batch_size=args.batch_size)
    model.evaluate(input_fn, name=eval_name)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    args = parse_args()
    main(args)
