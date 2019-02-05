import argparse
import tensorflow as tf
import pandas as pd

import utils

from model_helper import las_model_fn


def parse_args():
    parser = argparse.ArgumentParser(
        description='Listen, Attend and Spell(LAS) implementation based on Tensorflow. '
                    'The model utilizes input pipeline and estimator API of Tensorflow, '
                    'which makes the training procedure truly end-to-end.')

    parser.add_argument('--train', type=str, required=True,
                        help='training data in TFRecord format')
    parser.add_argument('--valid', type=str,
                        help='validation data in TFRecord format')
    parser.add_argument('--vocab', type=str, required=True,
                        help='vocabulary table, listing vocabulary line by line')
    parser.add_argument('--norm', type=str, default=None,
                        help='normalization params')
    parser.add_argument('--mapping', type=str,
                        help='additional mapping when evaluation')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='path of saving model')
    parser.add_argument('--eval_secs', type=int, default=300,
                        help='evaluation every N seconds, only happening when `valid` is specified')

    parser.add_argument('--encoder_units', type=int, default=128,
                        help='rnn hidden units of encoder')
    parser.add_argument('--encoder_layers', type=int, default=3,
                        help='rnn layers of encoder')
    parser.add_argument('--use_pyramidal', action='store_true',
                        help='whether to use pyramidal rnn')

    parser.add_argument('--decoder_units', type=int, default=128,
                        help='rnn hidden units of decoder')
    parser.add_argument('--decoder_layers', type=int, default=2,
                        help='rnn layers of decoder')
    parser.add_argument('--embedding_size', type=int, default=0,
                        help='embedding size of target vocabulary, if 0, one hot encoding is applied')
    parser.add_argument('--sampling_probability', type=float, default=0.1,
                        help='sampling probabilty of decoder during training')
    parser.add_argument('--attention_type', type=str, default='luong', choices=['luong', 'bahdanau', 'custom',
                            'luong_monotonic', 'bahdanau_monotonic'],
                        help='type of attention mechanism')
    parser.add_argument('--attention_layer_size', type=int,
                        help='size of attention layer, see tensorflow.contrib.seq2seq.AttentionWrapper'
                             'for more details')
    parser.add_argument('--bottom_only', action='store_true',
                        help='apply attention mechanism only at the bottommost rnn cell')
    parser.add_argument('--pass_hidden_state', action='store_true',
                        help='whether to pass encoder state to decoder')

    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch size')
    parser.add_argument('--num_channels', type=int, default=39,
                        help='number of input channels')
    parser.add_argument('--num_epochs', type=int, default=150,
                        help='number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout rate of rnn cell')
    parser.add_argument('--l2_reg_scale', type=float, default=1e-6,
                        help='L2 regularization scale')
    parser.add_argument('--use_text', action='store_true', help='use textual information for encoder regularization')
    parser.add_argument('--emb_loss', action='store_true',
                        help='audio encoder regularization using text encoder')
    parser.add_argument('--text_loss', action='store_true', help='train text encoder')
    parser.add_argument('--binary_outputs', action='store_true',
                        help='make projection layer output binary feature posteriors instead of phone posteriors')
    parser.add_argument('--output_ipa', action='store_true',
                        help='With --binary_outputs on, make the graph output phones and'
                             ' change sampling algorithm at training')
    parser.add_argument('--binf_map', type=str, default='misc/binf_map.csv',
                        help='Path to CSV with phonemes to binary features map')
    parser.add_argument('--reset', help='Reset HParams.', action='store_true')
    parser.add_argument('--binf_sampling', action='store_true',
                        help='with --output_ipa, do not use ipa sampling algorithm for trainin, only for validation')

    return parser.parse_args()


def input_fn(dataset_filename, vocab_filename, norm_filename=None, num_channels=39, batch_size=8, num_epochs=1,
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
        dataset, vocab_table, sos, eos, means, stds, batch_size, num_epochs,
        binary_targets=binary_targets, labels_shape=labels_shape)

    return dataset


def main(args):
    vocab_list = utils.load_vocab(args.vocab)
    binf2phone_np = None
    binf2phone = None
    if not args.binary_outputs:
        vocab_size = len(vocab_list)
    else:
        binf2phone = utils.load_binf2phone(args.binf_map, vocab_list)
        vocab_size = len(binf2phone.index)
        if args.output_ipa:
            binf2phone_np = binf2phone.values

    config = tf.estimator.RunConfig(model_dir=args.model_dir)
    hparams = utils.create_hparams(
        args, vocab_size, utils.SOS_ID, utils.EOS_ID)

    def model_fn(features, labels,
        mode, config, params):
        binf_map = binf2phone_np
        if tf.estimator.ModeKeys.TRAIN == mode and args.binf_sampling:
            binf_map = None
        return las_model_fn(features, labels, mode, config, params,
            binf2phone=binf_map)

    model = tf.estimator.Estimator(
        model_fn=model_fn,
        config=config,
        params=hparams)

    if args.valid:
        train_spec = tf.estimator.TrainSpec(
            input_fn=lambda: input_fn(
                args.train, args.vocab, args.norm, num_channels=args.num_channels, batch_size=args.batch_size,
                num_epochs=args.num_epochs, binf2phone=binf2phone))

        eval_spec = tf.estimator.EvalSpec(
            input_fn=lambda: input_fn(
                args.valid or args.train, args.vocab, args.norm, num_channels=args.num_channels,
                batch_size=args.batch_size, binf2phone=binf2phone),
            start_delay_secs=60,
            throttle_secs=args.eval_secs)

        tf.estimator.train_and_evaluate(model, train_spec, eval_spec)
    else:
        model.train(
            input_fn=lambda: input_fn(
                args.train, args.vocab, args.norm, num_channels=args.num_channels, batch_size=args.batch_size,
                num_epochs=args.num_epochs, binf2phone=binf2phone))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    args = parse_args()
    main(args)

