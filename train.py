import argparse
import tensorflow as tf

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
    parser.add_argument('--unidirectional', action='store_true',
                        help='Use unidirectional RNN')

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
    parser.add_argument('--num_parallel_calls', type=int, default=32,
                        help='Number of elements to be processed in parallel during the dataset transformation')
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
    parser.add_argument('--add_noise', type=int, default=0,
                        help='How often (in steps) to add Gaussian noise to the weights, zero for disabling noise addition.')
    parser.add_argument('--noise_std', type=float, default=0.1,
                        help='Weigth noise standard deviation.')
    parser.add_argument('--binary_outputs', action='store_true',
                        help='make projection layer output binary feature posteriors instead of phone posteriors')
    parser.add_argument('--output_ipa', action='store_true',
                        help='With --binary_outputs on, make the graph output phones and'
                             ' change sampling algorithm at training')
    parser.add_argument('--binf_map', type=str, default='misc/binf_map.csv',
                        help='Path to CSV with phonemes to binary features map')
    parser.add_argument('--ctc_weight', type=float, default=-1.,
                        help='If possitive, adds CTC mutlitask target based on encoder.')
    parser.add_argument('--reset', help='Reset HParams.', action='store_true')
    parser.add_argument('--binf_sampling', action='store_true',
                        help='with --output_ipa, do not use ipa sampling algorithm for trainin, only for validation')
    parser.add_argument('--binf_projection', action='store_true',
                        help='with --binary_outputs and --output_ipa, use binary features mapping instead of decoder''s projection layer.')
    parser.add_argument('--multitask', action='store_true',
                        help='with --binary_outputs use both binary features and IPA decoders.')
    parser.add_argument('--tpu_name', type=str, default='', help='TPU name. Leave blank to prevent TPU training.')

    return parser.parse_args()


def input_fn(dataset_filename, vocab_filename, norm_filename=None, num_channels=39, batch_size=8, num_epochs=1,
    binf2phone=None, num_parallel_calls=32):
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
        binary_targets=binary_targets, labels_shape=labels_shape, num_parallel_calls=num_parallel_calls)

    return dataset


def main(args):
    vocab_list = utils.load_vocab(args.vocab)
    binf2phone_np = None
    mapping = None
    vocab_size = len(vocab_list)
    binf_count = None
    if args.binary_outputs:
        if args.mapping is not None:
            vocab_list, mapping = utils.get_mapping(args.mapping, args.vocab)
            args.mapping = None
        binf2phone = utils.load_binf2phone(args.binf_map, vocab_list)
        binf_count = len(binf2phone.index)
        if args.output_ipa:
            binf2phone_np = binf2phone.values

    if args.tpu_name:
        iterations_per_loop = 100
        tpu_cluster_resolver = None
        if args.tpu_name != 'fake':
            tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(args.tpu_name)
        config = tf.estimator.tpu.RunConfig(
            cluster=tpu_cluster_resolver,
            model_dir=args.model_dir,
            save_checkpoints_steps=max(600, iterations_per_loop),
            tpu_config=tf.estimator.tpu.TPUConfig(
                iterations_per_loop=iterations_per_loop,
                per_host_input_for_training=tf.estimator.tpu.InputPipelineConfig.PER_HOST_V2))
    else:
        config = tf.estimator.RunConfig(model_dir=args.model_dir)
    hparams = utils.create_hparams(
        args, vocab_size, binf_count, utils.SOS_ID, utils.EOS_ID)
    if mapping is not None:
        hparams.del_hparam('mapping')
        hparams.add_hparam('mapping', mapping)

    def model_fn(features, labels,
        mode, config, params):
        binf_map = binf2phone_np
        return las_model_fn(features, labels, mode, config, params,
            binf2phone=binf_map)

    if args.tpu_name:
        model = tf.estimator.tpu.TPUEstimator(
            model_fn=model_fn, config=config, params=hparams, eval_on_tpu=False,
            train_batch_size=args.batch_size, use_tpu=args.tpu_name != 'fake'
        )
    else:
        model = tf.estimator.Estimator(
            model_fn=model_fn,
            config=config,
            params=hparams)

    if args.valid:
        train_spec = tf.estimator.TrainSpec(
            input_fn=lambda params: input_fn(
                args.train, args.vocab, args.norm, num_channels=args.num_channels,
                batch_size=params.batch_size,
                num_epochs=args.num_epochs, binf2phone=None, num_parallel_calls=args.num_parallel_calls))

        eval_spec = tf.estimator.EvalSpec(
            input_fn=lambda params: input_fn(
                args.valid or args.train, args.vocab, args.norm, num_channels=args.num_channels,
                batch_size=params.batch_size, binf2phone=None,
                num_parallel_calls=args.num_parallel_calls),
            start_delay_secs=60,
            throttle_secs=args.eval_secs)

        tf.estimator.train_and_evaluate(model, train_spec, eval_spec)
    else:
        tf.logging.warning('Training without evaluation!')
        model.train(
            input_fn=lambda params: input_fn(
                args.train, args.vocab, args.norm, num_channels=args.num_channels,
                batch_size=params.batch_size,
                num_epochs=args.num_epochs, binf2phone=None, num_parallel_calls=args.num_parallel_calls),
            steps=args.num_epochs * 1000 * hparams.batch_size
        )


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    args = parse_args()
    main(args)

