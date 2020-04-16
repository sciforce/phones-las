import argparse
import tensorflow as tf
import librosa
from joblib import dump

import utils
from model_helper import las_model_fn
from preprocess_all import calculate_acoustic_features

SAMPLE_RATE = 16000


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--waveform', type=str, required=True,
                        help='Acoustic file')
    parser.add_argument('--vocab', type=str, required=True,
                        help='vocabulary table, listing vocabulary line by line')
    parser.add_argument('--norm', type=str, default=None,
                        help='normalization params')
    parser.add_argument('--mapping', type=str,
                        help='additional mapping when evaluation')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='path of imported model')

    parser.add_argument('--beam_width', type=int, default=0,
                        help='number of beams (default 0: using greedy decoding)')

    parser.add_argument('--delimiter', help='Symbols delimiter. Default: " "', type=str, default=' ')
    parser.add_argument('--binf_map', type=str, default='misc/binf_map.csv',
                        help='Path to CSV with phonemes to binary features map')
    parser.add_argument('--use_phones_from_binf', help='User phonemes decoded from binary features decoder outputs.',
                        action='store_true')
    parser.add_argument('--feature_type', help='Acoustic feature type.', type=str,
                        choices=['mfe', 'mfcc', 'lyon'], default='mfcc')
    parser.add_argument('--backend', help='Library for calculating acoustic features.', type=str,
                        choices=['speechpy', 'librosa'], default='librosa')
    parser.add_argument('--n_mfcc', help='Number of MFCC coeffs.', type=int, default=13)
    parser.add_argument('--n_mels', help='Number of mel-filters.', type=int, default=40)
    parser.add_argument('--window', help='Analysis window length in ms.', type=int, default=20)
    parser.add_argument('--step', help='Analysis window step in ms.', type=int, default=10)
    parser.add_argument('--deltas', help='Calculate deltas and double-deltas.', action='store_true')
    parser.add_argument('--energy', help='Compute energy.', action='store_true')
    parser.add_argument('--output_file', help='location for saving predictions')

    return parser.parse_args()


def input_fn(features, vocab_filename, norm_filename=None):
    def gen():
        for item in features:
            yield item

    output_shapes = tf.TensorShape([None, features[0].shape[-1]])
    dataset = tf.data.Dataset.from_generator(gen, tf.float32, output_shapes)
    vocab_table = utils.create_vocab_table(vocab_filename)
    if norm_filename is not None:
        means, stds = utils.load_normalization(norm_filename)
    else:
        means = stds = None

    dataset = utils.process_dataset(dataset, vocab_table, utils.SOS, utils.EOS,
                                    means, stds, 1, 1, is_infer=True)
    return dataset


def to_text(vocab_list, sample_ids):
    sym_list = [vocab_list[x] for x in sample_ids] + [utils.EOS]
    return args.delimiter.join(sym_list[:sym_list.index(utils.EOS)])


def main(args):
    config = tf.estimator.RunConfig(model_dir=args.model_dir)
    hparams = utils.create_hparams(args)
    hparams.decoder.set_hparam('beam_width', args.beam_width)

    vocab_list = utils.load_vocab(args.vocab)
    binf2phone_np = None
    if hparams.decoder.binary_outputs:
        if args.mapping is not None:
            vocab_list, mapping = utils.get_mapping(args.mapping, args.vocab)
            hparams.del_hparam('mapping')
            hparams.add_hparam('mapping', mapping)

        binf2phone = utils.load_binf2phone(args.binf_map, vocab_list)
        binf2phone_np = binf2phone.values

    def model_fn(features, labels, mode, config, params):
        return las_model_fn(features, labels, mode, config, params,
                            binf2phone=binf2phone_np)

    model = tf.estimator.Estimator(
        model_fn=model_fn,
        config=config,
        params=hparams)

    audio, _ = librosa.load(args.waveform, sr=SAMPLE_RATE, mono=True)
    features = [calculate_acoustic_features(args, audio)]

    predictions = model.predict(
        input_fn=lambda: input_fn(features, args.vocab, args.norm))
    predictions = list(predictions)
    for p in predictions:
        phone_pred_key = next(k for k in p.keys() if k.startswith('sample_ids'))
        beams = p[phone_pred_key].T
        if len(beams.shape) > 1:
            i = beams[0]
        else:
            i = beams
        i = i.tolist() + [utils.EOS_ID]
        i = i[:i.index(utils.EOS_ID)]
        text = to_text(vocab_list, i)
        text = text.split(args.delimiter)
        for k in p.keys():
            print(f'{k}: {p[k].shape}')
        print(text)
    if args.output_file:
        dump(predictions, args.output_file)
        print(f'Predictions are saved to {args.output_file}')


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    args = parse_args()
    main(args)
