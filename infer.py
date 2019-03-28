import argparse
import os
import tensorflow as tf
from joblib import dump
import numpy as np
from tqdm import tqdm
from editdistance import eval as edist

import utils
from model_helper import las_model_fn
from utils import get_ipa


def parse_args():
    parser = argparse.ArgumentParser(
        description='Listen, Attend and Spell(LAS) implementation based on Tensorflow. '
                    'The model utilizes input pipeline and estimator API of Tensorflow, '
                    'which makes the training procedure truly end-to-end.')

    parser.add_argument('--data', type=str, required=True,
                        help='inference data in TFRecord format')
    parser.add_argument('--plain_targets', type=str, help='Path to CSV file with targets.')
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
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch size')
    parser.add_argument('--num_channels', type=int, default=39,
                        help='number of input channels')
    parser.add_argument('--delimiter', help='Symbols delimiter. Default: " "', type=str, default=' ')
    parser.add_argument('--take', help='Use this number of elements (0 for all).', type=int, default=0)
    parser.add_argument('--binf_map', type=str, default='misc/binf_map.csv',
                        help='Path to CSV with phonemes to binary features map')
    parser.add_argument('--use_phones_from_binf', help='User phonemes decoded from binary features decoder outputs.',
                        action='store_true')
    parser.add_argument('--convert_targets_to_ipa', help='Convert targets to ipa before comparison.',
                        action='store_true')

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
        binary_targets=binary_targets, labels_shape=labels_shape, is_infer=True)

    if args.take > 0:
        dataset = dataset.take(take)
    return dataset


def to_text(vocab_list, sample_ids):
    sym_list = [vocab_list[x] for x in sample_ids] + [utils.EOS]
    return args.delimiter.join(sym_list[:sym_list.index(utils.EOS)])


def main(args):
    config = tf.estimator.RunConfig(model_dir=args.model_dir)
    hparams = utils.create_hparams(args)

    hparams.decoder.set_hparam('beam_width', args.beam_width)

    vocab_list = utils.load_vocab(args.vocab)
    vocab_list_orig = vocab_list
    binf2phone_np = None
    binf2phone = None
    mapping = None
    if hparams.decoder.binary_outputs:
        if args.mapping is not None:
            vocab_list, mapping = utils.get_mapping(args.mapping, args.vocab)
            hparams.del_hparam('mapping')
            hparams.add_hparam('mapping', mapping)

        binf2phone = utils.load_binf2phone(args.binf_map, vocab_list)
        binf2phone_np = binf2phone.values

    def model_fn(features, labels,
        mode, config, params):
        return las_model_fn(features, labels, mode, config, params,
            binf2phone=binf2phone_np)

    model = tf.estimator.Estimator(
        model_fn=model_fn,
        config=config,
        params=hparams)

    phone_pred_key = 'sample_ids_phones_binf' if args.use_phones_from_binf else 'sample_ids'
    predictions = model.predict(
        input_fn=lambda: input_fn(
            args.data, args.vocab, args.norm, num_channels=args.num_channels, batch_size=args.batch_size,
            take=args.take, binf2phone=None),
        predict_keys=[phone_pred_key, 'embedding'])

    predictions = list(predictions)
    if args.plain_targets:
        targets = []
        for line in open(args.plain_targets, 'r'):
            delim = ','
            if '\t' in line:
                delim = '\t'
            cells = line.split(delim)
            _, lang, phrase = cells[:3]
            if args.convert_targets_to_ipa:
                if len(cells) == 4:
                    phrase = cells[-1].split(',')
                else:
                    phrase = get_ipa(phrase, lang)
            else:
                phrase = phrase.split()
                phrase = [x.strip().lower() for x in phrase]
            targets.append(phrase)
        err = 0
        tot = 0
        for p, t in tqdm(zip(predictions, targets)):
            beams = p[phone_pred_key].T
            if len(beams.shape) > 1:
                i = beams[0]
            else:
                i = beams
            i = i.tolist() + [utils.EOS_ID]
            i = i[:i.index(utils.EOS_ID)]
            if mapping is not None:
                target_ids = np.array([vocab_list_orig.index(p) for p in t])
                target_ids = np.array(mapping)[target_ids]
                t = [vocab_list[i] for i in target_ids]
            text = to_text(vocab_list, i)
            text = text.split(args.delimiter)
            err += edist(text, t)
            tot += len(t)
        print(f'PER: {100 * err / tot:2.2f}%')

    if args.beam_width > 0:
        predictions = [{
            'transcription': to_text(vocab_list, y[phone_pred_key][:, 0]),
            'embedding': y['embedding']
        } for y in predictions]
    else:
        predictions = [{
            'transcription': to_text(vocab_list, y[phone_pred_key]),
            'embedding': y['embedding']
        } for y in predictions]
    
    save_to = os.path.join(args.model_dir, 'infer.txt')
    with open(save_to, 'w') as f:
        f.write('\n'.join(p['transcription'] for p in predictions))
    
    save_to = os.path.join(args.model_dir, 'infer.dmp')
    dump(predictions, save_to)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    args = parse_args()
    main(args)
