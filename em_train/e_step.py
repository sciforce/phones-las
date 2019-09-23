import argparse
import os
import tensorflow as tf
from tqdm import tqdm
import numpy as np

import utils
from model_helper import las_model_fn


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=str, required=True,
                        help='inference data in TFRecord format')
    parser.add_argument('--data_dir', type=str, help='Data dir.')
    parser.add_argument('--vocab', type=str, required=True,
                        help='vocabulary table, listing vocabulary line by line')
    parser.add_argument('--norm', type=str, default=None,
                        help='normalization params')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='path of imported model')
    parser.add_argument('--beam_width', type=int, default=0,
                        help='number of beams (default 0: using greedy decoding)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch size')
    parser.add_argument('--num_channels', type=int, default=39,
                        help='number of input channels')
    parser.add_argument('--keep_elem_prob', help='Regulates amount of data to keep.', type=float, default=1.0)

    return parser.parse_args()


def to_text(vocab_list, sample_ids):
    sym_list = [vocab_list[x] for x in sample_ids] + [utils.EOS]
    return ' '.join(sym_list[:sym_list.index(utils.EOS)])


def main(args):
    config = tf.estimator.RunConfig(model_dir=args.model_dir)
    hparams = utils.create_hparams(args)

    hparams.decoder.set_hparam('beam_width', args.beam_width)

    vocab_list = utils.load_vocab(args.vocab)

    def model_fn(features, labels,
                 mode, config, params):
        return las_model_fn(features, labels, mode, config, params)

    model = tf.estimator.Estimator(
        model_fn=model_fn,
        config=config,
        params=hparams)

    predictions = model.predict(
        input_fn=lambda: utils.input_fn(
            args.data, args.vocab, args.norm, num_channels=args.num_channels, batch_size=args.batch_size,
            is_infer=True),
        predict_keys=['sample_ids'])

    targets, seed_data = [], {}
    for line in open(os.path.join(args.data_dir, 'train.csv'), 'r'):
        cells = line.split(',')
        sound, lang, _ = cells[:3]
        targets.append((sound, lang))
    for line in open(os.path.join(args.data_dir, 'seed.csv'), 'r'):
        cells = line.split(',')
        sound, _, phrase = cells[:3]
        seed_data[sound] = phrase.strip()
    with open(os.path.join(args.data_dir, 'e_step.csv'), 'w') as f:
        for p, target in tqdm(zip(predictions, targets)):
            if target[0] not in seed_data:
                if np.random.rand() > args.keep_elem_prob:
                    continue
                beams = p['sample_ids'].T
                i = beams.tolist() + [utils.EOS_ID]
                i = i[:i.index(utils.EOS_ID)]
                text = to_text(vocab_list, i)
            else:
                text = seed_data[target[0]]
            f.write('{},{},{}\n'.format(target[0], target[1], text))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    args = parse_args()
    main(args)
