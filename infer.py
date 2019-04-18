import argparse
import os
import tensorflow as tf
from joblib import dump
from tqdm import tqdm
from editdistance import eval as edist
import librosa
import pandas as pd

import utils
from model_helper import las_model_fn
from utils import get_ipa

SAMPLE_RATE = 16000

#TODO: move to parameters
WIN_LEN = 0.02 * SAMPLE_RATE
WIN_STEP = 0.01 * SAMPLE_RATE

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
    parser.add_argument('--calc_frame_binf_accuracy', help='Calculate binary features accuracy on frame leve. For TIMIT only!',
                        action='store_true')
    parser.add_argument('--mapping_for_frame_accuracy', type=str,
                        help='additional mapping when evaluating frame level accuracy. For TIMIT only!')
    parser.add_argument('--encoder_frame_step', type=int, default=40,
                        help='Encoder output frame step to be used for frame accuracy calculation.'
                             'Should be used only if --use_pyramidal was specified at trainig. For TIMIT only!')
    parser.add_argument('--use_markup_segments', action='store_true',
                        help='Map phonemes to markup segments using peaks in attention matrix. For TIMIT only!')

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

def phrase_to_binfs(phrase, df_mapping):
    binfs = np.zeros((len(phrase), len(df_mapping.index)), np.int)
    for i, phone in enumerate(phrase):
        binfs[i, :] = df_mapping[phone].values
    return binfs

def soft2hard_attention(attention):
    '''
    Convert soft attention to hard attention using DTW
    '''
    from fastdtw import dtw
    _, path = dtw(list(range(attention.shape[0])), list(range(attention.shape[1])),
        dist=lambda x, y: -attention[int(x), int(y)])
    hard_attention = np.zeros_like(attention)
    for i, j in path:
        hard_attention[i, j] = 1
    for frame in range(attention.shape[1]):
        inds = np.argwhere(hard_attention[:, frame] == 1)
        if inds.size <= 1:
            continue
        if attention[inds[0], frame] > attention[inds[-1], frame]:
            if frame < attention.shape[1] - 1 and hard_attention[inds[-1], frame + 1] == 1:
                hard_attention[inds[-1], frame] = 0
        else:
            if frame > 0 and hard_attention[inds[0], frame - 1] == 1:
                hard_attention[inds[0], frame] = 0
    return hard_attention

def attention_to_segments(attention):
    """
    :param attention: Attantion matrix, the last dimension corresponds to the input sequence
    """
    attention = soft2hard_attention(attention)

    alignment = []
    st_frame, en_frame, st_phone, en_phone, phone = 0, 0, 0, 0, 0
    phone2seg = np.zeros((attention.shape[0]), np.int32)
    for frame in range(attention.shape[1]):
        inds = np.argwhere(attention[:, frame] == 1)
        if inds.size > 0:
            phone = inds[-1][0]
        if phone > en_phone and frame > 0:
            en_phone = phone
            en_frame = frame
            alignment.append([st_frame, en_frame, st_phone, en_phone])
            phone2seg[st_phone:en_phone] = len(alignment) - 1
            st_phone = en_phone
            st_frame = en_frame
    alignment.append([st_frame, attention.shape[1],
                        st_phone, attention.shape[0]])
    phone2seg[st_phone:attention.shape[0]] = len(alignment) - 1
    alignment = np.vstack(alignment).astype(np.float)

    return alignment, phone2seg

def segments_to_attention(attention, markup_segments, text, df_mapping, binf_preds=None):
    SEARCH_WINDOW = 5 #phonemes
    last_phone = 0
    nphones = attention.shape[0]
    nfeatures = len(df_mapping.index)
    binfs = np.zeros((len(markup_segments), nfeatures))
    for i, segment in enumerate(markup_segments):
        phones_limit = min(last_phone + SEARCH_WINDOW, nphones)
        st = segment[0]
        en = max(segment[1], st + 1)
        phone = last_phone + np.squeeze(np.argmax(np.max(attention[last_phone:phones_limit, st:en], axis=1)))
        if binf_preds is None:
            binfs[i, :] = df_mapping[text[phone]].values
        else:
            binfs[i, :] = binf_preds[phone, :]
    return binfs

def segs_phones_to_frame_binf(alignment, phones, df_mapping, binf_preds=None):
    nfeatures = len(df_mapping.index)
    nframes_aligned = alignment.shape[0]
    nframes = int(alignment[-1, 1])
    frames_binf = np.zeros((nframes, nfeatures))
    for aligned_frame_ind in range(nframes_aligned):
        binfs = np.zeros((nfeatures), np.bool)
        st_phone = int(alignment[aligned_frame_ind, 2])
        en_phone = int(alignment[aligned_frame_ind, 3])
        for phone_ind in range(st_phone, en_phone):
            if binf_preds is not None:
                current_binfs = binf_preds[phone_ind, :]
            else:
                current_binfs = df_mapping[phones[phone_ind]].values
            binfs = np.logical_or(binfs, current_binfs)
        st = int(alignment[aligned_frame_ind, 0])
        en = int(alignment[aligned_frame_ind, 1])
        frames_binf[st:en, :] = binfs
    return frames_binf

def get_timit_binf_markup(sound_file, df_mapping, markup_mapping):
    nfeatures = len(df_mapping.index)
    phn_file =  sound_file.replace('.WAV', '.PHN')
    with open(phn_file, 'r') as fid:
        lines = fid.read().strip().split('\n')
    nphones = len(lines)
    markup = np.zeros((nphones, 2))
    binfs = np.zeros((nphones, nfeatures), np.bool)
    for i, line in enumerate(lines):
        st, en, phone = line.split()
        phone = markup_mapping[phone]
        if phone is None:
            phone = 'sil'
        markup[i, :] = [int(st), int(en)]
        binfs[i, :] = df_mapping[phone].values
    markup = markup / SAMPLE_RATE
    wav, _ = librosa.load(sound_file, SAMPLE_RATE)
    nsamples = len(wav)
    return markup, binfs, nsamples

def get_binf_markup_frames(markup_frames, binfs):
    nframes = markup_frames[-1, -1]
    nfeatures = binfs.shape[1]
    frames_binf = np.zeros((nframes, nfeatures), np.bool)
    for frame, binf in zip(markup_frames, binfs):
        frames_binf[frame[0]:frame[1]] = np.logical_or(frames_binf[frame[0]:frame[1]], binf)
    return frames_binf

def count_correct(binf_pred, binf_tgt):
    return np.sum(np.equal(binf_pred, binf_tgt), axis=0)

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
    predict_keys=[phone_pred_key, 'embedding', 'alignment']
    if args.use_phones_from_binf:
        predict_keys.append('logits_binf')
        predict_keys.append('alignment_binf')
    predictions = model.predict(
        input_fn=lambda: input_fn(
            args.data, args.vocab, args.norm, num_channels=args.num_channels, batch_size=args.batch_size,
            take=args.take, binf2phone=None),
        predict_keys=predict_keys)

    if args.calc_frame_binf_accuracy:
        with open(args.mapping_for_frame_accuracy, 'r') as fid:
            mapping_lines = fid.read().strip().split()
        mapping_targets = dict()
        for line in mapping_lines:
            phones = line.split('\t')
            if len(phones) < 3:
                mapping_targets[phones[0]] = None
            else:
                mapping_targets[phones[0]] = phones[-1]


    predictions = list(predictions)
    if args.plain_targets:
        targets = []
        for line in open(args.plain_targets, 'r'):
            delim = ','
            if '\t' in line:
                delim = '\t'
            cells = line.split(delim)
            sound, lang, phrase = cells[:3]
            if args.convert_targets_to_ipa:
                if len(cells) == 4:
                    phrase = cells[-1].split(',')
                else:
                    phrase = get_ipa(phrase, lang)
            else:
                phrase = phrase.split()
                phrase = [x.strip().lower() for x in phrase]
            if args.calc_frame_binf_accuracy:
                markup, binfs, nsamples = get_timit_binf_markup(sound, binf2phone, mapping_targets)
                targets.append((phrase, markup, binfs, nsamples))
            else:
                targets.append(phrase)
        save_to = os.path.join(args.model_dir, 'infer_targets.txt')
        with open(save_to, 'w') as f:
            f.write('\n'.join(args.delimiter.join(t) for t in targets))
        err = 0
        tot = 0
        optimistic_err = 0
        if args.calc_frame_binf_accuracy:
            frames_count = 0
            correct_frames_count = np.zeros((len(binf2phone.index)))
        for p, target in tqdm(zip(predictions, targets)):
            first_text = []
            min_err = 100000
            beams = p[phone_pred_key].T
            if len(beams.shape) > 1:
                for bi, i in enumerate(p['sample_ids'].T):
                    i = i.tolist() + [utils.EOS_ID]
                    i = i[:i.index(utils.EOS_ID)]
                    text = to_text(vocab_list, i)
                    text = text.split(args.delimiter)
                    min_err = min([min_err, edist(text, t)])
                    if bi == 0:
                        first_text = text.copy()
                i = first_text
            else:
                i = beams.tolist() + [utils.EOS_ID]
                i = i[:i.index(utils.EOS_ID)]
                text = to_text(vocab_list, i)
                first_text = text.split(args.delimiter)

            t = target[0] if args.calc_frame_binf_accuracy else target
            if mapping is not None:
                target_ids = np.array([vocab_list_orig.index(p) for p in t])
                target_ids = np.array(mapping)[target_ids]
                t = [vocab_list[i] for i in target_ids]
            err += edist(first_text, t)
            optimistic_err += min_err
            tot += len(t)

            if args.calc_frame_binf_accuracy:
                attention = p['alignment'][:len(first_text), :]
                binf_preds = None
                if args.use_phones_from_binf:
                    logits = p['logits_binf'][:-1, :]
                    binf_preds = np.round(1 / (1 + np.exp(-logits)))
                    attention = p['alignment_binf'][:binf_preds.shape[0], :]

                markup, binfs, nsamples = target[1:]
                markup = np.minimum(markup, nsamples / SAMPLE_RATE)
                markup_frames = librosa.time_to_frames(markup, SAMPLE_RATE,
                    args.encoder_frame_step * SAMPLE_RATE / 1000, WIN_LEN)
                markup_frames[markup_frames < 0] = 0
                markup_frames_binf = get_binf_markup_frames(markup_frames, binfs)

                if not args.use_markup_segments:
                    alignment, _ = attention_to_segments(attention)
                    pred_frames_binf = segs_phones_to_frame_binf(alignment, first_text, binf2phone, binf_preds)
                else:
                    binfs_pred = segments_to_attention(attention, markup_frames, first_text, binf2phone, binf_preds)
                    pred_frames_binf = get_binf_markup_frames(markup_frames, binfs_pred)

                if pred_frames_binf.shape[0] != markup_frames_binf.shape[0]:
                    print('Warining: sound {} prediction frames {} target frames {}'.format(t,
                        pred_frames_binf.shape[0], markup_frames_binf.shape[0]))
                    nframes_fixed = min(pred_frames_binf.shape[0], markup_frames_binf.shape[0])
                    pred_frames_binf = pred_frames_binf[:nframes_fixed, :]
                    markup_frames_binf = markup_frames_binf[:nframes_fixed, :]
                correct_frames_count += count_correct(pred_frames_binf, markup_frames_binf)
                frames_count += markup_frames_binf.shape[0]

            # Compare binary feature vectors

        print(f'PER: {100 * err / tot:2.2f}%')
        print(f'Optimistic PER: {100 * optimistic_err / tot:2.2f}%')

        if args.calc_frame_binf_accuracy:
            df = pd.DataFrame({'correct': correct_frames_count / frames_count}, index=binf2phone.index)
            print(df)

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
