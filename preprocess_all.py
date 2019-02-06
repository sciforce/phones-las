from collections import Counter
import librosa
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from multiprocessing import Lock
from joblib import Parallel, delayed, dump
from argparse import ArgumentParser
from speechpy.feature import mfe, mfcc, extract_derivative_feature

from utils import get_ipa, ipa2binf, load_binf2phone


SAMPLE_RATE = 16000

vocabulary = Counter()
means = None
stds = None
total = 0
par_handle = None
session = tf.Session()
tfrecord_mutex = Lock()
stats_mutex = Lock()
binf2phone = None


def make_example(input, label):
    if isinstance(label, list):
        label_list = tf.train.FeatureList(feature=[
            tf.train.Feature(bytes_list=tf.train.BytesList(value=[p.encode()]))
            for p in label
        ])
    else:
        label_list = tf.train.FeatureList(feature=[
            tf.train.Feature(float_list=tf.train.FloatList(value=f))
            for f in label
        ])
    feature_lists = tf.train.FeatureLists(feature_list={
        'labels': label_list,
        'inputs': tf.train.FeatureList(feature=[
            tf.train.Feature(float_list=tf.train.FloatList(value=f))
            for f in input
        ])
    })

    return tf.train.SequenceExample(feature_lists=feature_lists)


def read_audio_and_text(inputs):
    audio_path = inputs['file_path']
    text = inputs['text']
    language = inputs['language']
    text = ' '.join(text.split())
    for p in ',.:;?!-_':
        text = text.replace(p, '')
    text = text.lower().split()
    audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    return {
        'waveform': audio,
        'text': text,
        'language': language
    }


def calculate_acoustic_features(args, waveform):
    n_fft = int(args.window*SAMPLE_RATE/1000.0)
    hop_length = int(args.step * SAMPLE_RATE / 1000.0)
    if 'mfe' == args.feature_type:
        log_cut = 1e-8
        if args.backend=='speechpy':
            spec, energy = mfe(waveform, SAMPLE_RATE, frame_length=args.window*1e-3,
                frame_stride=args.step*1e-3, num_filters=args.n_mels, fft_length=n_fft)
            acoustic_features = np.hstack((spec, energy[:, np.newaxis]))
        else:
            spec = librosa.feature.melspectrogram(y=waveform, sr=SAMPLE_RATE, n_fft=n_fft, 
                hop_length=hop_length, n_mels=args.n_mels).transpose()
            energy = librosa.feature.rmse(y=waveform, frame_length=n_fft, hop_length=hop_length).transpose()
            acoustic_features = np.hstack((spec, energy))
        acoustic_features = np.log(log_cut + acoustic_features)
    elif 'mfcc' == args.feature_type:
        if args.backend=='speechpy':
            acoustic_features = mfcc(waveform, SAMPLE_RATE, frame_length=args.window*1e-3,
                frame_stride=args.step*1e-3, num_filters=args.n_mels, fft_length=n_fft,
                num_cepstral = args.n_mfcc)
        else:
            acoustic_features = librosa.feature.mfcc(y=waveform, sr=SAMPLE_RATE, n_mfcc=args.n_mfcc,
                n_fft=n_fft, hop_length=hop_length, n_mels=args.n_mels).transpose()
    else:
        raise ValueError('Unexpected features type.')
    if args.deltas:
        orig_shape = acoustic_features.shape
        if args.backend=='speechpy':
            acoustic_features = extract_derivative_feature(acoustic_features)
        else:
            delta = librosa.feature.delta(acoustic_features)
            ddelta = librosa.feature.delta(acoustic_features, order=2)
            acoustic_features = np.stack((acoustic_features[:, :, np.newaxis],
                delta[:, :, np.newaxis], ddelta[:, :, np.newaxis]), axis=-1)
        acoustic_features = np.reshape(acoustic_features, (-1, orig_shape[-1] * 3))
    return acoustic_features


def build_features_and_vocabulary_fn(args, inputs):
    global means, stds, total
    waveform = inputs['waveform']
    text = inputs['text']
    language = inputs['language']
    binf = None
    if args.targets in ('phones', 'binary_features'):
        if language not in ['arpabet', 'ipa']:
            text = ' '.join(text)
            text = get_ipa(text, language)
        if args.targets == 'binary_features':
            binf = ipa2binf(text, binf2phone, 'ipa'==language)
    vocabulary.update(text)
    acoustic_features = calculate_acoustic_features(args, waveform)
    if args.norm_file:
        with stats_mutex:
            if means is None:
                means = np.mean(acoustic_features, axis=0)
                stds = np.std(acoustic_features, axis=0)
            else:
                means += np.mean(acoustic_features, axis=0)
                stds += np.std(acoustic_features, axis=0)
            total += 1
    return {
        'mfcc': acoustic_features,
        'text': binf if args.targets == 'binary_features' else text
    }


def write_tf_output(writer, inputs):
    with tfrecord_mutex:
        writer.write(make_example(inputs['mfcc'], inputs['text']).SerializeToString())
    par_handle.update()


def process_line(args, writer, line):
    filename, language, text = line.split(',')
    inputs = {
        'file_path': filename,
        'text': text.strip(),
        'language': language
    }
    out = read_audio_and_text(inputs)
    out = build_features_and_vocabulary_fn(args, out)
    write_tf_output(writer, out)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--input_file', help='File with audio paths and texts.', required=True)
    parser.add_argument('--output_file', help='Target TFRecord file name.', required=True)
    parser.add_argument('--norm_file', help='File name for normalization data.', default=None)
    parser.add_argument('--vocab_file', help='Vocabulary file name.', default=None)
    parser.add_argument('--top_k', help='Max size of vocabulary.', type=int, default=1000)
    parser.add_argument('--feature_type', help='Acoustic feature type.', type=str,
                        choices=['mfe', 'mfcc'], default='mfcc')
    parser.add_argument('--backend', help='Library for calculating acoustic features.', type=str,
                        choices=['speechpy', 'librosa'], default='librosa')
    parser.add_argument('--n_mfcc', help='Number of MFCC coeffs.', type=int, default=13)
    parser.add_argument('--n_mels', help='Number of mel-filters.', type=int, default=40)
    parser.add_argument('--window', help='Analysis window length in ms.', type=int, default=20)
    parser.add_argument('--step', help='Analysis window step in ms.', type=int, default=10)
    parser.add_argument('--deltas', help='Calculate deltas and double-deltas.', action='store_true')
    parser.add_argument('--n_jobs', help='Number of parallel jobs.', type=int, default=4)
    parser.add_argument('--targets', help='Determines targets type.', type=str,
                        choices=['words', 'phones', 'binary_features'], default='words')
    parser.add_argument('--binf_map', help='Path to CSV with phonemes to binary features map',
                        type=str, default='misc/binf_map.csv')
    args = parser.parse_args()
    if args.targets in ('phones', 'binary_features'):
        binf2phone = load_binf2phone(args.binf_map)
    print('Processing audio dataset from file {}.'.format(args.input_file))
    window = int(SAMPLE_RATE * args.window / 1000.0)
    step = int(SAMPLE_RATE * args.step / 1000.0)
    lines = open(args.input_file, 'r').readlines()
    par_handle = tqdm(unit='sound')
    with tf.io.TFRecordWriter(args.output_file) as writer:
        if args.n_jobs > 1:
            Parallel(n_jobs=args.n_jobs, prefer="threads")(delayed(process_line)(args, writer, x) for x in lines)
        else:
            for x in lines:
                process_line(args, writer, x)
    session.close()
    par_handle.close()
    if args.norm_file is not None:
        dump([means / total, stds / total], args.norm_file)
    if args.vocab_file is not None:
        with open(args.vocab_file, 'w') as f:
            for x, _ in vocabulary.most_common(args.top_k):
                f.write(x + '\n')
