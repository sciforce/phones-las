import os
from argparse import ArgumentParser
from tqdm import tqdm
import random


if __name__ == '__main__':
    parser = ArgumentParser('Recipe for Timit corpus.')
    parser.add_argument('--timit_path', help='Path to TIMIT corpus.', required=True, type=str)
    parser.add_argument('--val_fraction', help='What fraction of train to use in validation. '
                                               'Positive value would result in validation by a chunk of train set. '
                                               'Pass -1 if you intend to test on core test set and validate on'
                                               ' part of test set that is disjoint with core test both on'
                                               ' speakers and phrases set.',
                        default=-1, type=float)
    parser.add_argument('--output_dir', help='Path to output directory.', required=True, type=str)
    parser.add_argument('--labels_type', help='What type of labels to use.',
        type=str, default='text', choices=['text', 'phones60', 'phones48', 'phones39'], )
    parser.add_argument('--phone_map', help='Path to phoneme map file (for phones48 or phones39)',
        default='misc/phones.60-48-39.map')

    args = parser.parse_args()

    output_train = open(os.path.join(args.output_dir, 'train.csv'), 'w')
    output_val = open(os.path.join(args.output_dir, 'val.csv'), 'w')
    output_test = open(os.path.join(args.output_dir, 'test.csv'), 'w')
    output_test_core = open(os.path.join(args.output_dir, 'test-core.csv'), 'w')
    annotation_ext = '.TXT' if args.labels_type == 'text' else '.PHN'

    core_test_speakers = ['DAB0', 'WBT0', 'ELC0', 'TAS1', 'WEW0', 'PAS0',
                          'JMP0', 'LNT0', 'PKT0', 'LLL0', 'TLS0', 'JLM0',
                          'BPM0', 'KLT0', 'NLP0', 'CMJ0', 'JDH0', 'MGD0',
                          'GRT0', 'NJM0', 'DHC0', 'JLN0', 'PAM0', 'MLD0']
    core_test_phrases = open(os.path.join(os.path.dirname(__file__),
                                          '..', 'misc', 'timit_core_test_phrases.txt'), 'r').readlines()
    core_test_phrases = [x.strip() for x in core_test_phrases]
    mapping = None
    if args.labels_type != 'text':
        mapping_ind = ['phones60', 'phones48', 'phones39'].index(args.labels_type)
        if mapping_ind > 0:
            with open(args.phone_map, 'r') as f:
                lines = [line.split('\t') for line in f.read().strip().split('\n')]
                mapping = {line[0]: (line[mapping_ind] if len(line) > mapping_ind else None) for line in lines}

    for root, _, files in tqdm(os.walk(args.timit_path), desc='Collecting filenames'):
        for audio_filename in filter(lambda x: x.endswith('.WAV'), files):
            if 'SA' in audio_filename:
                # skip dialect sentences
                continue
            text_filename = audio_filename.replace('.WAV', annotation_ext)
            text_path = os.path.join(root, text_filename)
            if not os.path.exists(text_path):
                continue
            with open(text_path, 'r') as f:
                if args.labels_type == 'text':
                    text = f.read().strip().replace(',', '')
                    text = ' '.join(text.split()[2:])
                else:
                    text = [line.split(' ')[-1]  for line in f.read().strip().split('\n')]
                    if mapping is not None:
                        text = [mapping[t] for t in text]
                    text = ' '.join([t for t in text if t is not None])
            audio_path = os.path.join(root, audio_filename)
            language_tag = 'en' if args.labels_type == 'text' else 'arpabet'
            write_text = '{},{},{}\n'.format(audio_path, language_tag, text)
            if 'TEST' in root:
                output_test.write(write_text)
                if any(x in root for x in core_test_speakers):
                    output_test_core.write(write_text)
                elif args.val_fraction < 0 and not any(x in audio_filename for x in core_test_phrases):
                    output_val.write(write_text)
            elif args.val_fraction > 0 and random.random() < args.val_fraction:
                output_val.write(write_text)
            else:
                output_train.write(write_text)

