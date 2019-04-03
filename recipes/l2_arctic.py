import os
from argparse import ArgumentParser
from tqdm import tqdm
import random

parser = ArgumentParser()
parser.add_argument('--arctic_path', help='Path to L2 Arctic corpus.', type=str, required=True)
parser.add_argument('--val_fraction', help='What fraction of train to use in validation.',
                    default=0.05, type=float)
parser.add_argument('--test_speaker', help='Speaker used for testing.'
                    'If not specified, choose test speaker randmoly.',
                    default=None, type=str)
parser.add_argument('--output_dir', help='Path to output directory.', required=True, type=str)
parser.add_argument('--random_seed', help='Random seed for splitting train, test, validation sets.',
                    type=int, default=None)

args = parser.parse_args()

if args.random_seed is not None:
    random.seed(args.random_seed)

speakers = [d for d in os.listdir(args.arctic_path)
    if os.path.isdir(os.path.join(args.arctic_path, d))]

test_speaker = args.test_speaker
if test_speaker is None:
    test_speaker = speakers[random.randint(0, len(speakers) - 1)]

output_train = open(os.path.join(args.output_dir, 'train.csv'), 'w')
output_val = open(os.path.join(args.output_dir, 'val.csv'), 'w')
output_test = open(os.path.join(args.output_dir, 'test.csv'), 'w')

for speaker in tqdm(speakers, desc='Collecting speakers'):
    speaker_dir = os.path.join(args.arctic_path, speaker)
    files = os.listdir(os.path.join(speaker_dir, 'wav'))
    for f in tqdm(filter(lambda x: x.endswith('.wav'), files),
        desc='Processing speaker {}'.format(speaker)):
        wav_path = os.path.join(speaker_dir, 'wav', f)
        markup_path = os.path.join(speaker_dir, 'transcript', f.replace('.wav', '.txt'))
        with open(markup_path, 'r') as fid:
            markup_text = fid.read().strip(' \n').replace(',', '')
        write_text = '{},{},{}\n'.format(wav_path, 'en', markup_text)
        if speaker == test_speaker:
            output_test.write(write_text)
        elif random.random() < args.val_fraction:
            output_val.write(write_text)
        else:
            output_train.write(write_text)