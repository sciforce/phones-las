import os
from argparse import ArgumentParser
from tqdm import tqdm


parser = ArgumentParser()
parser.add_argument('--librispeech_path', help='Path to LibriSpeech corpus.', required=True, type=str)
parser.add_argument('--output_dir', help='Path to output directory.', required=True, type=str)
parser.add_argument('--use_other_train', help='Use other subset for training?', action='store_true')
parser.add_argument('--use_other_dev', help='Use other dev subset?', action='store_true')
parser.add_argument('--use_other_test', help='Use other dev subset?', action='store_true')

args = parser.parse_args()

output_train = open(os.path.join(args.output_dir, 'train.csv'), 'w')
output_dev = open(os.path.join(args.output_dir, 'val.csv'), 'w')
output_test = open(os.path.join(args.output_dir, 'test.csv'), 'w')
for root, _, files in tqdm(os.walk(args.librispeech_path), desc='Collecting filenames'):
    if 'train-other' in root and not args.use_other_train:
        continue
    if 'dev-other' in root and not args.use_other_dev:
        continue
    if 'test-other' in root and not args.use_other_test:
        continue
    for text_filename in filter(lambda x: x.endswith('.trans.txt'), files):
        with open(os.path.join(root, text_filename)) as f:
            for line in f:
                words = line.split()
                audio_filename = '{}.wav'.format(words[0])
                text = ' '.join(words[1:]).lower().strip().replace(',', '')
                audio_path = os.path.join(root, audio_filename)
                write_text = '{},en,{}\n'.format(audio_path, text)
                if 'train' in root:
                    output_train.write(write_text)
                elif 'dev' in root:
                    output_dev.write(write_text)
                else:
                    output_test.write(write_text)
