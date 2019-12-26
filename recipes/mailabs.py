import json
import os
from argparse import ArgumentParser
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument('--data_path', help='Path to corpus by_book directory.', required=True, type=str)
parser.add_argument('--output_dir', help='Path to output directory.', required=True, type=str)
parser.add_argument('--dev_speakers', help='Use these speakers for validation.', required=True, type=str, nargs='+')
parser.add_argument('--language', help='Corpus language', default='uk')

args = parser.parse_args()

output_train = open(os.path.join(args.output_dir, 'train.csv'), 'w')
output_dev = open(os.path.join(args.output_dir, 'val.csv'), 'w')
for root, _, files in tqdm(os.walk(args.data_path), desc='Collecting filenames'):
    for text_filename in files:
        if text_filename != 'metadata_mls.json':
            continue
        with open(os.path.join(root, text_filename)) as f:
            data = json.load(f)
            for audio_filename, text_data in data.items():
                text = text_data['clean'].lower().strip().replace(',', '')
                audio_path = os.path.join(root, 'wavs', audio_filename)
                write_text = '{},{},{}\n'.format(audio_path, args.language, text)
                if os.path.basename(os.path.dirname(root)) in args.dev_speakers:
                    output_dev.write(write_text)
                else:
                    output_train.write(write_text)
