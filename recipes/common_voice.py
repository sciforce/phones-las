import os
from argparse import ArgumentParser
from tqdm import tqdm
import csv


parser = ArgumentParser()
parser.add_argument('--cv_path', help='Path to Mozilla Common Voice dataset.', required=True, type=str)
parser.add_argument('--language', help='Language for collection.', required=True, type=str)
parser.add_argument('--output_dir', help='Path to output directory.', required=True, type=str)

args = parser.parse_args()

types = ['train', 'dev', 'test']
base_path = os.path.join(args.cv_path, args.language)
for t in types:
    with open(os.path.join(args.output_dir, '{}.csv'.format(t)), 'w') as output:
        with open(os.path.join(base_path, '{}.tsv'.format(t)), 'r') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in tqdm(f, desc='Processing {} for {}'.format(t, args.language), unit='file'):
                text = row['sentence'].strip().replace(',', '')
                audio_path = os.path.join(base_path, 'clips', '{}.mp3'.format(row['path']))
                write_text = '{},{},{}\n'.format(audio_path, args.language, text)
                output.write(write_text)
