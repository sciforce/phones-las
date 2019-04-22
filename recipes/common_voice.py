import os
from argparse import ArgumentParser
from tqdm import tqdm
import csv


parser = ArgumentParser()
parser.add_argument('--cv_path', help='Path to Mozilla Common Voice dataset.', required=True, type=str)
parser.add_argument('--output_dir', help='Path to output directory.', required=True, type=str)
parser.add_argument('--lang', help='Parse only specified language.', required=False, type=str, default=None)

args = parser.parse_args()

quality = ['valid']
types = ['train', 'dev', 'test']
langs = [d for d in os.listdir(args.cv_path)
    if os.path.isdir(os.path.join(args.cv_path, d))]
for l in langs:
    if args.lang is not None and l != args.lang:
        continue
    for t in types:
        with open(os.path.join(args.output_dir, '{}.csv'.format(t)), 'a') as output:
            with open(os.path.join(args.cv_path, l, '{}.tsv'.format(t)), 'r') as f:
                reader = csv.reader(f, dialect='excel-tab')
                # skip header
                _ = next(reader)
                for row in tqdm(reader, desc='Processing {}-{}'.format(l, t), unit='file'):
                    utt_id, media_name, label = row[:3]
                    audio_path = os.path.join(args.cv_path, l, 'clips', media_name + '.mp3')
                    write_text = '{},{},{}\n'.format(audio_path, l, label.strip().replace(',', ''))
                    output.write(write_text)
