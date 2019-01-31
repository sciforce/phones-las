import os
from argparse import ArgumentParser
from tqdm import tqdm
import csv


parser = ArgumentParser()
parser.add_argument('--cv_path', help='Path to Mozilla Common Voice dataset.', required=True, type=str)
parser.add_argument('--use_other', help='Use other subset to increase data size?', required=True, type=bool)
parser.add_argument('--output_dir', help='Path to output directory.', required=True, type=str)

args = parser.parse_args()

quality = ['valid']
if args.use_other:
    quality += ['other']
types = ['train', 'dev', 'test']
for t in types:
    with open(os.path.join(args.output_dir, '{}.csv'.format(t)), 'w') as output:
        for q in quality:
            with open(os.path.join(args.cv_path, 'cv-{}-{}.csv'.format(q, t)), 'r') as f:
                reader = csv.DictReader(f)
                for row in tqdm(f, desc='Processing {}-{}'.format(q, t), unit='file'):
                    text = row['text'].strip().replace(',', '')
                    audio_path = os.path.join(args.cv_path, 'cv-{}-{}'.format(q, t), row['filename'])
                    write_text = '{},en,{}\n'.format(audio_path, text)
                    output.write(write_text)
