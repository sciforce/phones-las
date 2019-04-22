'''
A recipe for SLR32 TTS corpus of African languages from OpenSLR.
'''

import os
from argparse import ArgumentParser
from tqdm import tqdm
import csv


parser = ArgumentParser()
parser.add_argument('--cv_path', help='Path to Open SLR dataset for specific language.', required=True, type=str)
parser.add_argument('--output_dir', help='Path to output directory.', required=True, type=str)
parser.add_argument('--lang', help='Language code.', required=True, type=str, default='en')

args = parser.parse_args()

with open(os.path.join(args.output_dir, 'test.csv'), 'w') as output:
    with open(os.path.join(args.cv_path, 'line_index.tsv'), 'r') as f:
        reader = csv.reader(f, dialect='excel-tab')
        # skip header
        _ = next(reader)
        for row in tqdm(reader, desc='Processing', unit='file'):
            media_name, label = row[:2]
            audio_path = os.path.join(args.cv_path, 'wavs', media_name + '.wav')
            write_text = '{},{},{}\n'.format(audio_path, args.lang, label.strip().replace(',', ''))
            output.write(write_text)
