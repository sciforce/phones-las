import os
from argparse import ArgumentParser
from tqdm import tqdm


parser = ArgumentParser()
parser.add_argument('--vctk_path', help='Path to VCTK corpus.', required=True, type=str)
parser.add_argument('--output_dir', help='Path to output directory.', required=True, type=str)

args = parser.parse_args()

output_train = open(os.path.join(args.output_dir, 'train.csv'), 'w')
output_test = open(os.path.join(args.output_dir, 'test.csv'), 'w')
for root, _, files in tqdm(os.walk(args.vctk_path), desc='Collecting filenames'):
    for audio_filename in filter(lambda x: x.endswith('.wav'), files):
        text_root = root.replace('wav48', 'txt')
        text_filename = audio_filename.replace('.wav', '.txt')
        text_path = os.path.join(text_root, text_filename)
        if not os.path.exists(text_path):
            continue
        with open(text_path, 'r') as f:
            text = f.read().strip().replace(',', '')
        audio_path = os.path.join(root, audio_filename)
        write_text = '{},en,{}\n'.format(audio_path, text)
        if 'p3' in audio_filename:
            output_test.write(write_text)
        else:
            output_train.write(write_text)
