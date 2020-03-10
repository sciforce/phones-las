import argparse
import warnings

import librosa
from tqdm import tqdm

SAMPLE_RATE = 16000


def read_audio_and_text(inputs):
    audio_path = inputs['file_path']
    audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    return audio.size / float(sr)


def process_line(args, line):
    filename, language, text = line.split(args.delimiter)
    inputs = {
        'file_path': filename,
        'text': text.strip(),
        'language': language
    }
    try:
        return read_audio_and_text(inputs)
    except Exception as err:
        print(str(err))
        return 0


def main(args):
    with open(args.input_file) as f:
        total = 0
        for x in tqdm(f):
            total += process_line(args, x)
        total /= 3600
        print('Hours: ', total)


if __name__ == '__main__':
    warnings.simplefilter(action='ignore', category=FutureWarning)

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', help='File with audio paths and texts.', required=True)
    parser.add_argument('--step', help='Analysis window step in ms.', type=int, default=10)
    parser.add_argument('--start', help='Index of example to start from', type=int, default=0)
    parser.add_argument('--delimiter', help='CSV delimiter', type=str, default=',')

    main(parser.parse_args())
