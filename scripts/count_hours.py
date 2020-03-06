import warnings
import argparse
import librosa
SAMPLE_RATE = 16000


def read_audio_and_text(inputs):
    audio_path = inputs['file_path']
    text = inputs['text']
    language = inputs['language']
    text = ' '.join(text.split())
    for p in ',.:;?!-_':
        text = text.replace(p, '')
    text = text.lower().split()
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
    # window = int(SAMPLE_RATE * args.window / 1000.0)
    # step = int(SAMPLE_RATE * args.step / 1000.0)
    lines = open(args.input_file, 'r').readlines()
    count = len(lines) - args.start
    if args.count > 0 and args.count < len(lines):
        count = args.count
    lines = lines[args.start:count + args.start]
    total = 0
    for x in lines:
        total += process_line(args, x)
    total /= 3600
    print(total)


if __name__ == '__main__':
    warnings.simplefilter(action='ignore', category=FutureWarning)

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', help='File with audio paths and texts.', required=True)
    parser.add_argument('--step', help='Analysis window step in ms.', type=int, default=10)
    parser.add_argument('--start', help='Index of example to start from', type=int, default=0)
    parser.add_argument('--count', help='Maximal phrases count, -1 for all phrases', type=int, default=-1)
    parser.add_argument('--delimiter', help='CSV delimiter', type=str, default=',')
    args = parser.parse_args()

    main(args)
