import os
import argparse
import random


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Select initial seed for EM training.')
    parser.add_argument('--seed_size', type=int, default=1, help='how many transcripts to consider')
    parser.add_argument('--base_dir', type=str, required=True, help='directory with corpus files')
    args = parser.parse_args()

    with open(os.path.join(args.base_dir, 'train.csv'), 'r') as f:
        data = f.readlines()
    seed = random.sample(data, args.seed_size)
    with open(os.path.join(args.base_dir, 'seed.csv'), 'w') as f:
        for l in seed:
            f.write('{}\n'.format(l.strip()))
