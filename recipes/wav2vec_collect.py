from tqdm import tqdm
import h5py
import os
import argparse
import numpy as np
import tensorflow as tf

from preprocess_all import make_example


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='directory with generated h5context files')
    parser.add_argument('--input_file', type=str, help='csv file generated by recipe')
    parser.add_argument('--replace_dir', type=str, help='directory int csv to replace to get h5context files paths')
    parser.add_argument('--output_file', type=str, help='TF record to write results to')
    args = parser.parse_args()

    with open(args.input_file, 'r') as f:
        with tf.io.TFRecordWriter(args.output_file) as writer:
            for line in tqdm(f):
                p, _, phones = line.strip().split(',')
                h5_p = p.replace(args.replace_dir, args.data_dir).replace('.wav', '.h5context')
                with h5py.File(os.path.join(args.data_dir, h5_p), 'r') as h5_f:
                    shape = np.array(h5_f['info'])
                    features = np.array(h5_f['features']).reshape([int(shape[1]), int(shape[2])])
                phones = list(phones)
                writer.write(make_example(features, phones).SerializeToString())
