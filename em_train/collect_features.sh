#!/bin/bash
CUDA_VISIBLE_DEVICES="" python3 preprocess_all.py --input_file timit/e_step.csv --output_file timit/e_step.tfrecord \
                          --feature_type mfcc --energy --deltas --n_jobs 1 --targets words