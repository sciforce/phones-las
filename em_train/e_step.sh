#!/bin/bash
python3 -m em_train.e_step --data timit/train.tfrecord --data_dir timit --vocab timit/vocab.txt \
                           --norm timit/norm.dmp --model_dir timit/model --beam_width 0 --batch_size 1 \
                           --num_channels 42 --keep_elem_prob 0.1