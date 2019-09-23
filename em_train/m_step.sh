#!/bin/bash
python3 ./train.py --train timit/e_step.tfrecord --valid timit/test.tfrecord --vocab timit/vocab.txt \
                   --norm timit/norm.dmp --model_dir timit/model --encoder_units 256 --encoder_layers 3 \
                   --decoder_units 256 --decoder_layers 1 --attention_type luong --num_channels 42 \
                   --batch_size 32 --learning_rate 1e-4 --eval_secs 300 --use_pyramidal --num_epochs 2 \
                   --ctc_weight 0 --l2_reg_scale 1e-5 --add_noise 0 --dropout 0.4 --sampling_probability 0.2 --reset