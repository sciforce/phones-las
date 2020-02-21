# Phones recognition and articulatory features detection using LAS

## Overview
This repository is an official implementation for Interspeech 2019 "Attention model for articulatory features detection" [paper][paper_arxiv].
Base code is a fork of [WindQAQ implementation][original_implementation] of [*Listen, Attend and Spell*][las] (LAS) model in Tensorflow.

Our contributions:
* Recipes for some common datasets: Librispeech, Common Voice, TIMIT, L2 artctic, SLR32 and VCTK.
* Phones/words/characters/binary features data collection switch. Usefull if you intend to train on IPA targets.
* Support of MFE, MFCC and Lyon's model features.
* Decoder for phonological features.
* Support for multitask training: phones recognition and indicators estimation. 
* Option for joint training with text auto-encoder.

## Usage

### Requirements
Run `pip install -r requirements.txt` to get the necessary version.

To load mp3 files you need to have ffmpeg installed.
        
- conda: `conda install -c conda-forge ffmpeg`

- Linux: `apt-get install ffmpeg` or `apt-get install gstreamer1.0-plugins-base gstreamer1.0-plugins-ugly`
        
- Mac: `brew install ffmpeg` or `brew install gstreamer`

For GStreamer also install the Python bindings with `pip install pygobject`.
        
To do training on phone targets, you'll need `espeak-ng` installed.

### Data Preparing
Before running the training script, you should convert your data into TFRecord format, collect normalization data and prepare vocabulary.
To do that, collect your train and test data in separate CSV files like this:
```csv
filename1.wav,en,big brown fox
filename2.wav,en,another fox
```
Recipes for some datasets are available in `recipes` folder.
After that call data collection script: `process_all.py`.
```text
usage: preprocess_all.py [-h] --input_file INPUT_FILE --output_file
                         OUTPUT_FILE [--norm_file NORM_FILE]
                         [--vocab_file VOCAB_FILE] [--top_k TOP_K]
                         [--n_mfcc N_MFCC] [--n_mels N_MELS] [--window WINDOW]
                         [--step STEP] [--n_jobs N_JOBS]

optional arguments:
  -h, --help            show this help message and exit
  --input_file INPUT_FILE
                        File with audio paths and texts.
  --output_file OUTPUT_FILE
                        Target TFRecord file name.
  --norm_file NORM_FILE
                        File name for normalization data.
  --vocab_file VOCAB_FILE
                        Vocabulary file name.
  --top_k TOP_K         Max size of vocabulary.
  --n_mfcc N_MFCC       Number of MFCC coeffs.
  --n_mels N_MELS       Number of mel-filters.
  --window WINDOW       Analysis window length in ms.
  --step STEP           Analysis window step in ms.
  --n_jobs N_JOBS       Number of parallel jobs.
  --targets {words,phones}
                        Determines targets type.
```

### Training and Evaluation
Simply run `python3 train.py --train TRAIN_TFRECORD --vocab VOCAB_TABLE --model_dir MODEL_DIR --norm NORM_FILE`.
You can also specify the validation data and some hyperparameters.
To find out more, please run `python3 train.py -h`.
```text
usage: train.py [-h] --train TRAIN [--valid VALID] --vocab VOCAB [--norm NORM]
                [--mapping MAPPING] --model_dir MODEL_DIR
                [--eval_secs EVAL_SECS] [--encoder_units ENCODER_UNITS]
                [--encoder_layers ENCODER_LAYERS] [--use_pyramidal]
                [--decoder_units DECODER_UNITS]
                [--decoder_layers DECODER_LAYERS]
                [--embedding_size EMBEDDING_SIZE]
                [--sampling_probability SAMPLING_PROBABILITY]
                [--attention_type {luong,bahdanau,custom}]
                [--attention_layer_size ATTENTION_LAYER_SIZE] [--bottom_only]
                [--pass_hidden_state] [--batch_size BATCH_SIZE]
                [--num_channels NUM_CHANNELS] [--num_epochs NUM_EPOCHS]
                [--learning_rate LEARNING_RATE] [--dropout DROPOUT]

Listen, Attend and Spell(LAS) implementation based on Tensorflow. The model
utilizes input pipeline and estimator API of Tensorflow, which makes the
training procedure truly end-to-end.

optional arguments:
  -h, --help            show this help message and exit
  --train TRAIN         training data in TFRecord format
  --valid VALID         validation data in TFRecord format
  --vocab VOCAB         vocabulary table, listing vocabulary line by line
  --norm NORM           normalization params
  --mapping MAPPING     additional mapping when evaluation
  --model_dir MODEL_DIR
                        path of saving model
  --eval_secs EVAL_SECS
                        evaluation every N seconds, only happening when
                        `valid` is specified
  --encoder_units ENCODER_UNITS
                        rnn hidden units of encoder
  --encoder_layers ENCODER_LAYERS
                        rnn layers of encoder
  --use_pyramidal       whether to use pyramidal rnn
  --decoder_units DECODER_UNITS
                        rnn hidden units of decoder
  --decoder_layers DECODER_LAYERS
                        rnn layers of decoder
  --embedding_size EMBEDDING_SIZE
                        embedding size of target vocabulary, if 0, one hot
                        encoding is applied
  --sampling_probability SAMPLING_PROBABILITY
                        sampling probabilty of decoder during training
  --attention_type {luong,bahdanau,custom}
                        type of attention mechanism
  --attention_layer_size ATTENTION_LAYER_SIZE
                        size of attention layer, see
                        tensorflow.contrib.seq2seq.AttentionWrapperfor more
                        details
  --bottom_only         apply attention mechanism only at the bottommost rnn
                        cell
  --pass_hidden_state   whether to pass encoder state to decoder
  --batch_size BATCH_SIZE
                        batch size
  --num_channels NUM_CHANNELS
                        number of input channels
  --num_epochs NUM_EPOCHS
                        number of training epochs
  --learning_rate LEARNING_RATE
                        learning rate
  --dropout DROPOUT     dropout rate of rnn cell
```

### Tensorboard
With the help of tensorflow estimator API, you can launch tensorboard by `tensorboard --logdir=MODEL_DIR`  to see the training procedure.

### Notes

If you intend to use LAS architecture and not vanilla seq2seq model,
use `--use_pyramidal --pass_hidden_state --bottom_only` flags combination.

Rerun of `train.py` would result in most of parameters restored from original run.
Thus, if you wish to override this behaviour, delete `hparams.json` file.

This codebased has been used to train models referenced in our paper. It also can be used to replicate results on sequence level accuracies and word error rates.
For frame level accuracies computations we used a Jupyter Notebook that is not included in this repository.  

## References

- [Paper e-print][paper_arxiv]
- [WindQAQ implementation][original_implementation]
- [Listen, Attend and spell][las]
- [How to create TFRecord][sequence_example]
- [nabu's implementation][nabu]
- [Tensorflow official seq2seq code][nmt]
- [ASR model evaluation toolkit][asr_eval]

[paper_arxiv]: https://arxiv.org/pdf/1907.01914.pdf
[original_implementation]: https://github.com/WindQAQ/listen-attend-and-spell
[nabu]: https://github.com/vrenkens/nabu
[nmt]: https://github.com/tensorflow/nmt
[las]: https://arxiv.org/pdf/1508.01211.pdf
[sequence_example]: https://github.com/tensorflow/magenta/blob/master/magenta/common/sequence_example_lib.py
[asr_eval]: https://github.com/belambert/asr-evaluation
