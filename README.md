# Phones recognition and articulatory features detection using LAS

## Overview
This repository is an official implementation for Interspeech 2019 "Attention model for articulatory features detection" [paper][paper_arxiv].
Base code is a fork of [WindQAQ implementation][original_implementation] of [*Listen, Attend and Spell*][las] (LAS) model in Tensorflow.

Our contributions:
* Recipes for some common datasets: Librispeech, Common Voice, TIMIT, L2 artctic, SLR32 and VCTK.
* Phones/words/characters/binary features data collection switch. Useful if you intend to train on IPA targets.
* Support of MFE, MFCC and Lyon's model features.
* Decoder for phonological features.
* Support for multitask training: phones recognition and indicators estimation. 
* Option for joint training with CTC loss on encoder.

## Usage

### Requirements
Run `pip install -r requirements.txt` to get the necessary version.

To load mp3 files you need to have ffmpeg installed.
        
- conda: `conda install -c conda-forge ffmpeg`
- Linux: `apt-get install ffmpeg` or `apt-get install gstreamer1.0-plugins-base gstreamer1.0-plugins-ugly`
- Mac: `brew install ffmpeg` or `brew install gstreamer`

For GStreamer also install the Python bindings with `pip install pygobject`.
        
To do training on phone targets, you'll need `espeak-ng` installed.

### espeak-ng Mac installation

It should be build locally. Clone espeak-ng repository.
Then run:

```shell script
./configure --exec-prefix=/usr/local/ --datarootdir=/usr/local --sysconfdir=/usr/local --sharedstatedir=/usr/local --localstatedir=/usr/local --includedir=/usr/local --with-extdict-ru --with-extdict-zh --with-extdict-zhy
make 
sudo make LIBDIR=/usr/local/lib install
```

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
                         OUTPUT_FILE [--top_k TOP_K] [--save_norm]
                         [--save_vocab] [--feature_type {mfe,mfcc,lyon}]
                         [--backend {speechpy,librosa}] [--n_mfcc N_MFCC]
                         [--n_mels N_MELS] [--energy] [--window WINDOW]
                         [--step STEP] [--deltas] [--n_jobs N_JOBS]
                         [--targets {words,phones,binary_features,chars}]
                         [--binf_map BINF_MAP] [--start START] [--count COUNT]
                         [--delimiter DELIMITER]

optional arguments:
  -h, --help            show this help message and exit
  --input_file INPUT_FILE
                        File with audio paths and texts.
  --output_file OUTPUT_FILE
                        Target TFRecord file name.
  --top_k TOP_K         Max size of vocabulary.
  --save_norm           Specify if you want to save norm data
  --save_vocab          Specify if you want to save vocabulary
  --feature_type {mfe,mfcc,lyon}
                        Acoustic feature type.
  --backend {speechpy,librosa}
                        Library for calculating acoustic features.
  --n_mfcc N_MFCC       Number of MFCC coeffs.
  --n_mels N_MELS       Number of mel-filters.
  --energy              Compute energy.
  --window WINDOW       Analysis window length in ms.
  --step STEP           Analysis window step in ms.
  --deltas              Calculate deltas and double-deltas.
  --n_jobs N_JOBS       Number of parallel jobs.
  --targets {words,phones,binary_features,chars}
                        Determines targets type.
  --binf_map BINF_MAP   Path to CSV with phonemes to binary features map
  --start START         Index of example to start from
  --count COUNT         Maximal phrases count, -1 for all phrases
  --delimiter DELIMITER
                        CSV delimiter
```

For TIMIT experiments use `--targets words`, because text is already converted to space-separated phonemes.
For most of other cases you'd want to use `--target phones`.
Example call:
```bash script
python preprocess_all.py --input_file ./model/train.csv --output_file ./model/train.tfr --save_norm --save_vocab --feature_type mfcc --backend librosa --energy --deltas --n_jobs 1 --targets phones
```
Normally you'd want to generate norm and vocab files only from train set, thus for dev and test do not pass `--save_norm` and `--save_vocab` params.

### Training and Evaluation
Training is done via `train.py` script.
```text
usage: train.py [-h] --train TRAIN [--valid VALID] [--t2t_format]
                [--t2t_problem_name T2T_PROBLEM_NAME] [--mapping MAPPING]
                --model_dir MODEL_DIR [--eval_secs EVAL_SECS]
                [--encoder_units ENCODER_UNITS]
                [--encoder_layers ENCODER_LAYERS] [--use_pyramidal]
                [--unidirectional] [--decoder_units DECODER_UNITS]
                [--decoder_layers DECODER_LAYERS]
                [--embedding_size EMBEDDING_SIZE]
                [--sampling_probability SAMPLING_PROBABILITY]
                [--attention_type {luong,bahdanau,custom,luong_monotonic,bahdanau_monotonic}]
                [--attention_layer_size ATTENTION_LAYER_SIZE] [--bottom_only]
                [--pass_hidden_state] [--batch_size BATCH_SIZE]
                [--num_parallel_calls NUM_PARALLEL_CALLS]
                [--num_channels NUM_CHANNELS] [--num_epochs NUM_EPOCHS]
                [--learning_rate LEARNING_RATE] [--dropout DROPOUT]
                [--l2_reg_scale L2_REG_SCALE] [--add_noise ADD_NOISE]
                [--noise_std NOISE_STD] [--binary_outputs] [--output_ipa]
                [--binf_map BINF_MAP] [--ctc_weight CTC_WEIGHT] [--reset]
                [--binf_sampling] [--binf_projection]
                [--binf_projection_reg_weight BINF_PROJECTION_REG_WEIGHT]
                [--binf_trainable] [--multitask] [--tpu_name TPU_NAME]
                [--max_frames MAX_FRAMES] [--max_symbols MAX_SYMBOLS]
                [--tpu_checkpoints_interval TPU_CHECKPOINTS_INTERVAL]
                [--t2t_features_hparams_override T2T_FEATURES_HPARAMS_OVERRIDE]

Listen, Attend and Spell(LAS) implementation based on Tensorflow. The model
utilizes input pipeline and estimator API of Tensorflow, which makes the
training procedure truly end-to-end.

optional arguments:
  -h, --help            show this help message and exit
  --train TRAIN         training data in TFRecord format
  --valid VALID         validation data in TFRecord format
  --t2t_format          Use dataset in the format of ASR problems of
                        Tensor2Tensor framework. --train param should be
                        directory
  --t2t_problem_name T2T_PROBLEM_NAME
                        Problem name for data in T2T format.
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
  --unidirectional      Use unidirectional RNN
  --decoder_units DECODER_UNITS
                        rnn hidden units of decoder
  --decoder_layers DECODER_LAYERS
                        rnn layers of decoder
  --embedding_size EMBEDDING_SIZE
                        embedding size of target vocabulary, if 0, one hot
                        encoding is applied
  --sampling_probability SAMPLING_PROBABILITY
                        sampling probabilty of decoder during training
  --attention_type {luong,bahdanau,custom,luong_monotonic,bahdanau_monotonic}
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
  --num_parallel_calls NUM_PARALLEL_CALLS
                        Number of elements to be processed in parallel during
                        the dataset transformation
  --num_channels NUM_CHANNELS
                        number of input channels
  --num_epochs NUM_EPOCHS
                        number of training epochs
  --learning_rate LEARNING_RATE
                        learning rate
  --dropout DROPOUT     dropout rate of rnn cell
  --l2_reg_scale L2_REG_SCALE
                        L2 regularization scale
  --add_noise ADD_NOISE
                        How often (in steps) to add Gaussian noise to the
                        weights, zero for disabling noise addition.
  --noise_std NOISE_STD
                        Weigth noise standard deviation.
  --binary_outputs      make projection layer output binary feature posteriors
                        instead of phone posteriors
  --output_ipa          With --binary_outputs on, make the graph output phones
                        and change sampling algorithm at training
  --binf_map BINF_MAP   Path to CSV with phonemes to binary features map
  --ctc_weight CTC_WEIGHT
                        If possitive, adds CTC mutlitask target based on
                        encoder.
  --reset               Reset HParams.
  --binf_sampling       with --output_ipa, do not use ipa sampling algorithm
                        for trainin, only for validation
  --binf_projection     with --binary_outputs and --output_ipa, use binary
                        features mapping instead of decoders projection layer.
  --binf_projection_reg_weight BINF_PROJECTION_REG_WEIGHT
                        with --binf_projection, weight for regularization term
                        for binary features log probabilities.
  --binf_trainable      trainable binary features matrix
  --multitask           with --binary_outputs use both binary features and IPA
                        decoders.
  --tpu_name TPU_NAME   TPU name. Leave blank to prevent TPU training.
  --max_frames MAX_FRAMES
                        If positives, sets that much frames for each batch.
  --max_symbols MAX_SYMBOLS
                        If positives, sets that much symbols for each batch.
  --tpu_checkpoints_interval TPU_CHECKPOINTS_INTERVAL
                        Interval for saving checkpoints on TPU, in steps.
  --t2t_features_hparams_override T2T_FEATURES_HPARAMS_OVERRIDE
                        String with overrided parameters used by Tensor2Tensor
                        problem.
```

#### Phones targets
```bash script
python train.py --train ./model/train.tfr --valid ./model/dev.tfr --model_dir ./model --eval_secs 1800 --l2_reg_scale 0 --ctc_weight -1 --encoder_units 256 --encoder_layers 4 --decoder_units 256 --decoder_layers 1 --use_pyramidal --sampling_probability 0.2 --dropout 0.2 --attention_type luong --num_epochs 500 --learning_rate 1e-4 --batch_size 16 --num_channels 42
```
After some time you may want to decrease learning rate. To do that, stop training and restart it:
```bash script
python train.py --train ./model/train.tfr --valid ./model/dev.tfr --model_dir ./model --learning_rate 1e-5
```

#### Phones targets with indicators proxy
```bash script
python train.py --train ./model/train.tfr --valid ./model/dev.tfr --model_dir ./model --eval_secs 1800 --l2_reg_scale 0 --ctc_weight -1 --encoder_units 256 --encoder_layers 4 --decoder_units 256 --decoder_layers 1 --use_pyramidal --sampling_probability 0.2 --dropout 0.2 --attention_type luong --num_epochs 500 --learning_rate 1e-4 --batch_size 16 --num_channels 42 --binary_outputs --output_ipa --binf_projection --binf_map misc/binf_map.csv
```

For TIMIT use `--binf_map misc/binf_map_arpabet_extended.csv`.

### Tensorboard
With the help of tensorflow estimator API, you can launch tensorboard by `tensorboard --logdir=MODEL_DIR`  to see the training procedure.

Also it is possible to share your results via tensorboard.dev by running:
```bash script
tensorboard dev upload --logdir model \
    --name "phones-las experiment name" \                               # optional
    --description "phones-las experiment description"  # optional
```
Tensorboard.dev supports only scalar summaries, so attention images will not be there.

### Notes

In some cases setting `--n_jobs` in `preprocess_all.py` to values other than 1 may result in script being deadlocked.

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
