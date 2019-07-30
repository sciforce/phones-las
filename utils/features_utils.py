import tensorflow as tf
import joblib


def calculate_mfcc_op(sample_rate, coeffs, window, step, mels):
    def _mfcc_op(input_tensor):
        spectrograms = tf.contrib.signal.stft(input_tensor, frame_length=window, frame_step=step, fft_length=window)
        magnitude_spectrograms = tf.abs(spectrograms)
        num_spectrogram_bins = magnitude_spectrograms.shape[-1].value
        lower_edge_hertz, upper_edge_hertz = 80.0, 7600.0
        linear_to_mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(
            mels, num_spectrogram_bins, sample_rate, lower_edge_hertz, upper_edge_hertz)
        mel_spectrograms = tf.tensordot(magnitude_spectrograms, linear_to_mel_weight_matrix, 1)
        mel_spectrograms.set_shape(magnitude_spectrograms.shape[:-1].concatenate(
            linear_to_mel_weight_matrix.shape[-1:]))
        log_offset = 1e-6
        log_mel_spectrograms = tf.log(mel_spectrograms + log_offset)
        mfccs = tf.contrib.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)[..., :coeffs]
        return mfccs
    return _mfcc_op


def load_normalization(norm_path):
    with tf.io.gfile.GFile(norm_path, 'rb') as f:
        means, stds = joblib.load(f)
    return means, stds
