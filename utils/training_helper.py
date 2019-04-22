import tensorflow as tf
import tensorflow.contrib as tf_contrib

__all__ = [
    'TrainingSigmoidHelper',
    'ScheduledSigmoidHelper',
    'DenseBinfDecoder',
    'transform_binf_to_phones'
]

def transform_binf_to_phones(outputs, binf_to_ipa):
    # Transform binary features logits to phone log probabilities (unnormalized)
    log_prob_ones = -tf.log(1 + tf.exp(-outputs))
    log_prob_ones = tf.where(tf.is_inf(log_prob_ones), tf.zeros_like(log_prob_ones) - 1e6, log_prob_ones)
    log_prob_zeros = -outputs - log_prob_ones
    if outputs.shape.ndims == 3:
        binf_to_ipa_tiled = tf.tile(binf_to_ipa[None, :, :], [tf.shape(outputs)[0], 1, 1])
    else:
        binf_to_ipa_tiled = binf_to_ipa
    outputs = tf.matmul(log_prob_ones, binf_to_ipa_tiled) + tf.matmul(log_prob_zeros, 1 - binf_to_ipa_tiled)
    return outputs


class TrainingSigmoidHelper(tf_contrib.seq2seq.TrainingHelper):
    def __init__(self, inputs, sequence_length, time_major=False, name=None,
                 binf_to_ipa=None):
        self.binf_to_ipa = binf_to_ipa
        super().__init__(inputs, sequence_length, time_major, name)

    def sample(self, time, outputs, name=None, **unused_kwargs):
        with tf.name_scope(name, "TrainingSigmoidHelperSample", [time, outputs]):
            if self.binf_to_ipa is not None:
                # Convert binary features vector to one corresponding to the closest phoneme
                ids = tf.argmax(transform_binf_to_phones(outputs, self.binf_to_ipa), -1)
                sample_ids = tf.nn.embedding_lookup(tf.transpose(self.binf_to_ipa), ids)
            else:
                sample_ids = tf.round(tf.sigmoid(outputs))
                # sample_ids = tf.sigmoid(outputs) #TODO: experiment with non-binarlized outputs
            return sample_ids

class ScheduledSigmoidHelper(tf_contrib.seq2seq.ScheduledEmbeddingTrainingHelper):
    def __init__(self, inputs, sequence_length, embedding, sampling_probability,
                 time_major=False, seed=None, scheduling_seed=None, name=None,
                 binf_to_ipa=None):
        self.binf_to_ipa = binf_to_ipa
        super().__init__(inputs, sequence_length, embedding, sampling_probability,
                         time_major, seed, scheduling_seed, name)

    def sample(self, time, outputs, state, name=None):
        # Return -1s where we did not sample, and sample_ids elsewhere
        select_sampler = tf.distributions.Bernoulli(
            probs=self._sampling_probability, dtype=tf.bool)
        select_sample = select_sampler.sample(
            sample_shape=self.batch_size, seed=self._scheduling_seed)
        if self.binf_to_ipa is None:
            sample_id_sampler = tf.distributions.Bernoulli(logits=outputs)
            samples = tf.cast(sample_id_sampler.sample(seed=self._seed), tf.float32)
            # samples = tf.round(tf.sigmoid(outputs))
        else:
            # Convert binary features vector to one corresponding to the closest phoneme
            ids = tf.argmax(transform_binf_to_phones(outputs, self.binf_to_ipa), -1)
            samples = tf.nn.embedding_lookup(tf.transpose(self.binf_to_ipa), ids)
            # logits = transform_binf_to_phones(outputs, self.binf_to_ipa)
            # sample_id_sampler = tf.distributions.Categorical(logits=logits)
            # ids = sample_id_sampler.sample(seed=self._seed)
            # samples = tf.nn.embedding_lookup(tf.transpose(self.binf_to_ipa), ids)
        return tf.where(
            select_sample,
            samples,
            # tf.sigmoid(outputs), #TODO: experiment with non-binarlized outputs
            tf.fill(tf.shape(outputs), -1.0))

class DenseBinfDecoder(tf.layers.Dense):
    '''
    Fully connected layer modification, which transforms
    original layer's outputs, assumed to be binary features logits,
    to phonemes logits.
    '''
    def __init__(self, units, binf_to_ipa=None, **kwargs):
        self.binf_to_ipa = binf_to_ipa
        super().__init__(units, **kwargs)

    def call(self, inputs):
        outputs = super().call(inputs)
        if self.binf_to_ipa is not None:
            outputs = transform_binf_to_phones(outputs, self.binf_to_ipa)
        return outputs

    def compute_output_shape(self, input_shape):
        out = super().compute_output_shape(input_shape)
        if self.binf_to_ipa is not None:
            out = out[:-1].concatenate(self.binf_to_ipa.shape[-1])
        return out