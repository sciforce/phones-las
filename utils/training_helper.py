import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
import tensorflow.contrib as tf_contrib

__all__ = [
    'TrainingSigmoidHelper',
    'ScheduledSigmoidHelper',
    'TPUScheduledEmbeddingTrainingHelper',
    'DenseBinfDecoder',
    'transform_binf_to_phones',
    'decoders_factory'
]


def transform_binf_to_phones(outputs, binf_to_ipa):
    # Transform binary features logits to phone log probabilities (unnormalized)
    nfeatures = binf_to_ipa.shape[0]
    log_prob_ones = outputs[..., :nfeatures]
    log_prob_zeros = outputs[..., nfeatures:2 * nfeatures]
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


class TPUScheduledEmbeddingTrainingHelper(tf_contrib.seq2seq.ScheduledEmbeddingTrainingHelper):
    def __init__(self, inputs, sequence_length, embedding, sampling_probability,
                 time_major=False, seed=None, scheduling_seed=None, name=None,
                 outputs_count=False):
        self.outputs_count = outputs_count
        super().__init__(inputs, sequence_length, embedding, sampling_probability,
                         time_major, seed, scheduling_seed, name)

    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        with ops.name_scope(name, "ScheduledEmbeddingTrainingHelperNextInputs",
                            [time, outputs, state, sample_ids]):
            (finished, base_next_inputs, state) = (
                super(tf_contrib.seq2seq.ScheduledEmbeddingTrainingHelper, self).next_inputs(
                    time=time,
                    outputs=outputs,
                    state=state,
                    sample_ids=sample_ids,
                    name=name))

            def maybe_sample():
                """Perform scheduled sampling."""
                sampling_mask = math_ops.cast(sample_ids > -1, base_next_inputs.dtype)
                # Embedding lookup might fail for negative samples.
                outputs_sampled = self._embedding_fn(sample_ids + tf.cast(1 - sampling_mask, tf.int32))
                sampling_mask = tf.expand_dims(sampling_mask, axis=-1)
                outputs = sampling_mask * outputs_sampled + (1 - sampling_mask) * base_next_inputs
                return outputs

            all_finished = math_ops.reduce_all(finished)
            next_inputs = control_flow_ops.cond(
                all_finished, lambda: base_next_inputs, maybe_sample)
            return finished, next_inputs, state

    def sample(self, time, outputs, state, name=None):
        if self.outputs_count is not None:
            outputs_ = outputs[..., :self.outputs_count]
        else:
            outputs_ = outputs

        return super().sample(time, outputs_, state, name)

class ScheduledSigmoidHelper(TPUScheduledEmbeddingTrainingHelper):
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
    """
    Fully connected layer modification, which transforms
    original layer's outputs, assumed to be binary features logits,
    to phonemes logits.
    """
    def __init__(self, units, binf_to_ipa=None, inner_projection_layer=True,
                 concat_cell_outputs=False, **kwargs):
        self.binf_to_ipa = binf_to_ipa
        self.inner_projection_layer = inner_projection_layer
        self.concat_cell_outputs = concat_cell_outputs
        super().__init__(units, **kwargs)

    def call(self, inputs):
        if self.inner_projection_layer:
            outputs = super(DenseBinfDecoder, self).call(inputs)
        else:
            outputs = inputs
        if self.binf_to_ipa is not None:
            outputs = transform_binf_to_phones(outputs, self.binf_to_ipa)
        if self.concat_cell_outputs:
            outputs = tf.concat((outputs, inputs), axis=-1)
        return outputs

    def compute_output_shape(self, input_shape):
        out = super().compute_output_shape(input_shape)
        if self.binf_to_ipa is not None:
            updated_dim_size = self.binf_to_ipa.shape[-1]
            if self.concat_cell_outputs:
                updated_dim_size += input_shape[-1]
            out = out[:-1].concatenate(updated_dim_size)
        return out


class BasicTransparentProjectionDecoder(tf_contrib.seq2seq.BasicDecoder):
    """
    Runs as BasicDecoder, but outputs are returned as is, without going through projection.
    """
    def __init__(self, cell, helper, initial_state, output_layer):
        super().__init__(cell, helper, initial_state, output_layer)

    def _rnn_output_size(self):
        return self._cell.output_size

    def step(self, time, inputs, state, name=None):
        with ops.name_scope(name, "BasicTransparentProjectionDecoderStep", (time, inputs, state)):
            raw_cell_outputs, cell_state = self._cell(inputs, state)
            cell_outputs = self._output_layer(raw_cell_outputs)
            sample_ids = self._helper.sample(
                time=time, outputs=cell_outputs, state=cell_state)
            (finished, next_inputs, next_state) = self._helper.next_inputs(
                time=time,
                outputs=cell_outputs,
                state=cell_state,
                sample_ids=sample_ids)
        outputs = tf_contrib.seq2seq.BasicDecoderOutput(raw_cell_outputs, sample_ids)
        return outputs, next_state, next_inputs, finished


def decoders_factory(decoder_str):
    if decoder_str == 'basic':
        return tf_contrib.seq2seq.BasicDecoder
    elif decoder_str == 'basic_transparent':
        return BasicTransparentProjectionDecoder
    else:
        raise ValueError('Unknown decoder.')
