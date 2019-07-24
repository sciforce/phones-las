import tensorflow as tf
import tensorflow.contrib as tf_contrib

from tensorflow.python.util import nest

from las.ops import lstm_cell
from las.ops import pyramidal_bilstm
from utils import TrainingSigmoidHelper, ScheduledSigmoidHelper, DenseBinfDecoder

__all__ = [
    'listener',
    'speller',
]


"""Reference: https://github.com/tensorflow/nmt/blob/master/nmt/gnmt_model.py"""


class AttentionMultiCell(tf.nn.rnn_cell.MultiRNNCell):
    """A MultiCell with attention style."""

    def __init__(self, attention_cell, cells, use_new_attention=False):
        """Creates a AttentionMultiCell.
        Args:
          attention_cell: An instance of AttentionWrapper.
          cells: A list of RNNCell wrapped with AttentionInputWrapper.
          use_new_attention: Whether to use the attention generated from current
            step bottom layer's output. Default is False.
        """
        cells = [attention_cell] + cells
        self.use_new_attention = use_new_attention
        super(AttentionMultiCell, self).__init__(
            cells, state_is_tuple=True)

    def __call__(self, inputs, state, scope=None):
        """Run the cell with bottom layer's attention copied to all upper layers."""
        if not nest.is_sequence(state):
            raise ValueError(
                "Expected state to be a tuple of length %d, but received: %s"
                % (len(self.state_size), state))

        with tf.variable_scope(scope or "multi_rnn_cell"):
            new_states = []

            with tf.variable_scope("cell_0_attention"):
                attention_cell = self._cells[0]
                attention_state = state[0]
                cur_inp, new_attention_state = attention_cell(
                    inputs, attention_state)
                new_states.append(new_attention_state)

            for i in range(1, len(self._cells)):
                with tf.variable_scope("cell_%d" % i):

                    cell = self._cells[i]
                    cur_state = state[i]

                    if self.use_new_attention:
                        cur_inp = tf.concat(
                            [cur_inp, new_attention_state.attention], -1)
                    else:
                        cur_inp = tf.concat(
                            [cur_inp, attention_state.attention], -1)

                    cur_inp, new_state = cell(cur_inp, cur_state)
                    new_states.append(new_state)

        return cur_inp, tuple(new_states)


class CustomAttention(tf_contrib.seq2seq.LuongAttention):
    def __init__(self,
                 num_units,
                 memory,
                 memory_sequence_length=None,
                 scale=False,
                 probability_fn=None,
                 score_mask_value=None,
                 dtype=None,
                 name="CustomAttention"):

        super(CustomAttention, self).__init__(
            num_units=num_units,
            memory=memory,
            memory_sequence_length=memory_sequence_length,
            scale=scale,
            probability_fn=probability_fn,
            score_mask_value=score_mask_value,
            dtype=dtype,
            name=name)

        self._query_layer = tf.layers.Dense(
            num_units, name='query_layer', use_bias=False, dtype=dtype)

        self._keys = tf.nn.relu(self.keys)

    def __call__(self, query, state):
        processed_query = tf.nn.relu(self.query_layer(query))

        return super(CustomAttention, self).__call__(processed_query, state)


def listener(encoder_inputs,
             source_sequence_length,
             mode,
             hparams):

    if hparams.use_pyramidal:
        return pyramidal_bilstm(encoder_inputs, source_sequence_length, mode, hparams)
    else:
        forward_cell_list, backward_cell_list = [], []
        for layer in range(hparams.num_layers):
            with tf.variable_scope('fw_cell_{}'.format(layer)):
                cell = lstm_cell(hparams.num_units, hparams.dropout, mode)

            forward_cell_list.append(cell)
            if not hparams.unidirectional:
                with tf.variable_scope('bw_cell_{}'.format(layer)):
                    cell = lstm_cell(hparams.num_units, hparams.dropout, mode)

                backward_cell_list.append(cell)

        forward_cell = tf.nn.rnn_cell.MultiRNNCell(forward_cell_list)

        if not hparams.unidirectional:
            backward_cell = tf.nn.rnn_cell.MultiRNNCell(backward_cell_list)
            encoder_outputs, encoder_state = tf.nn.bidirectional_dynamic_rnn(
                forward_cell,
                backward_cell,
                encoder_inputs,
                sequence_length=source_sequence_length,
                dtype=tf.float32)
            encoder_outputs = tf.concat(encoder_outputs, -1)
        else:
            encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
                forward_cell,
                encoder_inputs,
                sequence_length=source_sequence_length,
                dtype=tf.float32)

        return (encoder_outputs, source_sequence_length), encoder_state


def attend(encoder_outputs,
           source_sequence_length,
           mode,
           hparams):

    memory = encoder_outputs

    att_kwargs = {}
    if hparams.attention_type == 'luong':
        attention_fn = tf_contrib.seq2seq.LuongAttention
    elif hparams.attention_type == 'bahdanau':
        attention_fn = tf_contrib.seq2seq.BahdanauAttention
    elif hparams.attention_type == 'luong_monotonic':
        attention_fn = tf_contrib.seq2seq.LuongMonotonicAttention
    elif hparams.attention_type == 'bahdanau_monotonic':
        attention_fn = tf_contrib.seq2seq.BahdanauMonotonicAttention
        if mode == tf.estimator.ModeKeys.TRAIN:
            att_kwargs['sigmoid_noise'] = 1.0
        else:
            att_kwargs['mode'] = 'hard'
    elif hparams.attention_type == 'custom':
        attention_fn = CustomAttention

    attention_mechanism = attention_fn(
        hparams.num_units, memory, source_sequence_length, **att_kwargs)

    cell_list = []
    for layer in range(hparams.num_layers):
        with tf.variable_scope('decoder_cell_'.format(layer)):
            cell = lstm_cell(hparams.num_units, hparams.dropout, mode)

        cell_list.append(cell)

    alignment_history = (mode != tf.estimator.ModeKeys.TRAIN)

    attention_layer_size = hparams.binf_count if hparams.binf_projection else hparams.attention_layer_size

    if hparams.bottom_only:
        attention_cell = cell_list.pop(0)

        attention_cell = tf_contrib.seq2seq.AttentionWrapper(
            attention_cell, attention_mechanism,
            attention_layer_size=attention_layer_size,
            alignment_history=alignment_history)

        decoder_cell = AttentionMultiCell(attention_cell, cell_list)
    else:
        decoder_cell = tf.nn.rnn_cell.MultiRNNCell(cell_list)

        decoder_cell = tf_contrib.seq2seq.AttentionWrapper(
            decoder_cell, attention_mechanism,
            attention_layer_size=attention_layer_size,
            alignment_history=alignment_history)

    return decoder_cell


def speller(encoder_outputs,
            encoder_state,
            decoder_inputs,
            source_sequence_length,
            target_sequence_length,
            mode,
            hparams,
            binary_outputs=False,
            binf_embedding=None):

    batch_size = tf.shape(encoder_outputs)[0]
    beam_width = hparams.beam_width

    if mode == tf.estimator.ModeKeys.PREDICT and beam_width > 0:
        encoder_outputs = tf_contrib.seq2seq.tile_batch(
            encoder_outputs, multiplier=beam_width)
        source_sequence_length = tf_contrib.seq2seq.tile_batch(
            source_sequence_length, multiplier=beam_width)
        encoder_state = tf_contrib.seq2seq.tile_batch(
            encoder_state, multiplier=beam_width)
        batch_size = batch_size * beam_width

    def embedding_fn(ids):
        # pass callable object to avoid OOM when using one-hot encoding
        if hparams.embedding_size != 0:
            target_embedding = tf.get_variable(
                'target_embedding', [
                    hparams.target_vocab_size,
                    hparams.embedding_size],
                dtype=tf.float32, initializer=tf_contrib.layers.xavier_initializer())

            return tf.nn.embedding_lookup(target_embedding, ids)
        elif binary_outputs:
            if binf_embedding is None or mode == tf.estimator.ModeKeys.TRAIN:
                return tf.cast(ids, tf.float32)
            else:
                return tf.nn.embedding_lookup(tf.transpose(binf_embedding), ids)
        else:
            return tf.one_hot(ids, hparams.target_vocab_size)

    decoder_cell = attend(
        encoder_outputs, source_sequence_length, mode, hparams)

    projection_layer = DenseBinfDecoder(
        hparams.target_vocab_size if not binary_outputs else hparams.binf_count,
        binf_to_ipa=binf_embedding if mode != tf.estimator.ModeKeys.TRAIN or hparams.binf_projection else None,
        use_bias=True, name='projection_layer',
        inner_projection_layer=not hparams.binf_projection,
        kernel_initializer=tf.random_uniform_initializer(minval=-0.075, maxval=0.075))

    if hparams.pass_hidden_state and hparams.bottom_only:
        initial_state = tuple(
            zs.clone(cell_state=es)
            if isinstance(zs, tf_contrib.seq2seq.AttentionWrapperState) else es
            for zs, es in zip(
                decoder_cell.zero_state(batch_size, tf.float32), encoder_state))
    else:
        initial_state = decoder_cell.zero_state(batch_size, tf.float32)

    maximum_iterations = None
    if mode != tf.estimator.ModeKeys.TRAIN:
        max_source_length = tf.reduce_max(source_sequence_length)
        maximum_iterations = tf.to_int32(tf.round(tf.to_float(
            max_source_length) * hparams.decoding_length_factor))

    if mode == tf.estimator.ModeKeys.TRAIN:
        decoder_inputs = embedding_fn(decoder_inputs)

        if hparams.sampling_probability > 0.0:
            if binary_outputs:
                helper = ScheduledSigmoidHelper(decoder_inputs, target_sequence_length,
                    embedding_fn, hparams.sampling_probability, binf_to_ipa=binf_embedding)
            else:
                helper = tf_contrib.seq2seq.ScheduledEmbeddingTrainingHelper(
                    decoder_inputs, target_sequence_length,
                    embedding_fn, hparams.sampling_probability)
        else:
            if binary_outputs:
                helper = TrainingSigmoidHelper(decoder_inputs, target_sequence_length,
                                               binf_to_ipa=binf_embedding)
            else:
                helper = tf_contrib.seq2seq.TrainingHelper(
                    decoder_inputs, target_sequence_length)
        decoder = tf_contrib.seq2seq.BasicDecoder(
            decoder_cell, helper, initial_state, output_layer=projection_layer)

    elif mode == tf.estimator.ModeKeys.PREDICT and beam_width > 0:
        start_tokens = tf.fill(
            [tf.div(batch_size, beam_width)], hparams.sos_id)

        decoder = tf_contrib.seq2seq.BeamSearchDecoder(
            cell=decoder_cell,
            embedding=embedding_fn,
            start_tokens=start_tokens,
            end_token=hparams.eos_id,
            initial_state=initial_state,
            beam_width=beam_width,
            output_layer=projection_layer)
    elif binary_outputs and binf_embedding is None:
        start_inputs = tf.concat(
            (tf.zeros([batch_size, hparams.binf_count - 2]),
             tf.ones([batch_size, 1]),
             tf.zeros([batch_size, 1])),
             axis=1
            )
        helper = tf_contrib.seq2seq.InferenceHelper(
            sample_fn=lambda x: tf.round(tf.sigmoid(x)),
            # sample_fn=lambda x: tf.sigmoid(x),    #TODO: experiment with non-binarlized outputs
            sample_shape=[hparams.binf_count],
            sample_dtype=tf.float32,
            next_inputs_fn=embedding_fn,
            start_inputs=start_inputs,
            end_fn=lambda x: tf.greater(x[..., -1], 0.5))
        decoder = tf_contrib.seq2seq.BasicDecoder(
            decoder_cell, helper, initial_state, output_layer=projection_layer)
    else:
        start_tokens = tf.fill([batch_size], hparams.sos_id)

        helper = tf_contrib.seq2seq.GreedyEmbeddingHelper(
            embedding_fn, start_tokens, hparams.eos_id)

        decoder = tf_contrib.seq2seq.BasicDecoder(
            decoder_cell, helper, initial_state, output_layer=projection_layer)

    decoder_outputs, final_context_state, final_sequence_length = tf_contrib.seq2seq.dynamic_decode(
        decoder, maximum_iterations=maximum_iterations)

    return decoder_outputs, final_context_state, final_sequence_length
