import tensorflow as tf
from tensorflow.nn.rnn_cell import LSTMStateTuple

from las.ops import lstm_cell


def reader(encoder_inputs,
           source_sequence_length,
           mode,
           hparams, target_vocab_size):
    encoder_features = tf.one_hot(encoder_inputs, target_vocab_size)

    forward_cell_list, backward_cell_list = [], []
    for layer in range(hparams.num_layers):
        with tf.variable_scope('fw_cell_{}'.format(layer)):
            cell = lstm_cell(hparams.num_units, hparams.dropout, mode)

        forward_cell_list.append(cell)

        with tf.variable_scope('bw_cell_{}'.format(layer)):
            cell = lstm_cell(hparams.num_units, hparams.dropout, mode)

        backward_cell_list.append(cell)

    forward_cell = tf.nn.rnn_cell.MultiRNNCell(forward_cell_list)
    backward_cell = tf.nn.rnn_cell.MultiRNNCell(backward_cell_list)

    encoder_outputs, encoder_state = tf.nn.bidirectional_dynamic_rnn(
        forward_cell,
        backward_cell,
        encoder_features,
        sequence_length=source_sequence_length,
        dtype=tf.float32)

    encoder_outputs = tf.concat(encoder_outputs, -1)

    return (encoder_outputs, source_sequence_length), encoder_state


def speller(encoder_outputs,
            encoder_state,
            decoder_inputs,
            source_sequence_length,
            target_sequence_length,
            mode,
            hparams):

    batch_size = tf.shape(encoder_outputs)[0]
    beam_width = hparams.beam_width

    if mode == tf.estimator.ModeKeys.PREDICT and beam_width > 0:
        source_sequence_length = tf.contrib.seq2seq.tile_batch(
            source_sequence_length, multiplier=beam_width)
        encoder_state = tf.contrib.seq2seq.tile_batch(
            encoder_state, multiplier=beam_width)
        batch_size = batch_size * beam_width

    def embedding_fn(ids):
        # pass callable object to avoid OOM when using one-hot encoding
        if hparams.embedding_size != 0:
            target_embedding = tf.get_variable(
                'target_embedding', [
                    hparams.target_vocab_size, hparams.embedding_size],
                dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())

            return tf.nn.embedding_lookup(target_embedding, ids)
        else:
            return tf.one_hot(ids, hparams.target_vocab_size)

    cell_list = []
    for layer in range(hparams.num_layers):
        with tf.variable_scope('decoder_cell_'.format(layer)):
            cell = lstm_cell(hparams.num_units * 2, hparams.dropout, mode)
        cell_list.append(cell)
    decoder_cell = tf.nn.rnn_cell.MultiRNNCell(cell_list)

    projection_layer = tf.layers.Dense(
        hparams.target_vocab_size, use_bias=True, name='projection_layer')

    initial_state = tuple([LSTMStateTuple(c=tf.concat([es[0].c, es[1].c], axis=-1),
                                          h=tf.concat([es[0].h, es[1].h], axis=-1))
                           for es in encoder_state[-hparams.num_layers:]])

    maximum_iterations = None
    if mode != tf.estimator.ModeKeys.TRAIN:
        max_source_length = tf.reduce_max(source_sequence_length)
        maximum_iterations = tf.to_int32(tf.round(tf.to_float(
            max_source_length) * hparams.decoding_length_factor))

    if mode == tf.estimator.ModeKeys.TRAIN:
        decoder_inputs = embedding_fn(decoder_inputs)

        if hparams.sampling_probability > 0.0:
            helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(
                decoder_inputs, target_sequence_length,
                embedding_fn, hparams.sampling_probability)
        else:
            helper = tf.contrib.seq2seq.TrainingHelper(
                decoder_inputs, target_sequence_length)

        decoder = tf.contrib.seq2seq.BasicDecoder(
            decoder_cell, helper, initial_state, output_layer=projection_layer)

    elif mode == tf.estimator.ModeKeys.PREDICT and beam_width > 0:
        start_tokens = tf.fill(
            [tf.div(batch_size, beam_width)], hparams.sos_id)

        decoder = tf.contrib.seq2seq.BeamSearchDecoder(
            cell=decoder_cell,
            embedding=embedding_fn,
            start_tokens=start_tokens,
            end_token=hparams.eos_id,
            initial_state=initial_state,
            beam_width=beam_width,
            output_layer=projection_layer)
    else:
        start_tokens = tf.fill([batch_size], hparams.sos_id)

        helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
            embedding_fn, start_tokens, hparams.eos_id)

        decoder = tf.contrib.seq2seq.BasicDecoder(
            decoder_cell, helper, initial_state, output_layer=projection_layer)

    decoder_outputs, final_context_state, final_sequence_length = tf.contrib.seq2seq.dynamic_decode(
        decoder, maximum_iterations=maximum_iterations)

    return decoder_outputs, final_context_state, final_sequence_length
