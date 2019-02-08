import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib as tf_contrib

import las
import utils
import text_ae.model

__all__ = [
    'las_model_fn',
]


GRAD_NORM = 2
NOISE_MEAN = 0.0
NOISE_STD = 0.5
ADD_NOISE_STEP = 2000


def compute_loss(logits, targets, final_sequence_length, target_sequence_length, mode):

    assert mode != tf.estimator.ModeKeys.PREDICT

    if mode == tf.estimator.ModeKeys.TRAIN:
        target_weights = tf.sequence_mask(
            target_sequence_length, dtype=tf.float32)
        loss = tf_contrib.seq2seq.sequence_loss(
            logits, targets, target_weights)
    else:
        '''
        # Reference: https://github.com/tensorflow/nmt/issues/2
        # Note that this method always trim the tensor with larger length to shorter one, 
        # and I think it is unfair. 
        # Consider targets = [[3, 3, 2]], and logits with shape [1, 2, VOCAB_SIZE]. 
        # This method will trim targets to [[3, 3]] and compute sequence_loss on new targets and logits.
        # However, I think the truth is that the model predicts less word than ground truth does,
        # and hence, both targets and logits should be padded to the same sequence length (dimension 1)
        # to compute loss.

        current_sequence_length = tf.to_int32(
            tf.minimum(tf.shape(targets)[1], tf.shape(logits)[1]))
        targets = tf.slice(targets, begin=[0, 0],
                           size=[-1, current_sequence_length])
        logits = tf.slice(logits, begin=[0, 0, 0],
                          size=[-1, current_sequence_length, -1])
        target_weights = tf.sequence_mask(
            target_sequence_length, maxlen=current_sequence_length, dtype=tf.float32)
        loss = tf.contrib.seq2seq.sequence_loss(
            logits, targets, target_weights)
        '''

        max_ts = tf.reduce_max(target_sequence_length)
        max_fs = tf.reduce_max(final_sequence_length)

        max_sequence_length = tf.to_int32(
            tf.maximum(max_ts, max_fs))

        logits = tf.slice(logits, begin=[0, 0, 0], size=[-1, max_fs, -1])

        # pad EOS to make targets and logits have same shape
        targets = tf.pad(targets, [[0, 0], [0, tf.maximum(
            0, max_sequence_length - tf.shape(targets)[1])]], constant_values=utils.EOS_ID)
        logits = tf.pad(logits, [[0, 0], [0, tf.maximum(
            0, max_sequence_length - tf.shape(logits)[1])], [0, 0]], constant_values=0)

        # find larger length between predictions and targets
        sequence_length = tf.reduce_max(
            [target_sequence_length, final_sequence_length], 0)

        target_weights = tf.sequence_mask(
            sequence_length, maxlen=max_sequence_length, dtype=tf.float32)

        loss = tf_contrib.seq2seq.sequence_loss(
            logits, targets, target_weights)

    return loss


def sequence_loss_sigmoid(logits, targets, weights):
    with tf.name_scope(name="sequence_loss",
                       values=[logits, targets, weights]):
        num_classes = tf.shape(logits)[2]
        logits_flat = tf.reshape(logits, [-1, num_classes])
        targets = tf.cast(tf.reshape(targets, [-1, num_classes]), tf.float32)
        crossent = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=targets, logits=logits_flat), axis=1)

        crossent *= tf.reshape(weights, [-1])
        crossent = tf.reduce_sum(crossent)
        total_size = tf.reduce_sum(weights)
        total_size += 1e-12  # to avoid division by 0 for all-0 weights
        crossent /= total_size
    return crossent


def compute_loss_sigmoid(logits, targets, final_sequence_length, target_sequence_length, mode):

    assert mode != tf.estimator.ModeKeys.PREDICT

    if mode == tf.estimator.ModeKeys.TRAIN:
        target_weights = tf.sequence_mask(
            target_sequence_length, dtype=tf.float32)
        loss = sequence_loss_sigmoid(logits, targets, target_weights)
    else:
        max_ts = tf.reduce_max(target_sequence_length)
        max_fs = tf.reduce_max(final_sequence_length)

        max_sequence_length = tf.to_int32(
            tf.maximum(max_ts, max_fs))

        logits = tf.slice(logits, begin=[0, 0, 0], size=[-1, max_fs, -1])

        # pad EOS to make targets and logits have same shape
        targets = tf.pad(targets, [[0, 0], [0, tf.maximum(
            0, max_sequence_length - tf.shape(targets)[1])], [0, 0]], constant_values=0)
        logits = tf.pad(logits, [[0, 0], [0, tf.maximum(
            0, max_sequence_length - tf.shape(logits)[1])], [0, 0]], constant_values=0)

        # find larger length between predictions and targets
        sequence_length = tf.reduce_max(
            [target_sequence_length, final_sequence_length], 0)

        target_weights = tf.sequence_mask(
            sequence_length, maxlen=max_sequence_length, dtype=tf.float32)

        loss = sequence_loss_sigmoid(logits, targets, target_weights)

    return loss


def compute_emb_loss(encoder_state, reader_encoder_state):
    try:
        emb_loss = 0
        for enc_s, enc_r in zip(encoder_state, reader_encoder_state):
            emb_loss += tf.losses.mean_squared_error(enc_s.c, enc_r.c)
            emb_loss += tf.losses.mean_squared_error(enc_s.h, enc_r.h)
    except AttributeError:
        emb_loss = 0
        for enc_s, enc_r in zip(encoder_state, reader_encoder_state):
            emb_loss += tf.losses.mean_squared_error(enc_s.c, enc_r[-1].c)
            emb_loss += tf.losses.mean_squared_error(enc_s.h, enc_r[-1].h)
    return emb_loss


def las_model_fn(features,
                 labels,
                 mode,
                 config,
                 params,
                 binf2phone=None,
                 run_name=None):
    if tf.estimator.ModeKeys.PREDICT == mode:
        params.use_text = False

    encoder_inputs = features['encoder_inputs']
    source_sequence_length = features['source_sequence_length']

    decoder_inputs = None
    targets = None
    target_sequence_length = None

    binf_embedding = None
    if binf2phone is not None and params.decoder.binary_outputs:
        binf_embedding = tf.constant(binf2phone, dtype=tf.float32, name='binf2phone')
    is_binf_outputs = params.decoder.binary_outputs and (
        binf_embedding is None or mode == tf.estimator.ModeKeys.TRAIN)

    if mode != tf.estimator.ModeKeys.PREDICT:
        decoder_inputs = labels['targets_inputs']
        targets = labels['targets_outputs']
        target_sequence_length = labels['target_sequence_length']

    text_loss = 0
    text_edit_distance = reader_encoder_state = None
    if params.use_text:
        tf.logging.info('Building reader')

        with tf.variable_scope('reader'):
            (reader_encoder_outputs, reader_source_sequence_length), reader_encoder_state = text_ae.model.reader(
                decoder_inputs, target_sequence_length, mode,
                params.encoder, params.decoder.target_vocab_size)

        tf.logging.info('Building writer')

        with tf.variable_scope('writer'):
            writer_decoder_outputs, writer_final_context_state, writer_final_sequence_length = text_ae.model.speller(
                reader_encoder_outputs, reader_encoder_state, decoder_inputs,
                reader_source_sequence_length, target_sequence_length,
                mode, params.decoder)

        with tf.name_scope('text_prediciton'):
            logits = writer_decoder_outputs.rnn_output
            sample_ids = tf.to_int32(tf.argmax(logits, -1))

        with tf.name_scope('text_metrics'):
            text_edit_distance = utils.edit_distance(
                sample_ids, targets, utils.EOS_ID, params.mapping)

            metrics = {
                'text_edit_distance': tf.metrics.mean(text_edit_distance),
            }

        tf.summary.scalar('text_edit_distance', metrics['text_edit_distance'][1])

        with tf.name_scope('text_cross_entropy'):
            text_loss = compute_loss(
                logits, targets, writer_final_sequence_length, target_sequence_length, mode)

    tf.logging.info('Building listener')

    with tf.variable_scope('listener'):
        (encoder_outputs, source_sequence_length), encoder_state = las.model.listener(
            encoder_inputs, source_sequence_length, mode, params.encoder)

    tf.logging.info('Building speller')

    with tf.variable_scope('speller'):
        decoder_outputs, final_context_state, final_sequence_length = las.model.speller(
            encoder_outputs, encoder_state, decoder_inputs,
            source_sequence_length, target_sequence_length,
            mode, params.decoder, binf_embedding)

    with tf.name_scope('prediction'):
        if mode == tf.estimator.ModeKeys.PREDICT and params.decoder.beam_width > 0:
            logits = tf.no_op()
            sample_ids = decoder_outputs.predicted_ids
        else:
            logits = decoder_outputs.rnn_output
            if is_binf_outputs:
                sample_ids = tf.to_int32(tf.round(tf.sigmoid(logits)))
            else:
                sample_ids = tf.to_int32(tf.argmax(logits, -1))

    if mode == tf.estimator.ModeKeys.PREDICT:
        emb_c = tf.concat([x.c for x in encoder_state], axis=1)
        emb_h = tf.concat([x.h for x in encoder_state], axis=1)
        emb = tf.stack([emb_c, emb_h], axis=1)
        predictions = {
            'sample_ids': sample_ids,
            'embedding': emb,
            'encoder_out': encoder_outputs,
            'source_length': source_sequence_length
        }
        try:
            predictions['alignment'] = tf.transpose(final_context_state.alignment_history.stack(), perm=[1, 0, 2])
        except AttributeError:
            if not isinstance(final_context_state.cell_state, tuple):
                alignment_history = tf.transpose(final_context_state.cell_state.alignment_history, perm=[1, 0, 2])
            else:
                alignment_history = tf.transpose(final_context_state.cell_state[-1].alignment_history, perm=[1, 0, 2])
            shape = tf.shape(alignment_history)
            predictions['alignment'] = tf.reshape(alignment_history,
                [-1, params.decoder.beam_width, shape[1], shape[2]])
        if params.decoder.beam_width == 0:
            if params.decoder.binary_outputs and binf_embedding is None:
                predictions['probs'] = tf.nn.sigmoid(logits)
            else:
                predictions['probs'] = tf.nn.softmax(logits)

        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    metrics = None

    if binf_embedding is not None and not is_binf_outputs:
        binf_to_ipa_tiled = tf.cast(
            tf.tile(binf_embedding[None, :, :], [tf.shape(targets)[0], 1, 1]), tf.int32)
        targets_transformed = tf.cast(
            tf.argmax(tf.matmul(targets, binf_to_ipa_tiled) + tf.matmul(1 - targets, 1 - binf_to_ipa_tiled), -1), tf.int32)
    else:
        targets_transformed = targets

    if not is_binf_outputs:
        with tf.name_scope('metrics'):
            edit_distance = utils.edit_distance(
                sample_ids, targets_transformed, utils.EOS_ID, params.mapping)

            metrics = {
                'edit_distance': tf.metrics.mean(edit_distance),
            }
        if params.use_text and not params.emb_loss:
            pass
        else:
            # In TRAIN model this becomes an significantly affected by early high values.
            # As a result in summaries train values would be high and drop after restart.
            # To prevent this, we use last batch average in case of TRAIN.
            if mode != tf.estimator.ModeKeys.TRAIN:
                tf.summary.scalar('edit_distance', metrics['edit_distance'][1])
            else:
                tf.summary.scalar('edit_distance', tf.reduce_mean(edit_distance))
    else:
        edit_distance = None

    with tf.name_scope('cross_entropy'):
        loss_fn = compute_loss_sigmoid if is_binf_outputs else compute_loss
        audio_loss = loss_fn(
            logits, targets_transformed, final_sequence_length, target_sequence_length, mode)

    emb_loss = 0
    if params.use_text:
        with tf.name_scope('embeddings_loss'):
            emb_loss = compute_emb_loss(encoder_state, reader_encoder_state)

    if mode == tf.estimator.ModeKeys.EVAL:
        with tf.name_scope('alignment'):
            attention_images = utils.create_attention_images(
                final_context_state)

        if params.use_text and not params.emb_loss:
            hooks = []
            loss = text_loss
        else:
            run_name = run_name or 'eval'
            if run_name != 'eval':
                # For other summaries eval is automatically added.
                run_name = 'eval_{}'.format(run_name)
            attention_summary = tf.summary.image(
                'attention_images', attention_images)
            eval_summary_hook = tf.train.SummarySaverHook(
                save_steps=20,
                output_dir=os.path.join(config.model_dir, run_name),
                summary_op=attention_summary)
            hooks = [eval_summary_hook]
            loss = audio_loss
        if not is_binf_outputs:
            log_data = {
                'edit_distance': tf.reduce_mean(edit_distance),
                'max_edit_distance': tf.reduce_max(edit_distance),
                'min_edit_distance': tf.reduce_min(edit_distance)
            }
            if params.use_text:
                if not params.emb_loss:
                    log_data = {}
                else:
                    log_data['emb_loss'] = tf.reduce_mean(emb_loss)
                log_data['text_edit_distance'] = tf.reduce_mean(text_edit_distance)
            logging_hook = tf.train.LoggingTensorHook(log_data, every_n_iter=20)
            hooks += [logging_hook]

        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics,
                                          evaluation_hooks=hooks)

    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(params.learning_rate)
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        if params.use_text:
            audio_var_list = [x for x in var_list
                              if not x.name.startswith('reader') and not x.name.startswith('writer')]
            total_params = np.sum([np.prod(x.shape.as_list()) for x in audio_var_list])
            tf.logging.info('Trainable audio parameters: {}'.format(total_params))
            text_var_list = [x for x in var_list
                             if not x.name.startswith('listener') and not x.name.startswith('speller')]
            total_params = np.sum([np.prod(x.shape.as_list()) for x in text_var_list])
            tf.logging.info('Trainable text parameters: {}'.format(total_params))
            gvs = optimizer.compute_gradients(audio_loss, var_list=audio_var_list)
            capped_gvs = [(tf.clip_by_norm(grad, GRAD_NORM), var) for grad, var in gvs]
            audio_train_op = optimizer.apply_gradients(capped_gvs, global_step=tf.train.get_global_step())
            gvs = optimizer.compute_gradients(text_loss, var_list=text_var_list)
            capped_gvs = [(tf.clip_by_norm(grad, GRAD_NORM), var) for grad, var in gvs]
            text_train_op = optimizer.apply_gradients(capped_gvs, global_step=tf.train.get_global_step())
            gvs = optimizer.compute_gradients(emb_loss, var_list=audio_var_list)
            capped_gvs = [(tf.clip_by_norm(grad, GRAD_NORM), var) for grad, var in gvs]
            emb_train_op = optimizer.apply_gradients(capped_gvs, global_step=tf.train.get_global_step())
            if not params.text_loss:
                tf.logging.info('Removing reader and writer from optimization.')
                train_op = tf.group(audio_train_op, emb_train_op)
            elif not params.emb_loss:
                tf.logging.info('Removing listener and speller from optimization params.')
                train_op = text_train_op
            else:
                raise ValueError('Either text_loss or emb_loss must be set with use_text!')
        else:
            total_params = np.sum([np.prod(x.shape.as_list()) for x in var_list])
            tf.logging.info('Trainable parameters: {}'.format(total_params))

            regularizer = tf_contrib.layers.l2_regularizer(params.l2_reg_scale)
            reg_term = tf.contrib.layers.apply_regularization(regularizer, var_list)
            audio_loss = audio_loss + reg_term

            gvs = optimizer.compute_gradients(audio_loss, var_list=var_list)
            capped_gvs = [(tf.clip_by_norm(grad, GRAD_NORM), var) for grad, var in gvs]
            train_op = optimizer.apply_gradients(capped_gvs, global_step=tf.train.get_global_step())
            if params.add_noise:
                def add_noise():
                    noise_ops = [train_op]
                    for var in var_list:
                        if var.name.endswith('kernel:0'):
                            shape = tf.shape(var)
                            noise_ops.append(tf.random_normal(shape, NOISE_MEAN, NOISE_STD, dtype=tf.float32))
                    return tf.group(*noise_ops)
                train_op = tf.cond(
                    tf.logical_and(tf.equal(tf.mod(tf.train.get_global_step(), ADD_NOISE_STEP), 0),
                        tf.greater(tf.train.get_global_step(), 0)),
                    add_noise,
                    lambda: train_op)

    loss = text_loss if params.use_text and not params.emb_loss else audio_loss
    train_log_data = {
        'loss': loss
    }
    if not is_binf_outputs:
        if params.use_text:
            if params.emb_loss:
                train_log_data['edit_distance'] = tf.reduce_mean(edit_distance)
                train_log_data['emb_loss'] = tf.reduce_mean(emb_loss)
            train_log_data['text_edit_distance'] = tf.reduce_mean(text_edit_distance)
        else:
            train_log_data['edit_distance'] = tf.reduce_mean(edit_distance)
    logging_hook = tf.train.LoggingTensorHook(train_log_data, every_n_iter=10)

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, training_hooks=[logging_hook])
