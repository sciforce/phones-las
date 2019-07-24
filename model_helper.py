import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib as tf_contrib

import las
import utils

from utils.training_helper import transform_binf_to_phones

__all__ = [
    'las_model_fn',
]


GRAD_NORM = 2
NOISE_MEAN = 0.0


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


def get_alignment_history(final_context_state, params):
    try:
        res = tf.transpose(final_context_state.alignment_history.stack(), perm=[1, 0, 2])
    except AttributeError:
        # if not isinstance(final_context_state.cell_state, tuple):
        if hasattr(final_context_state.cell_state, 'alignment_history'):
            alignment_history = tf.transpose(final_context_state.cell_state.alignment_history, perm=[1, 0, 2])
        else:
            alignment_history = tf.transpose(final_context_state.cell_state[-1].alignment_history, perm=[1, 0, 2])
        shape = tf.shape(alignment_history)
        res = tf.reshape(alignment_history, [-1, params.decoder.beam_width, shape[1], shape[2]])
    return res


def las_model_fn(features,
                 labels,
                 mode,
                 config,
                 params,
                 binf2phone=None,
                 run_name=None):
    encoder_inputs = features['encoder_inputs']
    source_sequence_length = features['source_sequence_length']

    decoder_inputs, decoder_inputs_binf = None, None
    targets = None
    target_sequence_length = None
    targets_binf = None
    binf_embedding = None
    if binf2phone is not None and params.decoder.binary_outputs:
        binf_embedding = tf.constant(binf2phone, dtype=tf.float32, name='binf2phone')

    mapping = None
    if params.mapping and binf_embedding is not None:
        mapping = tf.convert_to_tensor(params.mapping)

    if mode != tf.estimator.ModeKeys.PREDICT:
        decoder_inputs = labels['targets_inputs']
        targets = labels['targets_outputs']
        if mapping is not None:
            decoder_inputs = tf.nn.embedding_lookup(mapping, decoder_inputs)
            targets = tf.nn.embedding_lookup(mapping, targets)
        target_sequence_length = labels['target_sequence_length']
        if binf_embedding is not None:
            targets_binf = tf.nn.embedding_lookup(tf.transpose(binf_embedding), targets)
            decoder_inputs_binf = tf.nn.embedding_lookup(tf.transpose(binf_embedding), decoder_inputs)

    tf.logging.info('Building listener')
    with tf.variable_scope('listener'):
        (encoder_outputs, source_sequence_length), encoder_state = las.model.listener(
            encoder_inputs, source_sequence_length, mode, params.encoder)

    tf.logging.info('Building speller')
    decoder_outputs, final_context_state, final_sequence_length = None, None, None
    if not params.decoder.binary_outputs or params.decoder.multitask:
        with tf.variable_scope('speller'):
            decoder_outputs, final_context_state, final_sequence_length = las.model.speller(
                encoder_outputs, encoder_state, decoder_inputs,
                source_sequence_length, target_sequence_length,
                mode, params.decoder)

    decoder_outputs_binf, final_context_state_binf, final_sequence_length_binf = None, None, None
    if params.decoder.binary_outputs:
        with tf.variable_scope('speller_binf'):
            decoder_outputs_binf, final_context_state_binf, final_sequence_length_binf = las.model.speller(
                encoder_outputs, encoder_state, decoder_inputs_binf if not params.decoder.binf_projection else decoder_inputs,
                source_sequence_length, target_sequence_length,
                mode, params.decoder, not params.decoder.binf_projection,
                binf_embedding if not params.decoder.binf_sampling or params.decoder.beam_width > 0 else None)

    sample_ids_phones_binf, sample_ids_phones, sample_ids_binf, logits_binf, logits = None, None, None, None, None
    with tf.name_scope('prediction'):
        if mode == tf.estimator.ModeKeys.PREDICT and params.decoder.beam_width > 0:
            logits = tf.no_op()
            if decoder_outputs is not None:
                sample_ids_phones = decoder_outputs.predicted_ids
            if decoder_outputs_binf is not None:
                sample_ids_phones_binf = decoder_outputs_binf.predicted_ids
        else:
            if decoder_outputs is not None:
                logits = decoder_outputs.rnn_output
                sample_ids_phones = tf.to_int32(tf.argmax(logits, -1))
            if decoder_outputs_binf is not None:
                logits_binf = decoder_outputs_binf.rnn_output
                if params.decoder.binary_outputs and params.decoder.binf_sampling:
                    logits_phones_binf = transform_binf_to_phones(logits_binf, binf_embedding)
                    sample_ids_phones_binf = tf.to_int32(tf.argmax(logits_phones_binf, -1))
                else:
                    sample_ids_phones_binf = tf.to_int32(tf.argmax(logits_binf, -1))

    if mode == tf.estimator.ModeKeys.PREDICT:
        emb_c = tf.concat([x.c for x in encoder_state], axis=1)
        emb_h = tf.concat([x.h for x in encoder_state], axis=1)
        emb = tf.stack([emb_c, emb_h], axis=1)
        predictions = {
            'embedding': emb,
            'encoder_out': encoder_outputs,
            'source_length': source_sequence_length
        }
        if sample_ids_phones is not None:
            predictions['sample_ids'] = sample_ids_phones
        if logits_binf is not None:
            predictions['logits_binf'] = logits_binf
        if sample_ids_phones_binf is not None:
            predictions['sample_ids_phones_binf'] = sample_ids_phones_binf

        if final_context_state is not None:
            predictions['alignment'] = get_alignment_history(final_context_state, params)
        if final_context_state_binf is not None:
            predictions['alignment_binf'] = get_alignment_history(final_context_state_binf, params)

        if params.decoder.beam_width == 0:
            if params.decoder.binary_outputs and binf_embedding is None:
                predictions['probs'] = tf.nn.sigmoid(logits_binf)
            elif logits is not None:
                predictions['probs'] = tf.nn.softmax(logits)
            else:
                predictions['probs'] = tf.nn.softmax(logits_binf)

        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    metrics = None
    edit_distance, edit_distance_binf = None, None
    with tf.name_scope('metrics'):
        if sample_ids_phones is not None:
            edit_distance = utils.edit_distance(
                sample_ids_phones, targets, utils.EOS_ID, params.mapping if mapping is None else None)
        if sample_ids_phones_binf is not None:
            edit_distance_binf = utils.edit_distance(
                sample_ids_phones_binf, targets, utils.EOS_ID, params.mapping if mapping is None else None)
        metrics = {
            'edit_distance': tf.metrics.mean(edit_distance if edit_distance is not None else edit_distance_binf),
        }

    # In TRAIN model this becomes an significantly affected by early high values.
    # As a result in summaries train values would be high and drop after restart.
    # To prevent this, we use last batch average in case of TRAIN.
    if mode != tf.estimator.ModeKeys.TRAIN:
        tf.summary.scalar('edit_distance', metrics['edit_distance'][1])
    else:
        tf.summary.scalar('edit_distance', tf.reduce_mean(edit_distance if edit_distance is not None else edit_distance_binf))

    audio_loss_ipa, audio_loss_binf = None, None
    if logits is not None:
        with tf.name_scope('cross_entropy'):
            audio_loss_ipa = compute_loss(
                logits, targets, final_sequence_length, target_sequence_length, mode)

    if logits_binf is not None:
        with tf.name_scope('cross_entropy_binf'):
            if params.decoder.binf_projection:
                audio_loss_binf = compute_loss(
                    logits_binf, targets, final_sequence_length_binf, target_sequence_length, mode)
            else:
                if mode == tf.estimator.ModeKeys.TRAIN:
                    audio_loss_binf = compute_loss_sigmoid(logits_binf, targets_binf,
                        final_sequence_length_binf, target_sequence_length, mode)
                else:
                    audio_loss_binf = compute_loss_sigmoid(logits_binf, targets,
                        final_sequence_length_binf, target_sequence_length, mode)

    audio_loss = 0
    if audio_loss_ipa is not None:
        audio_loss += audio_loss_ipa
    if audio_loss_binf is not None:
        audio_loss += audio_loss_binf
        tf.summary.scalar('audio_loss_binf', audio_loss_binf)

    ctc_edit_distance = None
    if params.ctc_weight > 0:
        ctc_logits = tf.layers.dense(encoder_outputs, params.decoder.target_vocab_size + 1,
                                     activation=None, name='ctc_logits')
        decoded_ctc, _ = tf.nn.ctc_greedy_decoder(tf.transpose(ctc_logits, [1, 0, 2]), source_sequence_length)
        decoded_ctc = tf.sparse.to_dense(decoded_ctc[0])
        decoded_ctc = tf.cast(decoded_ctc, tf.int32)
        if target_sequence_length is not None:
            ctc_loss = tf.nn.ctc_loss_v2(labels=targets, logits=ctc_logits, logits_time_major=False,
                                         label_length=target_sequence_length, logit_length=source_sequence_length)
            ctc_loss = tf.reduce_mean(ctc_loss, name='ctc_phone_loss')
            audio_loss += ctc_loss * params.ctc_weight
            tf.summary.scalar('ctc_loss', ctc_loss)
            with tf.name_scope('ctc_metrics'):
                ctc_edit_distance = utils.edit_distance(
                    decoded_ctc, targets, utils.EOS_ID, params.mapping if mapping is None else None)
                metrics['ctc_edit_distance'] = tf.metrics.mean(ctc_edit_distance)
            if mode != tf.estimator.ModeKeys.TRAIN:
                tf.summary.scalar('ctc_edit_distance', metrics['ctc_edit_distance'][1])
            else:
                tf.summary.scalar('ctc_edit_distance', tf.reduce_mean(ctc_edit_distance))

    if mode == tf.estimator.ModeKeys.EVAL:
        with tf.name_scope('alignment'):
            attention_images = utils.create_attention_images(
                final_context_state or final_context_state_binf)

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
        log_data = {
            'edit_distance': tf.reduce_mean(edit_distance if edit_distance is not None else edit_distance_binf),
            'max_edit_distance': tf.reduce_max(edit_distance if edit_distance is not None else edit_distance_binf),
            'min_edit_distance': tf.reduce_min(edit_distance if edit_distance is not None else edit_distance_binf)
        }
        logging_hook = tf.train.LoggingTensorHook(log_data, every_n_iter=20)
        hooks += [logging_hook]

        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics,
                                          evaluation_hooks=hooks)

    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(params.learning_rate)
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        total_params = np.sum([np.prod(x.shape.as_list()) for x in var_list])
        tf.logging.info('Trainable parameters: {}'.format(total_params))

        regularizer = tf_contrib.layers.l2_regularizer(params.l2_reg_scale)
        reg_term = tf.contrib.layers.apply_regularization(regularizer, var_list)
        audio_loss = audio_loss + reg_term

        gvs = optimizer.compute_gradients(audio_loss, var_list=var_list)
        capped_gvs = [(tf.clip_by_norm(grad, GRAD_NORM), var) for grad, var in gvs]
        train_op = optimizer.apply_gradients(capped_gvs, global_step=tf.train.get_global_step())
        if params.add_noise > 0:
            def add_noise():
                noise_ops = [train_op]
                for var in var_list:
                    if var.name.endswith('kernel:0'):
                        shape = tf.shape(var)
                        noise_op = tf.assign_add(var, tf.random_normal(shape, NOISE_MEAN, params.noise_std,
                                                                       dtype=tf.float32))
                        noise_ops.append(noise_op)
                print_op = tf.print('Adding noise to weights')
                return tf.group(*noise_ops, print_op)
            train_op = tf.cond(
                tf.logical_and(tf.equal(tf.mod(tf.train.get_global_step(), params.add_noise), 0),
                               tf.greater(tf.train.get_global_step(), 0)),
                add_noise, lambda: train_op)

    loss = audio_loss
    train_log_data = {'loss': loss,
                      'edit_distance': tf.reduce_mean(edit_distance if edit_distance is not None else edit_distance_binf)}
    if ctc_edit_distance is not None:
        train_log_data['ctc_edit_distance'] = tf.reduce_mean(ctc_edit_distance)
    logging_hook = tf.train.LoggingTensorHook(train_log_data, every_n_iter=10)

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, training_hooks=[logging_hook])

