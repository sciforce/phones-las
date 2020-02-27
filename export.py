from argparse import ArgumentParser
import tensorflow as tf

import las
import utils


def export_las_model_fn(features, labels, mode, config, params):
    """Simplified model_fn for exporting basic version of LAS."""
    assert mode == tf.estimator.ModeKeys.PREDICT
    encoder_inputs = features['encoder_inputs']
    source_sequence_length = features['source_sequence_length']

    decoder_inputs = None
    target_sequence_length = None

    tf.logging.info('Building listener')

    with tf.variable_scope('listener'):
        (encoder_outputs, source_sequence_length), encoder_state = las.model.listener(
            encoder_inputs, source_sequence_length, mode, params.encoder)

    tf.logging.info('Building speller')

    with tf.variable_scope('speller'):
        decoder_outputs, final_context_state, final_sequence_length = las.model.speller(
            encoder_outputs, encoder_state, decoder_inputs,
            source_sequence_length, target_sequence_length,
            mode, params.decoder)

    with tf.name_scope('prediction'):
        if params.decoder.beam_width > 0:
            logits = tf.no_op()
            sample_ids = decoder_outputs.predicted_ids
        else:
            logits = decoder_outputs.rnn_output
            sample_ids = tf.to_int32(tf.argmax(logits, -1))

    predictions = {
        'sample_ids': sample_ids
    }
    try:
        predictions['alignment'] = tf.transpose(final_context_state.alignment_history.stack(), perm=[1, 0, 2])
    except AttributeError:
        # this works only for single audio inference!
        predictions['alignment'] = tf.expand_dims(tf.transpose(final_context_state[0].alignment_history,
                                                               perm=[1, 0, 2]), axis=0)
    if params.decoder.beam_width == 0:
        if params.decoder.binary_outputs:
            predictions['probs'] = tf.nn.sigmoid(logits)
        else:
            predictions['probs'] = tf.nn.softmax(logits)

    return tf.estimator.EstimatorSpec(mode, predictions=predictions)


def serving_input_factory(num_channels):
    """Return a serving function that accepts features with given number of channels."""
    def serving_input_receiver_fn():
        inputs = {
            'encoder_inputs': tf.placeholder(tf.float32, [None, None, num_channels]),
            'source_sequence_length': tf.placeholder(tf.int32, [None])
        }
        return tf.estimator.export.ServingInputReceiver(inputs, inputs)
    return serving_input_receiver_fn


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True, help='path to model')
    parser.add_argument('--num_channels', type=int, required=True, help='number of input channels')
    parser.add_argument('--export_dir', type=str, required=True, help='path where to save exported model')
    args = parser.parse_args()

    config = tf.estimator.RunConfig(model_dir=args.model_dir)
    hparams = utils.create_hparams(args, sos_id=utils.SOS_ID, eos_id=utils.EOS_ID)

    model = tf.estimator.Estimator(model_fn=export_las_model_fn, config=config, params=hparams)
    model.export_saved_model(args.export_dir, serving_input_receiver_fn=serving_input_factory(args.num_channels))
