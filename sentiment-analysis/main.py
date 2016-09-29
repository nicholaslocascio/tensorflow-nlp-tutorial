import numpy as np
from models import SentimentRNN
import pprint
import os
import utils
pp = pprint.PrettyPrinter()

import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_float("learning_rate", 0.005, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_float("keep_prob", 0.9, "Keep Probability for dropout")
flags.DEFINE_integer("batch_size", 128, "The size of batch [128]")
flags.DEFINE_integer("n_recurrent_layers", 2, "Number of recurrent layers [3]")
flags.DEFINE_integer("n_fc_layers", 2, "Number of fully connected layers [2]")
flags.DEFINE_integer("recurrent_layer_width", 128, "Width of recurrent layers [256]")
flags.DEFINE_integer("fc_layer_width", 128, "Width of fully connected layers [256]")
flags.DEFINE_integer("max_length", 50, "Max length of sentence to consider. [50]")
flags.DEFINE_boolean("is_train", True, "True for training, False for testing [False]")
flags.DEFINE_integer("epoch", 50, "Epoch to train [50]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
FLAGS = flags.FLAGS

def main(_):
    pp.pprint(flags.FLAGS.__flags)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)

    X, y, sentences, index_to_word = utils.load_sentiment_data(FLAGS.max_length)
    vocab_size, n_classes = X.shape[2], y.shape[1]
    X_train, y_train, X_test, y_test = utils.split_data(X, y)

    with tf.Session() as sess:
        deep_pdf = SentimentRNN(sess, vocab_size=vocab_size, n_classes=n_classes,
            batch_size=FLAGS.batch_size, keep_prob=FLAGS.keep_prob, max_length=FLAGS.max_length, 
            n_recurrent_layers=FLAGS.n_recurrent_layers, n_fc_layers= FLAGS.n_fc_layers,
            recurrent_layer_width=FLAGS.recurrent_layer_width, fc_layer_width=FLAGS.fc_layer_width,
            checkpoint_dir=FLAGS.checkpoint_dir, epoch=FLAGS.epoch)

        if FLAGS.is_train:
            deep_pdf.train(FLAGS, X_train, y_train, X_test, y_test)
        else:
            deep_pdf.load(FLAGS.checkpoint_dir)
            # Do inference with model


if __name__ == '__main__':
    tf.app.run()
