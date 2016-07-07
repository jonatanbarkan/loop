import tensorflow as tf
import numpy as np
import math
#import pandas as pd
#import sys


class Autoencoder(object):
    def __init__(self, input_size, sample_size, n_hidden, noise=False, scale=False):
        # # Autoencoder with 1 hidden layer
        # n_samp, n_input = input_data.shape
        # n_hidden = 2
        self.n_samp = sample_size
        self.n_input = input_size
        self.n_hidden = n_hidden
        self.noise = noise
        self.scale = scale

    def add_noise(self):
        # noisy_input = input + .2 * np.random.random_sample((input.shape)) - .1
        # output = input
        noisy_input = input + .2 * np.random.random_sample(((self.n_samp, self.n_input))) - .1
        return noisy_input

    def scale(self):
        # # Scale to [0,1]
        # scaled_input_1 = np.divide((noisy_input - noisy_input.min()), (noisy_input.max() - noisy_input.min()))
        # scaled_output_1 = np.divide((output - output.min()), (output.max() - output.min()))
        # # Scale to [-1,1]
        # scaled_input_2 = (scaled_input_1 * 2) - 1
        # scaled_output_2 = (scaled_output_1 * 2) - 1
        #
        # input_data = scaled_input_2
        # output_data = scaled_output_2
        pass



# input = np.array([[2.0, 1.0, 1.0, 2.0],
#                  [-2.0, 1.0, -1.0, 2.0],
#                  [0.0, 1.0, 0.0, 2.0],
#                  [0.0, -1.0, 0.0, -2.0],
#                  [0.0, -1.0, 0.0, -2.0]])

# Code here for importing data from file
    def run(self, input_data, n_rounds=5000, min_batch_size=50, gradiant_step=0.05):
        output_data = input_data
        if self.noise:
            input_data = self.add_noise()

        x = tf.placeholder("float", [None, self.n_input])
        # Weights and biases to hidden layer
        Wh = tf.Variable(tf.random_uniform((self.n_input, self.n_hidden),
                                           -1.0 / math.sqrt(self.n_input), 1.0 / math.sqrt(self.n_input)))
        bh = tf.Variable(tf.zeros([self.n_hidden]))
        h = tf.nn.tanh(tf.matmul(x, Wh) + bh)
        # Weights and biases to hidden layer
        Wo = tf.transpose(Wh)  # tied weights
        bo = tf.Variable(tf.zeros([self.n_input]))
        y = tf.nn.tanh(tf.matmul(h, Wo) + bo)
        # Objective functions
        y_ = tf.placeholder("float", [None, self.n_input])
        cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
        meansq = tf.reduce_mean(tf.square(y_ - y))
        train_step = tf.train.GradientDescentOptimizer(gradiant_step).minimize(meansq)

        init = tf.initialize_all_variables()
        sess = tf.Session()
        sess.run(init)

        batch_size = min(min_batch_size, self.n_samp)

        for i in range(n_rounds):
            sample = np.random.randint(self.n_samp, size=batch_size)
            batch_xs = input_data[sample][:]
            batch_ys = output_data[sample][:]
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
            if i % 100 == 0:
                print(i, sess.run(cross_entropy, feed_dict={x: batch_xs, y_: batch_ys}), sess.run(meansq, feed_dict={
                    x: batch_xs, y_: batch_ys}))

    def print_final(self,input_data, sess, output_data, x, y, Wh, bh, bo, h):
        print("Target:")
        print(output_data)
        print("Final activations:")
        print(sess.run(y, feed_dict={x: input_data}))
        print("Final weights (input => hidden layer)")
        print(sess.run(Wh))
        print("Final biases (input => hidden layer)")
        print(sess.run(bh))
        print("Final biases (hidden layer => output)")
        print(sess.run(bo))
        print("Final activations of hidden layer")
        print(sess.run(h, feed_dict={x: input_data}))
