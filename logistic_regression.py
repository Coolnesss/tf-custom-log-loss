'''
A logistic regression learning algorithm example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf
import numpy as np

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops

log_loss_module = tf.load_op_library('./log_loss.so')

@ops.RegisterGradient("LogLoss")
def _log_loss_grad(op, grad):
  """The gradients for `log_loss`.

  Args:
    op: The `log_loss` `Operation` that we are differentiating, which we can use
      to find the inputs and outputs of the original op.
    grad: Gradient with respect to the output of the `zero_out` op.

  Returns:
    Gradients with respect to the input of `zero_out`.
  """
  y = op.inputs[0]
  pred = op.inputs[1]
  return [pred - y, pred - y]

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/")

X = mnist.train.images
y = mnist.train.labels

X_binary = X[y == 1]
X_binary = np.concatenate((X_binary, X[y == 0]), axis=0)
y_binary = y[y == 1]
y_binary = np.concatenate((y_binary, y[y == 0]), axis=0)
print(X_binary.shape)
print(y_binary.shape)

N = X_binary.shape[0]

# Parameters
learning_rate = 0.01
training_epochs = 25
batch_size = 100
display_step = 1

# tf Graph Input
x = tf.placeholder(tf.float32, [None, 784]) 
y = tf.placeholder(tf.float32, [None, 1]) 

# Set model weights
W = tf.Variable(tf.zeros([784, 1]))
b = tf.Variable(tf.zeros([1]))

print(x.shape, W.shape)
# Construct model
pred = tf.nn.sigmoid(tf.matmul(x, W) + b) 

# Minimize error using CUSTOM LOG LOSS
#cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
cost = log_loss_module.log_loss(y, pred)

# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(N / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            #batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            idx = np.random.randint(N, size=batch_size)
            batch_xs = X_binary[idx]
            batch_ys = y_binary[idx]
            batch_ys = np.reshape(batch_ys, (batch_ys.shape[0], 1))

            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                          y: batch_ys})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

    print("Optimization Finished!")

    X_test = mnist.test.images
    y_test = mnist.test.labels

    X_binary = X_test[y_test == 1]
    X_binary = np.concatenate((X_binary, X_test[y_test == 0]), axis=0)
    y_binary = y_test[y_test == 1]
    y_binary = np.concatenate((y_binary, y_test[y_test == 0]), axis=0)
    y_binary = np.reshape(y_binary, (y_binary.shape[0], 1))

    print(X_binary.shape, y_binary.shape)

    # Test model
    correct_prediction = tf.equal(tf.round(pred), y)
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Test accuracy:", accuracy.eval({x: X_binary, y: y_binary}))
