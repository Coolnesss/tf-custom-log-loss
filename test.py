import tensorflow as tf
log_loss_module = tf.load_op_library('./log_loss.so')
with tf.Session(''):
    a = [[1, 2], [3, 4]]
    b = tf.zeros((2,2))
    print log_loss_module.log_loss(a, b).eval()