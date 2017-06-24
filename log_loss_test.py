import tensorflow as tf
import numpy as np
class LogLossTest(tf.test.TestCase):
  def testLogLoss(self):
    log_loss_module = tf.load_op_library('./log_loss.so')
    with self.test_session():
      pred = np.array([0.375, 0.371, 0.3939, 0.375])
      y = np.array([1, 0, 1, 0])


      result = log_loss_module.log_loss(y, pred).eval()
      correct = np.sum(-(y * np.log(pred + 1e-7) + (1 - y) * np.log(1 - pred + 1e-7)))

      self.assertAlmostEqual(result, correct, places=5)

if __name__ == "__main__":
  tf.test.main()