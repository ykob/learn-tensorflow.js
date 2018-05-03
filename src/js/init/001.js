import * as tf from '@tensorflow/tfjs';

export default function() {
  const a = tf.variable(tf.scalar(Math.random()));
  const b = tf.variable(tf.scalar(Math.random()));
  const c = tf.variable(tf.scalar(Math.random()));
  const d = tf.variable(tf.scalar(Math.random()));

  const predict = (x) => {
    return tf.tidy(() => {
      return a.mul(x.pow(tf.scalar(3)))
        .add(b.mul(x.square()))
        .add(c.mul(x))
        .add(d)
    });
  };

  const loss = (predictions, labels) => {
    return predictions.sub(labels).square().mean();
  };

  const train = (xs, ys, numIterations = 75) => {
    const learningRate = 0.5;
    const optimizer = tf.train.sgd(lerningRate);

    for (var i = 0; i < numIterations; i++) {
      optimizer.minimize(() => {
        const predsYs = predict(xs);
        return loss(predsYs, ys);
      });
    }
  }
};
