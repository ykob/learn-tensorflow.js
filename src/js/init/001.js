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
};
