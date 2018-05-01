import * as tf from '@tensorflow/tfjs';

export default function() {
  // Getting Start
  const model = tf.sequential();
  model.add(tf.layers.dense({
    units: 1,
    inputShape: [1],
  }));
  model.compile({
    loss: 'meanSquaredError',
    optimizer: 'sgd',
  });
  const xs = tf.tensor2d([1, 2, 3, 4], [4, 1]);
  const ys = tf.tensor2d([1, 3, 5, 7], [4, 1]);
  model.fit(xs, ys).then(() => {
    model.predict(tf.tensor2d([5], [1, 1])).print();
  });

  // Core Concepts in TensorFlow.js
  const shape = [2, 3]; // 2 rows, 3 columns
  const a = tf.tensor([1, 2, 3, 10, 20, 30], shape);
  a.print();

  const b = tf.tensor([[1, 2, 3], [10, 20, 30]]);
  b.print();

  const c = tf.tensor2d([[1, 2, 3], [10, 20, 30]]);
  c.print();

  const zeros = tf.zeros([3, 5]);
  zeros.print();

  const initVals = tf.zeros([5]);
  const biases = tf.variable(initVals);
  biases.print();

  const updateVals = tf.tensor1d([0, 1, 0, 1, 0]);
  biases.assign(updateVals);
  biases.print();

  const d = tf.tensor2d([[1, 2], [3, 4]]);
  const dSquared = d.square();
  dSquared.print();

  const e = tf.tensor2d([[1, 2], [3, 4]]);
  const f = tf.tensor2d([[4, 6], [7, 8]]);
  const ePlusF = e.add(f);
  ePlusF.print();

  const sqSum = e.add(f).square();
  sqSum.print();
};
