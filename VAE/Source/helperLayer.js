/**
 * Define a custom layer.
 *
 * This layer performs the following simple operation:
 *   output = input * (x ^ alpha);
 * - x is a trainable scalar weight.
 * - alpha is a configurable constant.
 *
 * This custom layer is written in a way that can be saved and loaded.
 */
class TimesXToThePowerOfAlphaLayer extends tf.layers.Layer {
  constructor(config) {
    super(config);
    this.alpha = config.alpha;
  }
  
  /**
   * build() is called when the custom layer object is connected to an
   * upstream layer for the first time.
   * This is where the weights (if any) are created.
   */
  build(inputShape) {
    const shape = [];  // Because our weight (x) is scalar.
    this.x = this.addWeight('x', shape, 'float32', tf.initializers.ones());
  }

  /**
   * call() contains the actual numerical computation of the layer.
   *
   * It is "tensor-in-tensor-out". I.e., it receives one or more
   * tensors as the input and should produce one or more tensors as
   * the return value.
   *
   * Be sure to use tidy() to avoid WebGL memory leak. 
   */
  call(input) {
    return tf.tidy(() => {
      const k = tf.pow(this.x.read(), this.alpha);
      return tf.add(input[0], k);
    });
  }

  /**
   * getConfig() generates the JSON object that is used
   * when saving and loading the custom layer object.
   */
  getConfig() {
    const config = super.getConfig();
    Object.assign(config, {alpha: this.alpha});
    return config;
  }
  
  /**
   * The static className getter is required by the 
   * registration step (see below).
   */
  static get className() {
    return 'TimesXToThePowerOfAlphaLayer';
  }
}
/**
 * Regsiter the custom layer, so TensorFlow.js knows what class constructor
 * to call when deserializing an saved instance of the custom layer.
 */
tf.serialization.registerClass(TimesXToThePowerOfAlphaLayer);

(async function main() {
  const input = tf.input({shape: [4]});
  let x;
  x = tf.layers.dense({units: 1, inputShape: [4]}).apply(input);
  // Here comes an instance of the custom layer.
  x = new TimesXToThePowerOfAlphaLayer({alpha: 1.5}).apply(x);
  const model = tf.model({inputs: input, outputs: x});
  model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});
  model.summary();

  // Train the model using some random data.
  const xs = tf.randomNormal([2, 4]);
  const ys = tf.randomNormal([2, 1]);

  await model.fit(xs, ys, {
    epochs: 5,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        console.log(`Epoch ${epoch}: loss = ${logs.loss}`);
      }
    }
  });
  
  // Save the model and load it back.
  await model.save('indexeddb://codepen-tfjs-model-example-jdBgwB-v1');
  console.log('Model saved.');
  
  const model2 = await tf.loadLayersModel('indexeddb://codepen-tfjs-model-example-jdBgwB-v1');
  console.log('Model2 loaded.')
  
  console.log('The two predict() outputs should be identical:');
  model.predict(xs).print();
  model2.predict(xs).print();
})();
