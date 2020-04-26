
/**
 * This layer performs the following simple operation:-
 *  output = input_mean  + exp( input_log_var/2)*epsilon
 *  epsilon is a configurable constant.
 */
class samplingLayer extends tf.layers.Layer {
    constructor(config){
        super(config);
    }


    /*
    can't use build here because using more then 1 input tensor
     */
    computeOutputShape(inputShape) {
        tf.util.assert(inputShape.length === 2 && Array.isArray(inputShape[0]),
        () => `Expected exactly 2 input shapes. ` +
        `But got: ${inputShape}`);

        return [inputShape[0]]
    }

  /**
   * call() contains the actual numerical computation of the layer.
   *
   * It is "tensor-in-tensor-out". I.e., it receives one or more
   * tensors as the input and should produce one or more tensors as
   * the return value.
   */
    call(input){

        return tf.tidy(() =>{
            const mean = input[0];
            const logVariance = input[1];
            const epsilon = tf.randomNormal(mean.shape, 0, 1);
            const k = tf.exp(logVariance.div(2)).mul(epsilon);
            return tf.add( mean, k);
        })
    }


  /**
   * getConfig() generates the JSON object that is used
   * when saving and loading the custom layer object.
   */
    getConfig(){
        const config = super.getConfig();
        Object.assign(config, {epsilon: this.epsilon});
        return config;
    }

  /**
   * The static className getter is required by the 
   * registration step (see below).
   */
    static get className(){
        return 'samplingLayer'
    }

}


/**
 * Regsiter the custom layer, so TensorFlow.js knows what class constructor
 * to call when deserializing an saved instance of the custom layer.
 */
tf.serialization.registerClass(samplingLayer);