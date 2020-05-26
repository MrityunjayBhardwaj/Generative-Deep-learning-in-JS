function AutoEncoder(inputDim, 
                     params={
                              encoderConvFilters: [], 
                              encoderConvKernelSize: [], 
                              encoderConvStrides: [], 
                              decoderConvTFilters: [], 
                              decoderConvTKernelSize: [], 
                              decoderConvTStrides: [1,2,1], 
                              zDim: 2, 
                              useBatchNorm: false, 
                              useDropout: false
                            }){
    let model={
            encoderConvFilters: params.encoderConvFilters || [], 
            encoderConvKernelSize: params.encoderConvKernelSize || [], 
            encoderConvStrides: params.encoderConvStrides || [], 
            decoderConvTFilters: params.decoderConvTFilters || [], 
            decoderConvTKernelSize: params.decoderConvTKernelSize || [], 
            decoderConvTStrides: params.decoderConvTStrides || [], 
            zDim: params.zDim || 2, 

            useBatchNorm: false, 
            useDropout: false,



            myModel: null
          }

        
          this.getEncoderModel = () => { return model.encoder };
          this.getDecoderModel = () => { return model.decoder };




    this.getMyModel = function(){
      return model.myModel;
    }
    this._build = function(){

      /* Encoder Network */
      const encoderInput = tf.input({shape: inputDim, name: 'encoder_input'});

      let encoderModel = encoderInput;

      // number of layers in encoder / decoder network
      const nLayersEncoder = model.encoderConvFilters.length;
      const nLayersDecoder = model.encoderConvFilters.length;

      let convLayer = 0;
      for (let i=0;i< nLayersEncoder; i++){
        convLayer = tf.layers.conv2d(
          {
            filters: model.encoderConvFilters[i],
            kernelSize: model.encoderConvKernelSize[i],
            strides: model.encoderConvStrides[i],
            padding: 'same',
            name: 'encoder_conv_'+ i
          }
        )

        encoderModel = convLayer.apply(encoderModel);

        encoderModel = tf.layers.leakyReLU().apply(encoderModel);

        if (model.useBatchNorm)
            encoderModel = tf.layers.batchNormalization().apply(encoderModel);

        if(model.useDropout)
            encoderModel = tf.layers.dropout({rate: 0.25}).apply(encoderModel);
       
      }

      
      const shapeBeforeFlattening = encoderModel.shape.slice(1);

      encoderModel = tf.layers.flatten().apply(encoderModel);

      const encoderOutput = tf.layers.dense({units: model.zDim, name:"encoder_output"}).apply(encoderModel);
      model.encoder = tf.model({inputs: encoderInput, outputs: encoderOutput});
    
      /* Decoder Model */
      const decoderInput = tf.layers.input({shape: [model.zDim], name:"decoder_input"});

      let decoderModel = tf.layers.dense({units: tf.prod(shapeBeforeFlattening).flatten().arraySync()[0]}).apply(decoderInput);
      decoderModel = tf.layers.reshape({targetShape: shapeBeforeFlattening}).apply(decoderModel);
      
      let convTLayer = 0;
      for (let i=0;i< nLayersDecoder; i++){
        convTLayer = tf.layers.conv2dTranspose(
          {
            filters: model.decoderConvTFilters[i],
            kernelSize: model.decoderConvTKernelSize[i],
            strides: model.decoderConvTStrides[i],
            padding: 'same',
            name: 'decoder_conv_t_'+ i
          }
        )


        decoderModel = convTLayer.apply(decoderModel);

        if (i < (nLayersDecoder -1) ){
          decoderModel = tf.layers.leakyReLU().apply(decoderModel);

          if (model.useBatchNorm)
            decoderModel = tf.layers.batchNormalization().apply(decoderModel);

          if (model.useDropout)
            decoderModel = tf.layers.dropout({rate: 0.25}).apply(decoderModel);

        }
        else{
          decoderModel = tf.layers.activation({activation: 'relu'}).apply(decoderModel);
        }

      }

      const decoderOutput = decoderModel;

      model.decoder = tf.model({inputs: decoderInput, outputs: decoderOutput});

      /* Joining the Forces... */

      const myModelInput = encoderInput;
      const myModelOutput = model.decoder.apply(encoderOutput);

      model.myModel = tf.model({inputs: myModelInput, outputs: myModelOutput});


      console.log(model)
       
      return this;
    }


    this._build();

    // this.setMyModel = function(mySavedModel){
    // }
    this.loadWeights = function(encoderWeights, decoderWeights, myModelWeights){
      model.encoder.setWeights(encoderWeights);
      model.decoder.setWeights(decoderWeights);
      model.myModel.setWeights(myModelWeights);

      return this;
    }

    this.compile = function(learningRate=.00005, rLossFactor=1000){

      // const optimizer = tf.train.adam({learningRate: learningRate, beta1: rLossFactor});
      // const optimizer = tf.train.adam({learningRate: learningRate});
      const optimizer = tf.train.adam();

      function rLoss(yTrue, yPred){
        return tf.mean(yTrue.sub(yPred).pow(2), axis=[1,2,3]);
      }
      model.myModel.compile(
          {
              optimizer: optimizer,
              // loss: tf.losses.meanSquaredError,
              loss: rLoss,
              metrics: ['mse'],
          }
      )

      return this;
    }

    this.train = function(data, params={batchSize: 5, epoch: 10}){

        // window.alert('epoch: '+params.epoch);


      const metrics = ['loss', 'val_loss', 'acc', 'val_acc'];
      const container = {
        name: 'Model Training', styles: { height: '1000px' }
      };
      const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);
      
      const BATCH_SIZE = params.batchSize || 5;

      console.log(model)
      console.log(model.myModel)
      return model.myModel.fit(data, data, {
        batchSize: BATCH_SIZE,
        epochs: params.epoch || 2,
        shuffle: true,
        callbacks: fitCallbacks
      });


      return this;
    }
}