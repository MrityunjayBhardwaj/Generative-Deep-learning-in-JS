function VariationalAutoencoder(inputDim, 
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

            myModel: null,

            distribution : {params: []}


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

      // The Variational Part:-

      let mean= tf.layers.dense({units: model.zDim, name:"mean"}).apply(encoderModel);
      let logVariance= tf.layers.dense({units: model.zDim, name:"LogVariance"}).apply(encoderModel);

      // TODO: make it general so that we can model any distribution

      // function sampling(params={mean :tf.tensor([]),logVariance: tf.tensor([])}){


      //   const epsilon =  tf.randomNormal(shape=[1,params.mean.shape[1]]).expandDims(1);
      //   return params.mean.add(tf.exp(params.logVariance.div(2)).mul(epsilon));
      // }

      const sampling = new samplingLayer({ outputShape: [model.zDim]});

      

      // let encoderOutput = sampling.apply([model.distribution.params.mean, model.distribution.params.logVariance]);
      let encoderOutput = sampling.apply([mean, logVariance]);

      model.encoder = tf.model({inputs: encoderInput, 
                                outputs: [mean, logVariance, encoderOutput[0]]});

     
      

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
      const encoderOut = model.encoder.apply(myModelInput);
      const myModelOutput = model.decoder.apply(encoderOut[2]);

      model.myModel = tf.model({inputs: myModelInput, outputs: [myModelOutput, ...encoderOut]});


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

    // last two args are optional
function processLargeArrayAsync(array, fn, maxTimePerChunk, context) {
    context = context || window;
    maxTimePerChunk = maxTimePerChunk || 200;
    var index = 0;

    function now() {
        return new Date().getTime();
    }

    function doChunk() {
        var startTime = now();
        while (index < array.length && (now() - startTime) <= maxTimePerChunk) {
            // callback called with args (value, index, array)
            fn.call(context, array[index], index, array);
            ++index;
        }
        if (index < array.length) {
            // set Timeout for async iteration
            setTimeout(doChunk, 1);
        }
    }    
    doChunk();    
}



    this.train = async function(data, params={batchSize: 5, epoch: 10}){

        // window.alert('epoch: '+params.epoch);

      // Get K-L Divergence loss:-
      function rLoss(yTrue, yPred){
        // return tf.mean(yTrue.sub(yPred).pow(2), axis=[1,2,3]);
        return tf.losses.meanSquaredError(yTrue, yPred).mul(data.shape[1])
      }

      function KLLoss(mean, logVariance){
        return tf.sum(logVariance.add(1)
               .sub( mean.pow(2))
               .sub(tf.exp(logVariance)), axis=1 ).mul(-1/2);
      }

      function VAELoss(yTrue, yPred, mean, logVariance){
       const r_loss = rLoss(yTrue, yPred);
       const kl_loss = KLLoss(mean, logVariance);

       return r_loss.add(kl_loss).mean();
      }

      function vaeLoss(yTrue, yPred, mean, logVariance){
        return tf.tidy(() => {
            const originalDim = data.shape[1];
            const zMean = mean;
            const zLogVar = logVariance;

            // First we compute a 'reconstruction loss' terms. The goal of minimizing
            // tihs term is to make the model outputs match the input data.
            const reconstructionLoss =
                tf.losses.meanSquaredError(yTrue, yPred).mul(originalDim);

            // binaryCrossEntropy can be used as an alternative loss function
            // const reconstructionLoss =
            //  tf.metrics.binaryCrossentropy(inputs, decoderOutput).mul(originalDim);

            // Next we compute the KL-divergence between zLogVar and zMean, minimizing
            // this term aims to make the distribution of latent variable more normally
            // distributed around the center of the latent space.
            let klLoss = zLogVar.add(1).sub(zMean.square()).sub(zLogVar.exp());
            klLoss = klLoss.sum(-1).mul(-0.5).mul(.5);

            return [reconstructionLoss.add(klLoss).mean(), reconstructionLoss, klLoss];
          })
      }



      const metrics = ['loss', 'val_loss', 'acc', 'val_acc'];
      const container = {
        name: 'Model Training', styles: { height: '1000px' }
      };
      const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);

      console.log(fitCallbacks)



      const dataBatches = tfCreateBatch(data, params.batchSize);

      let cLoss = 0;




          let optimizer = tf.train.adam(learningRate= 0.00008);
      // for (let epoch = 0; epoch < params.epoch; epoch++) {


        // processLargeArrayAsync(veryLargeArray,

        let epoch = 0;
        
        const loopIntervel = await setInterval( () =>{

          // console.log("inside setIntervel");
        // function 
        dataBatches.forEach((cData) => {

            // tf.engine().startScope()
          optimizer.minimize(() => {

            return tf.tidy(() =>{

            const [decoderOutput, mean, logVariance, encoderOutput] = model.myModel.apply(cData);

            const loss = vaeLoss(cData, decoderOutput, mean, logVariance);

              cLoss = loss[0]

              if (epoch % 10 === 0) {
                console.log('epoch:'+epoch+' batch:'+epoch+'\nLoss:', cLoss.dataSync());
                console.log('recon loss:'+loss[1].dataSync());
                console.log('KL loss:'+loss[2].mean().dataSync());
              }

            return cLoss;

            });
          });

          // cData.dispose();
            // tf.engine().endScope()
        }

        );

        

        if(epoch > params.epoch)
          clearInterval(loopIntervel)

        epoch++;

        // if (epoch % 10 === 0){

        //   const series1 = Array(100).fill(0)
        //   .map(y => Math.random() * 100 - (Math.random() * 50))
        //   .map((y, x) => ({ x, y, }));

        // const series2 = Array(100).fill(0)
        //   .map(y => Math.random() * 100 - (Math.random() * 150))
        //   .map((y, x) => ({ x, y, }));
        // const series = ['First', 'Second'];
        // const data2 = { values: [series1, series2], series }

        // const surface = { name: 'Scatterplot', tab: 'Charts' };
        //   // optimizer.dispose();
        // //     console.log(tf.memory().numTensors)
        // tfvis.render.scatterplot(surface, data2);
        // await fitCallbacks.on

        // }

        }, 1)

        // );
      // }

      // console.log(loopIntervel)



      // for(let i=0;i< params.epoch;i++){

      //   // if (i > 20)
      //   //   cLearningRate = 0.0005
      //    optimizer = tf.train.adam(learningRate= cLearningRate);


      //   for(let j=0;j< noOfBatches;j++){

      //     /* generating batch Images:- */
      //     let a = new Array(data.shape.length).fill(0);
      //     let b = new Array(data.shape.length).fill(-1);

      //     a[0] = j;
      //     b[0] = params.batchSize;

      //     const batchedImages = data.slice(a,b);

      //     await fitCallbacks.onEpochEnd();

      //     let cLoss = 0;

      //     optimizer.minimize(() => {
      //       const [decoderOutput, mean, logVariance, encoderOutput] = model.myModel.apply(batchedImages);

      //       const loss = vaeLoss(batchedImages, decoderOutput, mean, logVariance);

      //       if (i % 10 === 0) {
      //         console.log('epoch:'+i+' batch:'+j+'\nLoss:', loss[0].dataSync());
      //         console.log('recon loss:'+loss[1].dataSync());
      //         console.log('KL loss:'+loss[2].mean().dataSync());
      //       }

      //         cLoss = loss[0]

      //         // console.log('points: ', encoderOutput.print() )
      //         // console.log('mean: ', mean.print() )
      //         // console.log('log_covariance: ', logVariance.print() )
      //         //  fitCallbacks;
      //       // }
      //       return loss[0];
      //     })

      //     tf.dispose([batchedImages, a, b]);
      //   }


      // return this;

      }
}