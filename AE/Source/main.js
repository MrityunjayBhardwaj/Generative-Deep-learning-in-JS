import {MnistData} from '../../dependency/data/mnist_data.js';

async function showExamples(n='i d ex', data) {
  // Create a container in the visor
  const surface =
    tfvis.visor().surface({ name: n, tab: 'Input Data'});  

  // Get the examples
  const examples = data;
  const numExamples = 20;
  
  // Create a canvas element to render each example
  for (let i = 0; i < numExamples; i++) {
    const imageTensor = tf.tidy(() => {
      // Reshape the image to 28x28 px
      return examples
        .slice([i, 0], [1, examples.shape[1]])
        .reshape([28, 28, 1]);
    });
    
    const canvas = document.createElement('canvas');
    canvas.width = 28;
    canvas.height = 28;
    canvas.style = 'margin: 4px;';
    await tf.browser.toPixels(imageTensor, canvas);
    surface.drawArea.appendChild(canvas);

    imageTensor.dispose();
  }
}

function doPrediction(model, data, testDataSize = 500) {
  const IMAGE_WIDTH = 28;
  const IMAGE_HEIGHT = 28;
  const testData = data.nextTestBatch(testDataSize);
  const testxs = testData.xs.reshape([testDataSize, IMAGE_WIDTH, IMAGE_HEIGHT, 1]);
  const preds = model.predict(testxs);

  return [preds, testxs];
}

async function showAccuracy(model, data) {
  const [preds, labels] = doPrediction(model, data);
  const classAccuracy = await tfvis.metrics.perClassAccuracy(labels, preds);
  const container = {name: 'Accuracy', tab: 'Evaluation'};
  tfvis.show.perClassAccuracy(container, classAccuracy, classNames);

  labels.dispose();
}

async function showConfusion(model, data) {
  const [preds, labels] = doPrediction(model, data);
  const confusionMatrix = await tfvis.metrics.confusionMatrix(labels, preds);
  const container = {name: 'Confusion Matrix', tab: 'Evaluation'};
  tfvis.render.confusionMatrix(
      container, {values: confusionMatrix}, classNames);

  labels.dispose();
}



let predOut, dataOut;

let model = 0;
async function runAE() {  
  const data = new MnistData();
  await data.load();
  // await showExamples('i d n',data.nextTestBatch(20).xs);


  const LEARNING_RATE = .0005;
  const BATCH_SIZE = 100;
  const INITIAL_EPOCH = 0;
  const nEpoch = 1000;

  const AE = new AutoEncoder([28, 28, 1],
    { encoderConvFilters : [32,64,64, 64],
     encoderConvKernelSize : [3,3,3,3],
     encoderConvStrides : [1,2,2,1],
     decoderConvTFilters : [64,64,32,1],
     decoderConvTKernelSize : [3,3,3,3],
     decoderConvTStrides : [1,2,2,1],
     z_dim : 2,
    }
  );

   model = AE.getMyModel();

  tfvis.show.modelSummary({name: 'Encoder Architecture'}, AE.getEncoderModel());
  tfvis.show.modelSummary({name: 'Decoder Architecture'}, AE.getDecoderModel());


  

  // using pre-trained network..
  let savedEncoderModel = await tf.loadLayersModel('./SavedModel/model-2/encoder2.json');
  let savedDecoderModel = await tf.loadLayersModel('./SavedModel/model-2/decoder2.json');
  let savedMyModel = await tf.loadLayersModel('./SavedModel/model-2/MyModel-2.json');
  
  AE.loadWeights(savedEncoderModel.getWeights(), savedDecoderModel.getWeights(), savedMyModel.getWeights());

  console.log("fine");
  // let fromPy = await tf.loadLayersModel('./SavedModel/trainedFromPy/model.json');
 
  
  tfvis.show.modelSummary({name: 'Model Architecture'}, model);

  const TRAIN_DATA_SIZE = 100;
  const trainXs = data.nextTrainBatch(TRAIN_DATA_SIZE).xs.reshape([TRAIN_DATA_SIZE, 28, 28, 1]);


  // await AE.compile(LEARNING_RATE, 1000);
  // await AE.train(trainXs, {batchSize: BATCH_SIZE, epoch: nEpoch});

  
  // [predOut, dataOut] = doPrediction(AE.getMyModel(), data);

  // await showExamples('generated Samples ',predOut.reshape([predOut.shape[0],784]));
  // showExamples('Input Samples',dataOut.reshape([dataOut.shape[0],784]));


  await showLatentSpace(AE, trainXs)
  await reconstructingOriginalImage(AE, trainXs);

  window.AE = AE;


loop()
}



document.addEventListener('DOMContentLoaded', runAE);

async function showLatentSpace(AE, data){

  const surface0 =
    tfvis.visor().surface({ name: 'LatentSpace', tab: 'Input Data'});  

  const numExamples = data.shape[0];
  let examples = data;

  examples = examples.reshape([examples.shape[0], 28, 28, 1])

  const zPoints = AE.getEncoderModel().predict(examples)[2];

const myData = zPoints.arraySync().map((a) =>{return { x: a[0], y: a[1]}});

const data2 = {values: myData};
// console.log(series2)
// console.log(myData)


tfvis.render.scatterplot(surface0, data2);


}


async function reconstructingOriginalImage(AE, data){

  const surface0 =
    tfvis.visor().surface({ name: 'original Images', tab: 'Input Data'});  

  // Get the examples
  const numExamples = data.shape[0];
  let examples = data;

  examples = examples.reshape([examples.shape[0], 28, 28, 1])

  const zPoints = AE.getEncoderModel().predict(examples);
  const reconstructedImage = AE.getDecoderModel().predict(zPoints);

  zPoints.print();

  // const reconstructedImage = AE.getMyModel().predict(examples);

  
  // Create a canvas element to render each example
  for (let i = 0; i < numExamples; i++) {
    const imageTensor = tf.tidy(() => {
      // Reshape the image to 28x28 px
      return examples
        .slice([i, 0], [1, examples.shape[1]])
        .reshape([28, 28, 1]);
    });
    
    const canvas = document.createElement('canvas');
    canvas.width = 28;
    canvas.height = 28;
    canvas.style = 'margin: 4px;';
    await tf.browser.toPixels(imageTensor, canvas);
    surface0.drawArea.appendChild(canvas);

    imageTensor.dispose();
  }



  const surface1 =
    tfvis.visor().surface({ name: 'reconstructed Images', tab: 'Input Data'});  




  for (let i = 0; i < numExamples; i++) {
    let imageTensor = tf.tidy(() => {
      // Reshape the image to 28x28 px
      return reconstructedImage
        .slice([i, 0], [1, examples.shape[1]])
        .reshape([28, 28, 1]);
    });

    imageTensor = imageTensor.div(imageTensor.max().sub(imageTensor.min()));

    
    const canvas = document.createElement('canvas');
    canvas.width = 28;
    canvas.height = 28;
    canvas.style = 'margin: 4px;';
    await tf.browser.toPixels(imageTensor, canvas);
    surface1.drawArea.appendChild(canvas);

    imageTensor.dispose();
  }
}







