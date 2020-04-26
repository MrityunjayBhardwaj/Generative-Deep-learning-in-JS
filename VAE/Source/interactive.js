let reconImgElem = document.getElementById('reconstructedImg');

const canvas = document.createElement('canvas');
canvas.width = 580;
canvas.height = 580;
// canvas.style = 'margin: 4px;';
// canvas.style.marginTop = '30px';
canvas.style.backgroundColor= 'black';

let img = [];

let tensorImgs = 0;
function convert2Tensor(){

  console.log('inside run')
  for(let i=0;i<img.length;i++){
    img[i].loadPixels();

    let cPixelVals = img[i].imageData.data;
    cPixelVals = Array.prototype.slice.call(cPixelVals); 

    // Converting to tf.tensor
    let currTensorImg = tf.tensor(cPixelVals).reshape([img[i].width, img[i].height, 4]);

    console.log(cPixelVals.length, currTensorImg.shape);
    // extract only the RGB part
    currTensorImg = currTensorImg.slice([0,0,0],[-1,-1,3]);
    if(!tensorImgs){
      tensorImgs = currTensorImg.expandDims();
    }
    else{
      tensorImgs = tensorImgs.concat(currTensorImg.expandDims())
    }

  }
}

function preload(){
  img.push(loadImage('./Assets/Imgs/100px/3.jpg'));
  img.push(loadImage('./Assets/Imgs/100px/4.jpg'));
  img.push(loadImage('./Assets/Imgs/100px/5.jpg'));
  img.push(loadImage('./Assets/Imgs/100px/6.jpg'));
  img.push(loadImage('./Assets/Imgs/100px/7.jpg'));
  img.push(loadImage('./Assets/Imgs/100px/8.jpg'));
  




}


function setup() {

convert2Tensor();
  console.log('its awesome!', tensorImgs.shape, img[0].width);

    var canvas = createCanvas(500, 500);
 
  // Move the canvas so it’s inside our <div id="sketch-holder">.
  canvas.parent('latentSpaceViz');

  frameRate(30)

  noLoop();


  
}

let prevZPoint = [0,0]

function draw() {
  // if (mouseIsPressed) {
  //   fill(0);
  // } else {
  //   fill(255);
  // }
  background(200);

  fill(255);
  ellipse(mouseX, mouseY, 50, 50);

  // make our mouse position as the coordinates on our latent space

  const boxWidth = 100 
  let zPoints = [-boxWidth + (mouseX/width)*(boxWidth*2), -boxWidth + (mouseY/height)*(boxWidth*2)];

  if (!(prevZPoint[0] === zPoints[0] && prevZPoint[1] === zPoints[1]) && mouseX < width && mouseY < height){
    try{
    prevZPoint = zPoints
    zPoints = tf.tensor(zPoints).expandDims();

    // reconstructing image given the latent vector


      const reconstructedImage = window.VAE.getDecoderModel().predict(zPoints);

      let imageTensor =
        // Reshape the image to 28x28 px
        reconstructedImage
          .reshape(reconstructedImage.shape.slice(1));


      imageTensor = imageTensor.sub(imageTensor.min()).div( imageTensor.max().sub(imageTensor.min()) );

      
      tf.browser.toPixels(tf.image.resizeBilinear(imageTensor, [580,580]) , canvas);
      // surface1.drawArea.appendChild(canvas);
      reconImgElem.appendChild(canvas);
    }
    catch(E){

      console.log(E)
    }

  }
    // console.log("slkdjf")

    // imageTensor.dispose();

    //   var c = document.getElementById("myCanvas");
//   var ctx = canvas.getContext("2d");
//   var img = document.getElementById("scream");
//   ctx.drawImage(imageTensor.arraySync(), 28, 28);

}