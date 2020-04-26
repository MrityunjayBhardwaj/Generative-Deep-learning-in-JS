let reconImgElem = document.getElementById('reconstructedImg');

const canvas = document.createElement('canvas');
canvas.width = 580;
canvas.height = 580;
// canvas.style = 'margin: 4px;';
// canvas.style.marginTop = '30px';
canvas.style.backgroundColor= 'black';

function setup() {


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

  const boxWidth = 6
  let zPoints = [-boxWidth + (mouseX/width)*(boxWidth*2), -boxWidth + (mouseY/height)*(boxWidth*2)];

  if (!(prevZPoint[0] === zPoints[0] && prevZPoint[1] === zPoints[1]) && mouseX < width && mouseY < height){
    prevZPoint = zPoints
    zPoints = tf.tensor(zPoints).expandDims();

    // reconstructing image given the latent vector
    const reconstructedImage = window.AE.getDecoderModel().predict(zPoints);

      let imageTensor =
        // Reshape the image to 28x28 px
        reconstructedImage
          .reshape([28, 28, 1]);

      imageTensor = imageTensor.div(imageTensor.max().sub(imageTensor.min()));


      
      tf.browser.toPixels(tf.image.resizeBilinear(imageTensor, [580,580]) , canvas);
      // surface1.drawArea.appendChild(canvas);
      reconImgElem.appendChild(canvas);

  }
    // console.log("slkdjf")

    // imageTensor.dispose();

    //   var c = document.getElementById("myCanvas");
//   var ctx = canvas.getContext("2d");
//   var img = document.getElementById("scream");
//   ctx.drawImage(imageTensor.arraySync(), 28, 28);

}