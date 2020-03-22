const functionOutput = document.querySelector(".function-output");
const AudioContext = window.AudioContext || window.webkitAudioContext;
const modelSampleRate = 30000;

async function loadModel() {
  return await tf.loadLayersModel("/models/model.json");
}

function interpolate(y, inFs, outFs) {
  if (y.length < 2) {
    throw "y must have length >= 2";
  }
  yNew = new Array(Math.ceil(((y.length - 1) * inFs) / outFs) + 1);
  x = 0;
  x0 = 0;
  x1 = inFs;
  oldIndex = 0;
  for (i = 0; i < yNew.length; i++) {
    while (x1 < i * outFs) {
      x0 = x1;
      x1 += inFs;
      oldIndex++;
    }
    alpha = (x - x0) / inFs;
    yNew[i] = (1 - alpha) * y[oldIndex] + alpha * y[oldIndex + 1];
    x += outFs;
  }
  yNew[yNew.length - 1] = y[y.length - 1];
  return yNew;
}

class Spectrogram extends tf.layers.Layer {
  constructor(config) {
    super(config || {});
    this.frameLength = config["frameLength"];
    this.frameStep = config["frameStep"];
  }

  computeOutputShape(_inputShape) {
    return [null, null, this.frameLength / 2 + 1];
  }

  call(xList) {
    if (Array.isArray(xList)) {
      // TFJS wraps inputs into lists...
      var x = xList[0];
    } else {
      var x = xList;
    }
    if (x.shape[0] !== 1) {
      throw "TFJS's stft function only supports 1 dimension.";
    }
    return tf
      .sqrt(
        tf.abs(tf.signal.stft(x.flatten(), this.frameLength, this.frameStep))
      )
      .expandDims();
  }

  static get className() {
    return "Spectrogram";
  }
}
tf.serialization.registerClass(Spectrogram);

if (navigator.mediaDevices) {
  console.log("getUserMedia supported.");
  loadModel()
    .then(function(model) {
      navigator.mediaDevices
        .getUserMedia({ audio: true })
        .then(function(stream) {
          const audioCtx = new AudioContext();
          const source = audioCtx.createMediaStreamSource(stream);

          var audioBuffer = new Array(audioCtx.sampleRate * 20).fill(0);

          scriptNode = audioCtx.createScriptProcessor(16384, 1, 1);
          scriptNode.onaudioprocess = function(event) {
            resampled = interpolate(
              event.inputBuffer.getChannelData(0),
              1 / audioCtx.sampleRate,
              1 / modelSampleRate
            );
            audioBuffer = audioBuffer.slice(resampled.length).concat(resampled);

            // 1. Resample to desired sample rate (30,000 Hz)
            // 2. Call model on audioBuffer
            functionOutput.innerHTML = model.predict(
              tf.tensor(audioBuffer, [1, audioBuffer.length])
            );
            console.log(audioBuffer[0]);
          };

          source.connect(scriptNode);
          scriptNode.connect(audioCtx.destination);
        })
        .catch(function(err) {
          console.log("Error with getUserMedia: ", err);
        });
    })
    .catch(function(err) {
      throw err;
    });
} else {
  console.log("getUserMedia not supported on your browser");
}
