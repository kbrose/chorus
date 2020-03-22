const functionOutput = document.getElementById("function-output");
const canvasCtx = document.getElementById("spectrogram").getContext("2d");
const AudioContext = window.AudioContext || window.webkitAudioContext;
const modelSampleRate = 30000;
const myImageData = canvasCtx.createImageData(669, 257);
for (var i = 0; i < myImageData.data.length; i += 4) {
  myImageData.data[i + 0] = 0;
  myImageData.data[i + 1] = 0;
  myImageData.data[i + 2] = 0;
  myImageData.data[i + 3] = 255;
}

async function loadModel() {
  return await tf.loadLayersModel("/models/model.json");
}

if (navigator.mediaDevices) {
  navigator.mediaDevices
    .getUserMedia({ audio: true })
    .then(function(stream) {
      const audioCtx = new AudioContext();
      const source = audioCtx.createMediaStreamSource(stream);

      class Spectrogram extends tf.layers.Layer {
        constructor(config) {
          super(config || {});
          this.frameLength = config["frameLength"];
          this.frameStep = config["frameStep"];
        }

        computeOutputShape(_inputShape) {
          return [null, null, this.frameLength / 2 + 1];
        }

        call(yList) {
          if (Array.isArray(yList)) {
            // TFJS wraps inputs into lists...
            var y = yList[0];
          } else {
            var y = yList;
          }
          if (y.shape[0] !== 1) {
            throw "TFJS's stft function only supports 1 dimension.";
          }
          y = y.flatten();

          // Resample.
          // This is not part of the python implementation of Spectrogram.
          // Compute the 1 / new sample rate, as if the original sampling
          // rate were exactly 1. This makes some stuff easier.
          const relativeFs = modelSampleRate / audioCtx.sampleRate;
          // Hack our own linspace(), the built-in version is woefully broken
          const xResampled = tf
            .range(0, [Math.ceil(y.size * relativeFs)])
            .mul(1 / relativeFs);
          // Assuming the original sampling rate is 1 means that the
          // relevant bounding indexes are to the left/right of the
          // truncated form of the new x sample locations.
          const yLeftIndexes = xResampled
            .asType("int32")
            .clipByValue(0, y.shape[0] - 1)
            .asType("int32");
          const yRightIndexes = tf.add(yLeftIndexes, tf.scalar(1, "int32"));
          // The mixture (alpha) value is also 1 - the decimal part
          const rightSidedAlphas = tf.sub(
            1,
            tf.sub(xResampled, xResampled.asType("int32"))
          );
          // Do the linear interpolation at each point.
          const yResampled = tf.add(
            tf.mul(y.gather(yLeftIndexes), tf.sub(1, rightSidedAlphas)),
            tf.mul(y.gather(yRightIndexes), rightSidedAlphas)
          );

          // Do the actual spectrogram now...
          return tf
            .sqrt(
              tf.abs(
                tf.signal.stft(yResampled, this.frameLength, this.frameStep)
              )
            )
            .expandDims();
        }

        static get className() {
          return "Spectrogram";
        }
      }
      tf.serialization.registerClass(Spectrogram);

      // Spectrogram must be registered before calling loadModel().
      loadModel().then(function(model) {
        // Modify topology so we can get the spectrogram output
        const modelWithSpectrogram = tf.model({
          inputs: model.input,
          outputs: [
            model.output,
            model.getLayer("spectrogram_Spectrogram1").output
          ]
        });
        var audioBuffer = new Array(audioCtx.sampleRate * 10).fill(0);

        scriptNode = audioCtx.createScriptProcessor(16384, 1, 1);
        scriptNode.onaudioprocess = function(event) {
          audioBuffer = audioBuffer
            .slice(event.inputBuffer.getChannelData(0).length)
            .concat(Array.from(event.inputBuffer.getChannelData(0)));

          modelOut = modelWithSpectrogram.predict(
            tf.tensor(audioBuffer, [1, audioBuffer.length])
          );
          probs = modelOut[0].arraySync()[0];
          ffts = modelOut[1];
          ffts = ffts
            .transpose([0, 2, 1])
            .flatten()
            .div(tf.add(ffts.max(), 0.000001))
            .mul(255)
            .asType("int32")
            .arraySync();
          for (var i = 3; i < myImageData.data.length; i += 4) {
            myImageData.data[i] = ffts[Math.floor(i / 4)];
          }
          canvasCtx.putImageData(myImageData, 0, 0);
          console.log(probs);
          functionOutput.innerText = probs;
        };

        source.connect(scriptNode);
        scriptNode.connect(audioCtx.destination);
      });
    })
    .catch(function(err) {
      console.log("Error with getUserMedia: ", err);
    });
} else {
  alert("Your browser is not supported.");
}
