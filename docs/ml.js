importScripts(
  "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@1.7.0/dist/tf.min.js"
);

const modelSampleRate = 30000;
const audioCtxSampleRate = 44100; // TODO: remove this hard coding

async function loadModel() {
  return await tf.loadLayersModel("/chorus/models/model.json");
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
    const relativeFs = modelSampleRate / audioCtxSampleRate;
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
        tf.abs(tf.signal.stft(yResampled, this.frameLength, this.frameStep))
      )
      .expandDims();
  }

  static get className() {
    return "Spectrogram";
  }
}
tf.serialization.registerClass(Spectrogram);

var model = undefined;
loadModel().then(m => {
  console.log("model loaded");
  model = tf.model({
    inputs: m.input,
    outputs: [m.output, m.getLayer("spectrogram_Spectrogram1").output]
  });
});

onmessage = function(e) {
  if (model === undefined) {
    // The model hasn't finished loading yet.
    this.postMessage([null, null]);
  }
  modelOut = model.predict(
    tf.tensor(e.data.audioBuffer, [1, e.data.audioBuffer.length])
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
  postMessage([probs, ffts]);
};
