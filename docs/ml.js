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

    // Those are the theoretical levels, but we need to adjust for
    // variable sampling rates.
    this.actualFrameLength = Math.ceil(
      (this.frameLength * audioCtxSampleRate) / modelSampleRate
    );
    this.actualFrameStep = Math.ceil(
      (this.frameStep * audioCtxSampleRate) / modelSampleRate
    );
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

    return tf
      .sqrt(
        tf.abs(tf.signal.stft(y, this.actualFrameLength, this.actualFrameStep))
      )
      .slice([0, 0], [-1, 257]) // Implicitly resample by ignoring higher freqs
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
    .reverse(1)
    .flatten()
    .div(tf.add(ffts.max(), 0.000001))
    .mul(255)
    .asType("int32")
    .arraySync();
  postMessage([probs, ffts]);
};
