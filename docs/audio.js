async function loadModel() {
    return await tf.loadLayersModel("/models/model.json");
}

const functionOutput = document.querySelector(".function-output");

var AudioContext = window.AudioContext || window.webkitAudioContext;

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
            throw "TFJS's stft function only supports 1 dimension."
        }
        return tf.sqrt(
            tf.abs(
                tf.signal.stft(
                    x.flatten(),
                    this.frameLength,
                    this.frameStep
                )
            )
        ).expandDims();
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

                    var audioBuffer = new Array(audioCtx.sampleRate * 20).fill(
                        0
                    );

                    scriptNode = audioCtx.createScriptProcessor(16384, 1, 1);
                    scriptNode.onaudioprocess = function(event) {
                        audioBuffer = audioBuffer
                            .slice(event.inputBuffer.length)
                            .concat(
                                Array.from(event.inputBuffer.getChannelData(0))
                            );

                        // 1. Resample to desired sample rate (30,000 Hz)
                        // 2. Call model on audioBuffer
                        functionOutput.innerHTML = model.predict(
                            tf.tensor(audioBuffer, [1, audioBuffer.length])
                        );
                        console.log(audioBuffer[0])
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
