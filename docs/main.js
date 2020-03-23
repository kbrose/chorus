const functionOutput = document.getElementById("function-output");
const AudioContext = window.AudioContext || window.webkitAudioContext;

const canvasCtx = document.getElementById("spectrogram").getContext("2d");
const myImageData = canvasCtx.createImageData(669, 257);
for (var i = 0; i < myImageData.data.length; i += 4) {
  myImageData.data[i + 0] = 0;
  myImageData.data[i + 1] = 0;
  myImageData.data[i + 2] = 0;
  myImageData.data[i + 3] = 255;
}

const worker = new Worker("/chorus/ml.js");

const runButton = document.getElementById("controller");
runButton.addEventListener("click", main);

let audioCtx = undefined;
let targets;
fetch("/chorus/models/targets.json")
  .then(response => response.json())
  .then(json => (targets = json));

function main(_e) {
  if (audioCtx === undefined) {
    startup();
  } else if (audioCtx.state === "running") {
    pause();
  } else {
    resume();
  }
}

function startup() {
  audioCtx = new AudioContext();
  var audioBuffer = new Array(audioCtx.sampleRate * 10).fill(0);
  var workerReady = true;

  if (navigator.mediaDevices) {
    navigator.mediaDevices
      .getUserMedia({ audio: true })
      .then(function(stream) {
        const source = audioCtx.createMediaStreamSource(stream);

        scriptNode = audioCtx.createScriptProcessor(16384, 1, 1);
        scriptNode.onaudioprocess = function(e) {
          audioBuffer = audioBuffer
            .slice(e.inputBuffer.getChannelData(0).length)
            .concat(Array.from(e.inputBuffer.getChannelData(0)));
          if (workerReady) {
            worker.postMessage({ audioBuffer: audioBuffer });
            workerReady = false;
          }
        };

        source.connect(scriptNode);
        scriptNode.connect(audioCtx.destination);

        worker.onmessage = function(e) {
          workerReady = true;
          if (e.data[0] === null) {
            return;
          }
          probs = e.data[0];
          ffts = e.data[1];
          for (var i = 3; i < myImageData.data.length; i += 4) {
            myImageData.data[i] = ffts[Math.floor(i / 4)];
          }
          canvasCtx.putImageData(myImageData, 0, 0);
          txt = "";
          for (i = 0; i < targets.length; i++) {
            txt +=
              "<li>" + targets[i] + ": " + Math.round(probs[i] * 100) + "</li>";
          }
          functionOutput.innerHTML = txt;
        };
      })
      .catch(function(err) {
        console.log("Error with getUserMedia: ", err);
      });
  } else {
    alert("Your browser is not supported.");
  }
  runButton.textContent = "Pause";
}

function pause() {
  audioCtx.suspend();
  runButton.textContent = "Resume";
}

function resume() {
  audioCtx.resume();
  runButton.textContent = "Pause";
}
