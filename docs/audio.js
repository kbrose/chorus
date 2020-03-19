const functionOutput = document.querySelector(".function-output");

var AudioContext = window.AudioContext || window.webkitAudioContext;

if (navigator.mediaDevices) {
    console.log("getUserMedia supported.");
    navigator.mediaDevices
        .getUserMedia({ audio: true })
        .then(function(stream) {
            const audioCtx = new AudioContext();
            const source = audioCtx.createMediaStreamSource(stream);

            var audioBuffer = new Array(audioCtx.sampleRate * 20).fill(0);

            scriptNode = audioCtx.createScriptProcessor(16384, 1, 1);
            scriptNode.onaudioprocess = function(event) {
                audioBuffer = audioBuffer
                    .slice(event.inputBuffer.length)
                    .concat(Array.from(event.inputBuffer.getChannelData(0)));

                // 1. Resample to desired sample rate (30,000 Hz)
                // 2. Call model on audioBuffer
                functionOutput.innerHTML = audioBuffer[0];
            };

            source.connect(scriptNode);
            scriptNode.connect(audioCtx.destination);
        })
        .catch(function(err) {
            console.log("The following gUM error occured: " + err);
        });
} else {
    console.log("getUserMedia not supported on your browser!");
}
