const functionOutput = document.querySelector(".function-output");

var AudioContext = window.AudioContext || window.webkitAudioContext;

// getUserMedia block - grab stream
// put it into a MediaStreamAudioSourceNode
if (navigator.mediaDevices) {
    console.log("getUserMedia supported.");
    navigator.mediaDevices
        .getUserMedia({ audio: true })
        .then(function(stream) {
            const audioCtx = new AudioContext();
            const source = audioCtx.createMediaStreamSource(stream);

            scriptNode = audioCtx.createScriptProcessor(4096, 1, 1);
            scriptNode.onaudioprocess = function(audioProcessingEvent) {
                var inputBuffer = audioProcessingEvent.inputBuffer;

                var x = inputBuffer.getChannelData(0);
                functionOutput.innerHTML = x[0];
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
