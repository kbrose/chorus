This folder contains code and information necessary to deploy the model onto a raspberry pi.

# Pre-requisites

1. A trained model
1. A machine running some kind of linux with `systemd` and `arecord` programs installed
    * This code was tested on a [Raspberry Pi 4](https://www.raspberrypi.com/products/raspberry-pi-4-model-b/) running the "Raspberry Pi OS 64-bit port of Debian Bookworm with the Raspberry Pi Desktop"
1. A microphone attached to the machine and visible to `arecord`
1. Free time

# Goals

A LAN-accessible website that:

1. Shows real time bird identification every _n_ (15?) seconds
1. Visualizes historical bird identifications
    * Long term view
    * Per-audio-file view
1. Enables correction annotations
1. Can download audio files
1. Can download historical inference results
1. Monitors RAM, CPU, temperature, disk space

with a backend that also:

1. Has automatic restarts
1. Cleans up old audio files
1. Backs up to the cloud???

Optionally, an initial set up workflow that:

1. Downloads a model?
1. Lets you select the microphone
1. Set up a cloud backup solution?

# Notes

Record audio indefinitely, saving to a file every 5 seconds:

```bash
arecord -D plughw:CARD=Mic,DEV=0 -f cd --max-file-time 5 --use-strftime "%Y-%m-%d-%H-%M-%S.wav"
```
