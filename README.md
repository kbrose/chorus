# chorus
Determine the bird from its song.

## Performance

(Performance data as of commit [cecb3f933](https://github.com/kbrose/chorus/commit/cecb3f9331689e8c604236460efbf79a11501f6e) using xeno canto data as of February 20th, 2024.)

The current model uses a resnet-style architecture on the raw audio waveform (sampled at 30,000 Hz).

Across the 264 species with enough data, the model is able to pick out the correct species _exactly_ about 45% of the time. If given five attempts, the model can guess correctly about 70% of the time.

**NOTE:** This is on data that looks similar to the training data, specifically, most of the tested audio files were recorded with higher end equipment than your average smartphone.

Adjusting the model's score by the odds that the species is present in that area during that time of the year (using ebird's range maps) provides a slight improvement.

![](./static/top-n.png)

_This figure shows where the model ranked the correct species (with ranks along the x axis). The top plot is a histogram of the raw counts, and the bottom plot shows the cumulative percent covered. Orange indicates rankings that were adjusted according to the geographic data, with blue indicating ranks obtained purely from the audio model._

![](./static/roc.png)

_This figure shows the ROC curves for all 205 species. It is useful to get a sense of overall performance in a one-vs-rest scenario._

This performance is on audio recordings that are fairly long (a majority are over 30 seconds long) and often contain multiple species. If I could tell the model exactly what portion of each recording corresponds to each species, the model would likely learn better.

Related work, such as the Cornell Lab of Ornithology's model, has [used human labelers to take exactly that approach](https://www.macaulaylibrary.org/2021/06/22/behind-the-scenes-of-sound-id-in-merlin/?doing_wp_cron=1625711942.0293428897857666015625). Their data is not open, unfortunately.

## Data

Audio recordings from [Xeno Canto](https://www.xeno-canto.org) are used. Species range maps from [ebird](https://cornelllabofornithology.github.io/ebirdst/articles/ebirdst-introduction.html) are also used. The Xeno Canto data is really quite nice, all of it is CC-licensed, and their meta data has really good coverage for attributes like date and location of recording, the main species captured, other species in the recordings, and the type (song/call, adult/juvenile, etc.).

Macaulay Library and Avocet were considered as other audio sources, but Macaulay Library requires a licensing agreement including the stipulation that you cannot build a (non-commercial or commercial) "product" based on their data (product was left undefined and I assume can cover just about anything), and Avocet just doesn't have many recordings.

It's also worth noting that the Cornell Lab of Ornithology (who runs the Macaulay Library) have hosted bird-audio-recognition competitions in the past and [_have used Xeno Canto's data instead of their own_](https://www.kaggle.com/c/birdclef-2021/data). In other words, all signs point to Xeno Canto being _the_ resource to use.

See the section "Getting the data" below if you want to download it yourself.

## Developing

### Python dependencies
g
This code was tested on python v3.12.2.

This project uses `pip-tools` to track requirements. It's recommended, but not required, to run the code.

```bash
# Bare minimum for running the code.
pip-sync requirements.txt
# alternatively, pip install -r requirements.txt

# To get packages needed for development / data downloading as well:
pip-sync dev-requirements.txt
```

If you want to try and be looser with the package versions, you can just reference the corresponding `.in` file.

### ffmpeg

You must install `ffmpeg` for the audio loading.

### gdal

In order to process the species range maps, we use the `rasterio` package. You [need to install gdal](https://rasterio.readthedocs.io/en/latest/installation.html#linux) before installing `rasterio`.:

```bash
sudo apt-get install gdal-bin libgdal-dev
```

### llvm

LLVM may also need to be installed, if it is not already.

### Getting the data

Use the command line interface. ***You'll need about 600 GB of free space and a few days to download the data.***

```bash
######### For Audio
# First get the meta data (about 200MB)
python cli.py data xc-meta --help

# Next download the audio (about 80GB, ~24 hours)
python cli.py data xc-audio --help

# Finally, convert audio to numpy format (about 480GB, ~24 hours depending on your CPU)
# This format takes up more space, but loads 800 times faster.
python cli.py data xc-to-npy --help

######### For Range maps
# First get the meta data (about 100KB)
python cli.py data range-meta --help

# Then download and process the range maps (about 3GB)
# Range maps are processed in memory and the full res versions
# are never persisted to save on space.
python cli.py data range-map --help

######### For background (ambient) noise to augment (200MB)
python cli.py data background
```

### Training the model

```
python cli.py train classifier <model name>
```

You can monitor progress with tensorboard:

```
tensorboard --logdir=./logs --samples_per_plugin images=150
```

### Running the model

```
python cli.py run classifier path/to/model/folder path/to/audio/file <optional location/date info>
```
