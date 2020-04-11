# chorus
Determine the bird from its call

## Resources

* Databases
    * https://www.macaulaylibrary.org
        * Tons of recordings, but requires signup/license agreement.
    * https://avocet.integrativebiology.natsci.msu.edu
        * All recordings seem to be licensed under "Creative Commons Attribution-Noncommercial-Share Alike 3.0 United States License".
        * Files available in `.wav` format.
    * https://www.xeno-canto.org
        * Full rest API offered.
        * Seems like most recordings are "only" MP3 files.
        * Recordings typically (always?) licensed under with one of the CC licenses.

After reviewing, it seems like xeno-canto is a clear winner in the short term. If the Macaulay Library is willing to share their data they might be a good follow up.

*Update:* After receiving the terms of use from Macaulay Library, it seems like they are too restrictive:

> Macaulay Library media and data may not be reproduced, distributed, **or used to make products of any kind (whether commercial or noncommercial)** without prior written permission of the Cornell Lab of Ornithology.
> ...
> All Macaulay Library media assets are provided with additional supporting metadata sufficient to make sensible and informed decisions about data use.

(Emphasis mine.) I'll just stick with xeno-canto.

## Developing

### Python dependencies

This code requires python 3.7.

This project uses `pip-tools` to track requirements. It's recommended, but not required, to run the code.

```bash
# For running the code.
pip-sync requirements.txt
# alternatively, pip install -r requirements.txt

# To get packages that help with development as well:
pip-sync dev-requirements.txt

# If you just want to run tests, use the following.
pip-sync test-requirements.txt
```

If you want to try and be looser with the package versions, you can just reference the corresponding `.in` file. But be aware that tests are only run on the specific package versions referenced in the `.txt` files.

### ffmpeg

You must install `ffmpeg` for the audio loading.

### Getting the data

Use the command line interface. ***You'll need about 400 GB of free space and 48 hours.***

```bash
# First get the meta data (about 200MB)
python cli.py data xc-meta --help

# Next download the audio (about 60GB, ~24 hours)
python cli.py data xc-audio --help

# Finally, convert audio to numpy format (about 320GB, ~24 hours)
# This format takes up more space, but loads 800 times faster.
python cli.py data xc-to-npy --help
```
