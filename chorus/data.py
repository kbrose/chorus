from collections import defaultdict
from pathlib import Path
import time
from typing import Dict, Iterable, Tuple, Optional
import json
import multiprocessing as mp
import warnings
from functools import partial
import io

import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
import librosa
import rasterio
import rasterio.windows

from chorus._typing import XenoCantoRecording, XenoCantoResponse

DATA_FOLDER = Path(__file__).parents[1] / 'data'
SECONDS_BETWEEN_REQUESTS = 0.2
XENO_CANTO_URL = (
    'https://www.xeno-canto.org/api/2/recordings'
    '?query=cnt:"United States"&page={page}'
)

warnings.filterwarnings(
    'ignore', 'PySoundFile failed. Trying audioread instead.'
)


def get_all_xeno_canto_meta(progress=False) -> Iterable[XenoCantoRecording]:
    """
    Get all the meta data for USA-based recordings on xeno-canto.

    Inputs
    ------
    progress : bool
        Whether or not a progress bar should be displayed.

    Yields
    ------
    recording : XenoCantoRecording
        A dictionary with meta data on a recording from xeno-canto.
    """
    r: XenoCantoResponse = requests.get(XENO_CANTO_URL.format(page=1)).json()
    with tqdm(total=int(r['numRecordings']), disable=not progress) as pbar:
        for recording in r['recordings']:
            yield recording
            pbar.update()
        num_pages = r['numPages']
        for page in range(2, num_pages + 1):
            time.sleep(SECONDS_BETWEEN_REQUESTS)
            r = requests.get(XENO_CANTO_URL.format(page=page)).json()
            for recording in r['recordings']:
                yield recording
                pbar.update()


def save_all_xeno_canto_meta(progress=True):
    """
    Saves all the meta data for USA-based recordings on xeno-canto to disk.

    Files will be placed in `DATA_FOLDER / 'xeno-canto' / 'meta'`.
    The names will be the xeno-canto ID, with ".json" appended.

    Inputs
    ------
    progress : bool
        Whether or not a progress bar should be displayed.
    """
    folder = DATA_FOLDER / 'xeno-canto' / 'meta'
    folder.mkdir(parents=True, exist_ok=True)
    for recording_meta in get_all_xeno_canto_meta(progress):
        with open(folder / f'{recording_meta["id"]}.json', 'w') as f:
            json.dump(recording_meta, f, indent=2)


def save_all_xeno_canto_audio(progress=True, skip_existing=True):
    """
    Download the audio recordings from xeno-canto.

    Assumes the meta data has already been downloaded.

    Inputs
    ------
    progress : bool
        Whether or not a progress bar should be displayed.
    skip_existing : bool
        If True and the audio file exists, do not re-download.
    """
    meta_folder = DATA_FOLDER / 'xeno-canto' / 'meta'
    meta_files = list(meta_folder.glob('*.json'))
    audio_folder = DATA_FOLDER / 'xeno-canto' / 'audio'
    audio_folder.mkdir(parents=True, exist_ok=True)
    audio_file_stems = [f.stem for f in audio_folder.glob('*')]
    for meta_path in tqdm(meta_files, disable=not progress):
        if meta_path.stem in audio_file_stems:
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        try:
            r = requests.get(f'https:{meta["file"]}')
            filename = meta['id'] + '.' + meta['file-name'].split('.')[-1]
            with open(audio_folder / filename, 'wb') as f:
                f.write(r.content)
        except Exception as e:
            print(f'Problem downloading id {meta["id"]}: {e}')
        time.sleep(SECONDS_BETWEEN_REQUESTS)


def load_saved_xeno_canto_meta() -> pd.DataFrame:
    """
    Load the previously saved xeno-canto meta data.

    Returns
    -------
    df : pd.DataFrame
        The lightly processed dataframe obtained from the JSON files.
    """
    folder = DATA_FOLDER / 'xeno-canto' / 'meta'
    metas = []
    for filepath in folder.glob('*.json'):
        with open(filepath) as f:
            metas.append(json.load(f))
    df = pd.DataFrame(metas)
    df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
    df['lng'] = pd.to_numeric(df['lng'], errors='coerce')
    df['alt'] = pd.to_numeric(df['alt'], errors='coerce')
    df['length-seconds'] = pd.to_timedelta(df['length'].apply(
        lambda x: '0:' + x if x.count(':') == 1 else x  # put in hour if needed
    )).dt.seconds
    df['scientific-name'] = df['gen'] + ' ' + df['sp']
    return df


def scientific_to_en(df: pd.DataFrame) -> Dict[str, str]:
    """
    Create mapping of scientific name to english name.

    If the mapping is not known, defaults to the string 'unknown'

    Inputs
    ------
    df : pd.DataFrame
        Must have columns 'scientific-name' and 'en'.
        Typically, this will be the output of `load_saved_xeno_canto_meta`.

    Returns
    -------
    mapping : Dict[str, str]
        Maps scientific names to english names.
    """
    mapping = (
        df.drop_duplicates(subset=['scientific-name'])
        .set_index('scientific-name')['en']
        .to_dict()
    )
    return defaultdict(lambda: 'unknown', mapping)


def _load_audio(file: Path, sr: int) -> Tuple[Path, Optional[np.ndarray]]:
    """
    Load the audio file. Used for multiprocessing.Pool() applications.
    """
    try:
        return file, librosa.load(file, sr=sr)[0]
    except Exception as e:
        print(f'Exception with file {file}: {e}', flush=True)
        return file, None


def convert_to_numpy(sample_rate: int, progress=True, skip_existing=True):
    """
    Convert the audio files to numpy files (np.save()).

    These files are loaded much faster (approximately 800 times faster).
    Unfortunately they are larger than MP3 files by a factor of ~6.

    Inputs
    ------
    sample_rate : int
        Target sampling rate of the audio.
    progress : bool
        Print a progress bar?
    skip_existing : bool
        If True, skip the saving of audio files which already have a
        corresponding file in the `numpy/` folder.
    """
    xc_folder = DATA_FOLDER / 'xeno-canto'
    (xc_folder / 'numpy').mkdir(exist_ok=True, parents=True)
    files = list((xc_folder / 'audio').glob('*'))
    total = len(files)
    initial = 0
    if skip_existing:
        done = set(f.stem for f in (xc_folder / 'numpy').glob('*'))
        files = [f for f in files if f.stem not in done]
        initial = len(done)
    with mp.Pool() as pool:
        for f, x in tqdm(
            pool.imap_unordered(partial(_load_audio, sr=sample_rate), files),
            total=total,
            initial=initial,
            smoothing=0.025,
            disable=not progress
        ):
            if x is not None:
                np.save(xc_folder / 'numpy' / f'{f.stem}.npy', x)


def save_range_map_meta():
    """
    Download the meta data (CSV table) for the range maps.
    This file lets us know the URLs of the raster map images.

    More info:
    https://cornelllabofornithology.github.io/ebirdst/index.html
    Click "Introduction - Data Access and Structure" for info on the data.
    """
    r = requests.get(
        'https://s3-us-west-2.amazonaws.com/ebirdst-data/ebirdst_run_names.csv'
    )
    folder = DATA_FOLDER / 'ebird' / 'range-meta'
    folder.mkdir(exist_ok=True, parents=True)
    with open(folder / 'ebirdst_run_names.csv', 'w') as f:
        f.write(r.text)


def load_range_map_meta() -> pd.DataFrame:
    """
    Load and return the range map information.

    More info:
    https://cornelllabofornithology.github.io/ebirdst/index.html
    Click "Introduction - Data Access and Structure" for info on the data.
    """
    return pd.read_csv(
        DATA_FOLDER / 'ebird' / 'range-meta' / 'ebirdst_run_names.csv'
    )


def save_range_maps(progress=True, skip_existing=True):
    """
    Save all of the ebird range map geoTiff files. Files will be
    filtered to the USA to reduce the size.

    You must save the range map meta data using save_range_map_meta()
    before you run this function.

    This function uses several approaches to reduce the disk space used
    up by the saved range maps. While this is nice in it's own right,
    the primary purpose is to speed up reads. The optimizations here
    speed up a query of a single lat/long point for _Spiza americana_
    from ~900ms to ~35ms, at the cost of some precision.

    More info:
    https://cornelllabofornithology.github.io/ebirdst/index.html
    Click "Introduction - Data Access and Structure" for info on the data.
    """
    df = load_range_map_meta()
    filename = '{run}_hr_2018_occurrence_median.tif'
    url = 'https://s3-us-west-2.amazonaws.com/ebirdst-data/{run}/results/tifs/'

    folder = DATA_FOLDER / 'ebird' / 'range'
    folder.mkdir(exist_ok=True, parents=True)

    # This window into the data was found through EDA. See the
    # notebooks/range-maps.ipynb file. Sampling the data to this
    # window reduces the dataset size by 2 orders of magnitude (100GB to 4GB).
    win = rasterio.windows.Window(2950, 1400, 2200, 1104)
    # We down sample by a factor of 8 in each spatial dimension, further
    # reducing the dataset size by a factor of 16.
    down_ratio = 8
    if (win.height % down_ratio) or (win.width % down_ratio):
        raise ValueError(
            f'{down_ratio=} must be a factor of {win.height=} and {win.width=}'
        )
    for run in tqdm(df['run_name'].values, disable=not progress):
        if skip_existing and (folder / filename.format(run=run)).exists():
            continue
        try:
            full_url = url.format(run=run) + filename.format(run=run)
            r = requests.get(full_url)
        except requests.ConnectionError:
            print(f'\nFailed to GET {full_url}')
            continue

        # This code was adapted from the docs:
        # https://rasterio.readthedocs.io/en/latest/topics/windowed-rw.html
        with rasterio.open(io.BytesIO(r.content), driver='GTiff') as raster:
            t = rasterio.windows.transform(win, raster.transform)
            transform = rasterio.Affine(
                t.a * down_ratio, t.b, t.c, t.d, t.e * down_ratio, t.f
            )

            kwargs = raster.meta.copy()
            kwargs.update({
                'height': win.height // down_ratio,
                'width': win.width // down_ratio,
                'transform': transform,
                'compress': 'deflate',
                'dtype': 'uint8'
            })
            data = raster.read(window=win)
            # -inf is used for missing values (e.g. over oceans),
            # but this really screws up the averaging, especially near
            # the coast. Replace these with nan and use nanmean().
            data = np.nan_to_num(data, nan=np.nan, neginf=np.nan)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', 'Mean of empty slice')
                data = np.nanmean(
                    [
                        data[:, i::down_ratio, j::down_ratio]
                        for i in range(down_ratio)
                        for j in range(down_ratio)
                    ],
                    axis=0
                )
            # The final space saving measure us to cast the data to uint8,
            # for a final factor of 4 (over float32, the default).
            data = np.nan_to_num(data, nan=0.0, neginf=0.0, posinf=0.0)
            data = (data * 255).astype('uint8')

            with rasterio.open(
                folder / filename.format(run=run), 'w', **kwargs
            ) as new_raster:
                new_raster.write(data)
