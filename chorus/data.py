from pathlib import Path
import time
from typing import Iterable, Tuple
import json
import multiprocessing as mp
import warnings
from functools import partial

import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
import librosa

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
    df['lat'] = df['lat'].astype(float, errors='ignore')
    df['lng'] = df['lng'].astype(float, errors='ignore')
    df['alt'] = df['alt'].astype(float, errors='ignore')
    df['scientific-name'] = df['gen'] + ' ' + df['sp']
    return df


def _load_audio(file: Path, sample_rate: int) -> Tuple[Path, np.ndarray]:
    return file, librosa.load(file, sr=sample_rate)[0]


def convert_to_numpy(sample_rate: int, progress=True):
    (DATA_FOLDER / 'xeno-canto' / 'numpy').mkdir(exist_ok=True, parents=True)
    files = list((DATA_FOLDER / 'xeno-canto' / 'audio').glob('*'))
    with mp.Pool(6) as pool:
        for f, x in tqdm(
            pool.imap_unordered(
                partial(_load_audio, sample_rate=sample_rate), files
            ),
            total=len(files),
            disable=not progress
        ):
            np.save(DATA_FOLDER / 'xeno-canto' / 'numpy' / f'{f.stem}.npy', x)
