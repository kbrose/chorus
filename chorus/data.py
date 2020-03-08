from pathlib import Path
import time
from typing import Iterable
import json

import pandas as pd
import requests
from tqdm import tqdm

from chorus._typing import XenoCantoRecording, XenoCantoResponse

DATA_FOLDER = Path(__file__).parents[1] / 'data'
SECONDS_BETWEEN_REQUESTS = 0.2
XENO_CANTO_URL = (
    'https://www.xeno-canto.org/api/2/recordings'
    '?query=cnt:"United States"&page={page}'
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
