from pathlib import Path
import time
from typing import Iterable
import json

import requests
from tqdm import tqdm

from chorus._typing import XenoCantoRecording, XenoCantoResponse

DATA_FOLDER = Path(__file__).parents[1] / 'data'
SECONDS_BETWEEN_REQUESTS = 0.2
XENO_CANTO_URL = (
    'https://www.xeno-canto.org/api/2/recordings'
    '?query=cnt:"United States"&page={page}'
)


def _save_recording_meta(meta: dict):
    pass


def get_all_xeno_canto_meta(progress=False) -> Iterable[XenoCantoRecording]:
    r: XenoCantoResponse = requests.get(XENO_CANTO_URL.format(page=1)).json()
    with tqdm(
        total=int(r['numRecordings']), disable=not progress, smoothing=0
    ) as pbar:
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
    folder = DATA_FOLDER / 'xeno-canto' / 'meta'
    folder.mkdir(parents=True, exist_ok=True)
    for recording_meta in get_all_xeno_canto_meta(progress):
        with open(folder / f'{recording_meta["id"]}.json', 'w') as f:
            json.dump(recording_meta, f, indent=2)
