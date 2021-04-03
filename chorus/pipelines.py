import io
import json
import multiprocessing as mp
import random
import time
import warnings
from functools import partial
from itertools import chain
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from tempfile import TemporaryDirectory

import librosa
import numpy as np
import rasterio
import rasterio.windows
import requests
from tqdm import tqdm
from typing_extensions import Literal, TypedDict

from chorus import metadata
from chorus.config import DATA_FOLDER

SECONDS_BETWEEN_REQUESTS = 0.2
XENO_CANTO_URL = (
    "https://www.xeno-canto.org/api/2/recordings"
    '?query=cnt:"United States"&page={page}'
)
_XenoCantoRecording = TypedDict(
    "_XenoCantoRecording",
    {
        "id": str,
        "gen": str,
        "sp": str,
        "ssp": str,
        "en": str,
        "rec": str,
        "cnt": str,
        "loc": str,
        "lat": str,
        "lng": str,
        "alt": str,
        "type": str,
        "url": str,
        "file": str,
        "file-name": str,
        "sono": Dict[Literal["small", "med", "large", "full"], str],
        "lic": str,
        "q": Literal["A", "B", "C", "D", "E"],
        "length": str,
        "time": str,
        "date": str,
        "uploaded": str,
        "also": List[str],
        "rmk": str,
        "bird-seen": Literal["yes", "no"],
        "playback-used": Literal["yes", "no"],
    },
)
_XenoCantoResponse = TypedDict(
    "_XenoCantoResponse",
    {
        "numRecordings": str,
        "numSpecies": str,
        "page": int,
        "numPages": int,
        "recordings": List[_XenoCantoRecording],
    },
)

warnings.filterwarnings(
    "ignore", "PySoundFile failed. Trying audioread instead."
)


def get_all_xeno_canto_meta(progress=False) -> Iterable[_XenoCantoRecording]:
    """
    Get all the meta data for USA-based recordings on xeno-canto.

    Inputs
    ------
    progress : bool
        Whether or not a progress bar should be displayed.

    Yields
    ------
    recording : _XenoCantoRecording
        A dictionary with meta data on a recording from xeno-canto.
    """
    r: _XenoCantoResponse = requests.get(XENO_CANTO_URL.format(page=1)).json()
    with tqdm(total=int(r["numRecordings"]), disable=not progress) as pbar:
        for recording in r["recordings"]:
            yield recording
            pbar.update()
        num_pages = r["numPages"]
        for page in range(2, num_pages + 1):
            time.sleep(SECONDS_BETWEEN_REQUESTS)
            r = requests.get(XENO_CANTO_URL.format(page=page)).json()
            for recording in r["recordings"]:
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
    folder = DATA_FOLDER / "xeno-canto" / "meta"
    folder.mkdir(parents=True, exist_ok=True)
    for recording_meta in get_all_xeno_canto_meta(progress):
        with open(folder / f'{recording_meta["id"]}.json', "w") as f:
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
    meta_folder = DATA_FOLDER / "xeno-canto" / "meta"
    meta_files = list(meta_folder.glob("*.json"))
    audio_folder = DATA_FOLDER / "xeno-canto" / "audio"
    audio_folder.mkdir(parents=True, exist_ok=True)
    numpy_folder = DATA_FOLDER / "xeno-canto" / "numpy"
    existing_files = set(
        f.stem for f in chain(audio_folder.glob("*"), numpy_folder.glob("*"))
    )
    if skip_existing:
        meta_files = [f for f in meta_files if f.stem not in existing_files]
    skipped = 0
    with tqdm(total=len(meta_files), disable=not progress, ncols=80) as pbar:
        for meta_path in meta_files:
            pbar.update()
            with open(meta_path) as f:
                meta = json.load(f)
            # Some files have restricted access. Skip these.
            if not meta["file-name"]:
                skipped += 1
                pbar.set_postfix_str(f"skipped = {skipped}")
                continue
            try:
                r = requests.get(f'https:{meta["file"]}')
                filename = meta["id"] + "." + meta["file-name"].split(".")[-1]
                with open(audio_folder / filename, "wb") as f:
                    f.write(r.content)
            except Exception as e:
                print(f'Problem downloading id {meta["id"]}: {e}')
            time.sleep(SECONDS_BETWEEN_REQUESTS)


def _load_audio(file: Path, sr: int) -> Tuple[Path, Optional[np.ndarray]]:
    """
    Load the audio file. Used for multiprocessing.Pool() applications.
    """
    try:
        return file, librosa.load(file, sr=sr)[0]
    except Exception as e:
        print(f"Exception with file {file}: {e}", flush=True)
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
    xc_folder = DATA_FOLDER / "xeno-canto"
    (xc_folder / "numpy").mkdir(exist_ok=True, parents=True)
    files = list((xc_folder / "audio").glob("*"))
    total = len(files)
    initial = 0
    if skip_existing:
        done = set(f.stem for f in (xc_folder / "numpy").glob("*"))
        files = [f for f in files if f.stem not in done]
        initial = len(done)
    with mp.Pool() as pool:
        for f, x in tqdm(
            pool.imap_unordered(partial(_load_audio, sr=sample_rate), files),
            total=total,
            initial=initial,
            smoothing=0.025,
            disable=not progress,
        ):
            if x is not None:
                np.save(xc_folder / "numpy" / f"{f.stem}.npy", x)


def save_range_map_meta():
    """
    Download the meta data (CSV table) for the range maps.
    This file lets us know the URLs of the raster map images.

    More info:
    https://cornelllabofornithology.github.io/ebirdst/index.html
    Click "Introduction - Data Access and Structure" for info on the data.
    """
    r = requests.get(
        "https://s3-us-west-2.amazonaws.com/ebirdst-data/ebirdst_run_names.csv"
    )
    folder = DATA_FOLDER / "ebird" / "range-meta"
    folder.mkdir(exist_ok=True, parents=True)
    with open(folder / "ebirdst_run_names.csv", "w") as f:
        f.write(r.text)


def save_range_maps(progress=True):
    """
    Download & reformat all of the ebird range map geoTiff files.
    The data will be massaged into a large numpy array and some meta data.
    This massaging brings the query time down from ~1 second / species to
    ~30ms for all 600 species, at the cost of some precision.

    You must save the range map meta data using save_range_map_meta()
    before you run this function.

    More info:
    https://cornelllabofornithology.github.io/ebirdst/index.html
    Click "Introduction - Data Access and Structure" for info on the data.
    """
    df = metadata.range_map()
    filename = "{run}_hr_2018_occurrence_median.tif"
    url = "https://s3-us-west-2.amazonaws.com/ebirdst-data/{run}/results/tifs/"

    folder = DATA_FOLDER / "ebird" / "range"
    folder.mkdir(exist_ok=True, parents=True)

    # This window into the data was found through EDA. See the
    # notebooks/range-maps.ipynb file. Sampling the data to this
    # window reduces the dataset size by 2 orders of magnitude (100GB to 4GB).
    win = rasterio.windows.Window(2950, 1400, 2200, 1100)
    # We down sample by a factor of 10 in each spatial dimension, further
    # reducing the dataset size by a factor of 100. The resulting grid
    # is squares of 15 miles x 15 miles (approximately).
    down_ratio = 10
    if (win.height % down_ratio) or (win.width % down_ratio):
        raise ValueError(
            f"{down_ratio=} must be a factor of {win.height=} and {win.width=}"
        )

    # This is where we'll save the output data
    data_out = np.zeros(
        (df.shape[0], 52, win.height // down_ratio, win.width // down_ratio),
        dtype="float32",
    )
    # These will be used to ensure that all transformations are the same
    transform = None
    crs = None
    for i, run in enumerate(tqdm(df["run_name"].values, disable=not progress)):
        full_url = url.format(run=run) + filename.format(run=run)
        for _ in range(3):
            try:
                r = requests.get(full_url)
                break
            except requests.ConnectionError:
                print(f"\nFailed to GET {full_url}")
                time.sleep(2)
                continue
        else:
            raise RuntimeError("Failed to download file.")

        # This code was adapted from the docs:
        # https://rasterio.readthedocs.io/en/latest/topics/windowed-rw.html
        with rasterio.open(io.BytesIO(r.content), driver="GTiff") as raster:
            t = rasterio.windows.transform(win, raster.transform)
            transform_new = rasterio.Affine(
                t.a * down_ratio, t.b, t.c, t.d, t.e * down_ratio, t.f
            )
            if transform is None:
                transform = transform_new
            elif transform != transform_new:
                raise RuntimeError(f"transform for {run} does not match")

            if crs is None:
                crs = raster.crs.wkt
            elif crs != raster.crs.wkt:
                raise RuntimeError(f"CRS for {run} does not match")

            data = raster.read(window=win)
            # -inf is used for missing values (e.g. over oceans),
            # but this really screws up the averaging, especially near
            # the coast. Replace these with nan and use nanmean().
            data = np.nan_to_num(data, nan=np.nan, neginf=np.nan)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", "Mean of empty slice")
                data = np.nanmean(
                    [
                        data[:, i::down_ratio, j::down_ratio]
                        for i in range(down_ratio)
                        for j in range(down_ratio)
                    ],
                    axis=0,
                )
            data_out[i, :, :, :] = data
    np.save(folder / "ranges.npy", data_out)
    with open(folder / "meta.json", "w") as f:
        json.dump(
            f,
            {
                "scientific_names": list(df["scientific_name"].values),
                "transform": list(np.array(transform[:-3])),
                "crs": crs,
            },
        )


def save_background_sounds(sample_rate: int):
    folder = DATA_FOLDER / "background"

    def download(filename_template, urls):
        for i, url in enumerate(urls):
            with TemporaryDirectory() as tmpdir:
                filepath = Path(tmpdir) / "tmp.mp3"
                with open(filepath, "wb") as f:
                    f.write(requests.get(url).content)
                    time.sleep(1.5 + random.random())
                x = librosa.load(filepath, sr=sample_rate)[0]
            np.save(folder / filename_template.format(i), x)

    sounds = {
        "wind": [
            "http://www.soundgator.com/adata/510.mp3",
            "http://www.soundgator.com/adata/467.mp3",
            "https://www.zapsplat.com/wp-content/uploads/2015/sound-effects-344-audio/344_audio_arctic_wind_filtered_rumble_006.mp3",
            "https://www.zapsplat.com/wp-content/uploads/2015/sound-effects-41945/zapsplat_nature_wind_blustery_trees_against_house_distant_surf_balcolny_46210.mp3",
            "https://www.zapsplat.com/wp-content/uploads/2015/sound-effects-four/nature_wind_trees_strong_park_traffic_in_background.mp3",
            "https://www.zapsplat.com/wp-content/uploads/2015/sound-effects-felix-blume/felix_blume_nature_wind_strong_blowing_through_grass_patagonian_plain.mp3",
            "https://www.zapsplat.com/wp-content/uploads/2015/sound-effects-four/ambience_city_london_uk_park_millbank_gardens_001.mp3",
        ],
        "city": [
            "http://www.soundgator.com/adata/460.mp3",
            "http://www.soundgator.com/adata/454.mp3",
            "https://www.zapsplat.com/wp-content/uploads/2015/sound-effects-four/ambience_city_london_uk_sirens_distant_constant_construction_pounding_wind_trees.mp3",
            "https://www.zapsplat.com/wp-content/uploads/2015/sound-effects-four/ambience_city_london_uk_sirens_distant_constant_construction_pounding_wind_trees.mp3",
            "https://www.zapsplat.com/wp-content/uploads/2015/sound-effects-the-sound-pack-tree/tspt_calm_town_background_traffic_loop_015.mp3",
            "https://www.zapsplat.com/wp-content/uploads/2015/sound-effects-pmsfx/PM_Spain_Alicante_Night_Plaza_De_La_Puerta_De_San_Francisco_Binaural_LoopSeamlessly_264.mp3",
        ],
        "running-water": [
            "https://www.zapsplat.com/wp-content/uploads/2015/sound-effects-felix-blume/felix_blume_nature_water_brook_close_up.mp3",
            "https://www.zapsplat.com/wp-content/uploads/2015/sound-effects-one/water_stream_waterfall_small_mountain_001.mp3",
            "https://www.zapsplat.com/wp-content/uploads/2015/sound-effects-one/water_stream_waterfall_small_mountain_003.mp3",
            "https://www.zapsplat.com/wp-content/uploads/2015/sound-effects-kevin-boucher/kevin_boucher_nature_creek_small_wate_tumbling_over_limestone_rock.mp3",
            "https://www.zapsplat.com/wp-content/uploads/2015/sound-effects-blastwave-fx/Blastwave_FX_LakeShore_BW.61003.mp3",
            "https://www.zapsplat.com/wp-content/uploads/2015/sound-effects-55112/zapsplat_nature_creek_fast_flowing_60m_below_australia_55314.mp3",
            "https://www.zapsplat.com/wp-content/uploads/2015/sound-effects-adam-a-johnson/aaj_0795_MtStrmRush01.mp3",
        ],
        "machinery": [
            "https://www.zapsplat.com/wp-content/uploads/2015/sound-effects-55112/zapsplat_industrial_grass_mower_buggy_commercial_cutting_grass_distant_then_close_002_56463.mp3",
            "https://www.zapsplat.com/wp-content/uploads/2015/sound-effects-free-to-use-sounds/ftus_vehicles_helicopter_military_eurocopter_AS332_super_puma_fly_overhead.mp3",
            "https://www.zapsplat.com/wp-content/uploads/2015/sound-effects-10157/zapsplat_transport_helicopter_pass_overhead_10989.mp3",
            "https://www.zapsplat.com/wp-content/uploads/2015/sound-effects-your-free-sounds/yfs_excavators_contruction_site_idles_9624_186.mp3",
            "https://www.zapsplat.com/wp-content/uploads/2015/sound-effects-your-free-sounds/yfs_excavators_contruction_site_moving_digging_9624_185.mp3",
        ],
    }

    for name, urls in sounds.items():
        download(name + "-{}.npy", urls)
