from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import affine
import numpy as np
import pandas as pd
from pyproj import Transformer

DATA_FOLDER = Path(__file__).parents[1] / "data"


def xeno_canto_meta() -> pd.DataFrame:
    """
    Load the previously saved xeno-canto meta data.

    Returns
    -------
    df : pd.DataFrame
        The lightly processed dataframe obtained from the JSON files.
    """
    folder = DATA_FOLDER / "xeno-canto" / "meta"
    metas = []
    for filepath in folder.glob("*.json"):
        with open(filepath) as f:
            metas.append(json.load(f))
    df = pd.DataFrame(metas)
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lng"] = pd.to_numeric(df["lng"], errors="coerce")
    df["alt"] = pd.to_numeric(df["alt"], errors="coerce")
    df["length-seconds"] = pd.to_timedelta(
        df["length"].apply(
            lambda x: "0:" + x
            if x.count(":") == 1
            else x  # put in hour if needed
        )
    ).dt.seconds
    df["scientific-name"] = df["gen"] + " " + df["sp"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["week"] = ((df["date"].dt.dayofyear // 7) + 1).clip(1, 52)
    return df


def scientific_to_en(df: pd.DataFrame) -> dict[str, str]:
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
        df.drop_duplicates(subset=["scientific-name"])
        .set_index("scientific-name")["en"]
        .to_dict()
    )
    return defaultdict(lambda: "unknown", mapping)


def range_map_meta() -> pd.DataFrame:
    """
    Load and return the range map information.

    More info:
    https://cornelllabofornithology.github.io/ebirdst/index.html
    Click "Introduction - Data Access and Structure" for info on the data.
    """
    return pd.read_csv(
        DATA_FOLDER / "ebird" / "range-meta" / "ebirdst_run_names.csv"
    )


class Presence:
    """
    What are the odds that a bird is present in a location?
    """

    def __init__(self):
        self._folder = DATA_FOLDER / "ebird" / "range"

        with open(self._folder / "meta.json") as f:
            meta = json.load(f)
        self.scientific_names = meta["scientific_names"]

        # lat/long -> coordinate system of raster
        transformer = Transformer.from_crs("EPSG:4326", meta["crs"])
        self._crs_transform = transformer.transform
        # coordinate system of raster -> array indices
        self._raster_transform = ~affine.Affine(*meta["transform"])

    def __call__(
        self,
        *,
        lat: float,
        lng: float,
        week: int,
    ) -> dict[str, float]:
        """
        Return the probability that a bird is present, or nan if unknown.

        The EBird folks define this probability as follows:

            This [value] represents the expected probability of occurrence
            of the species, ranging from 0 to 1, on an eBird Traveling Count
            by a skilled eBirder starting at the optimal time of day with
            the optimal search duration and distance that maximizes detection
            of that species in a region.

        https://cornelllabofornithology.github.io/ebirdst/articles/ebirdst-introduction.html#occurrence_median

        Inputs
        ------
        lat, lng : float
            Location to query.
        week : int
            The week number from 1 to 52 inclusive. There are slightly more
            than 52 weeks in a year, so make sure you handle your edge
            cases.

        Returns
        -------
        probabilities : Dict[str, float]
            A map of scientific name -> probability that the bird can be
            observed at the given location at the given time of year.
            Will be NaN if you request a lat/lng point outside the range
            of the data.
            See above for rigorous definition of the probability.
        """
        if not 1 <= week <= 52:
            raise ValueError("week must be between 1 and 52 inclusive")

        # "load" data from disk (use mmap to do lazy loading of range data)
        data = np.load(self._folder / "ranges.npy", mmap_mode="r")

        try:
            col, row = self._raster_transform * self._crs_transform(lat, lng)
            values = data[:, int(week), int(row), int(col)]
        except IndexError:
            values = np.nan * np.ones(len(self.scientific_names))
        return dict(zip(self.scientific_names, values))
