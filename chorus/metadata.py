from __future__ import annotations

import json
from collections import defaultdict

import pandas as pd

from chorus.config import DATA_FOLDER


def xeno_canto() -> pd.DataFrame:
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
    for col in ["lat", "lng", "alt"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
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


def get_sci2en() -> dict[str, str]:
    filepath = DATA_FOLDER / "sci2en.json"
    if not filepath.exists():
        with open(filepath, "w") as f:
            json.dump(_generate_scientific_to_en(xeno_canto()), f)
    with open(filepath) as f:
        sci2en = json.load(f)
    return defaultdict(lambda: "unknown", sci2en)


def get_en2sci() -> dict[str, str]:
    filepath = DATA_FOLDER / "sci2en.json"
    if not filepath.exists():
        with open(filepath, "w") as f:
            json.dump(_generate_scientific_to_en(xeno_canto()), f)
    with open(filepath) as f:
        sci2en = json.load(f)
    return defaultdict(
        lambda: "unknown", {val: key for key, val in sci2en.items()}
    )


def _generate_scientific_to_en(df: pd.DataFrame) -> dict[str, str]:
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


def range_map() -> pd.DataFrame:
    """
    Load and return the range map information.

    More info:
    https://cornelllabofornithology.github.io/ebirdst/index.html
    Click "Introduction - Data Access and Structure" for info on the data.
    """
    return pd.read_csv(
        DATA_FOLDER / "ebird" / "range-meta" / "ebirdst_run_names.csv"
    )
