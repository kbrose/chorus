def load_saved_xeno_canto_meta() -> pd.DataFrame:
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
            lambda x: "0:" + x if x.count(":") == 1 else x  # put in hour if needed
        )
    ).dt.seconds
    df["scientific-name"] = df["gen"] + " " + df["sp"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["week"] = ((df["date"].dt.dayofyear // 7) + 1).clip(1, 52)
    return df
