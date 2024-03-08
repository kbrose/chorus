import datetime
from pathlib import Path

import click

from chorus import train as c_train
from chorus import pipelines

from chorus import infer
from chorus.config import SAMPLE_RATE


@click.group()
def cli():
    pass


@cli.group(help="subcommands to download data")
def data():
    pass


@data.command("xc-meta", help="download xeno-canto meta data")
@click.option("-v", "--verbose", is_flag=True, help="Show progress bar.")
def xeno_meta(verbose: bool):
    pipelines.save_all_xeno_canto_meta(verbose)


@data.command("xc-audio", help="download xeno-canto audio data")
@click.option("-v", "--verbose", is_flag=True, help="Show progress bar.")
@click.option("--redownload", is_flag=True, help="Redownload all files.")
def xeno_audio(verbose: bool, redownload: bool):
    pipelines.save_all_xeno_canto_audio(verbose, skip_existing=not redownload)


@data.command("xc-to-npy", help="convert audio to .npy data at [SAMPLERATE]")
@click.option(
    "--samplerate",
    type=int,
    default=SAMPLE_RATE,
    help="Edit chorus/config.py if you change this value!",
)
@click.option("--reprocess", is_flag=True, help="Reprocess all audio.")
@click.option("-v", "--verbose", is_flag=True, help="Show progress bar.")
def xeno_to_numpy(samplerate: int, reprocess: bool, verbose: bool):
    pipelines.convert_to_numpy(samplerate, verbose, not reprocess)


@data.command("range-meta", help="download range map meta data")
def range_meta():
    pipelines.save_range_map_meta()


@data.command("range-map", help="download range map data")
@click.option("-v", "--verbose", is_flag=True, help="Show progress bar.")
def range_maps(verbose: bool):
    pipelines.save_range_maps(verbose)


@data.command("background", help="download background audio files")
@click.option(
    "--samplerate",
    type=int,
    default=SAMPLE_RATE,
    help="Edit chorus/config.py if you change this value!",
)
def background(samplerate: int):
    pipelines.save_background_sounds(samplerate)


@cli.group(help="subcommands to train models")
def train():
    pass


@train.command(help="train the classifier model", name="classifier")
@click.argument("name", type=str)
def train_classifier(name: str):
    c_train.train_classifier(name)


@train.command(help="train the isolator model", name="isolator")
@click.argument("name", type=str)
@click.argument("classifier_filepath", type=str)
def train_isolator(name: str, classifier_filepath: str):
    c_train.train_isolator(name, classifier_filepath)


@train.command(
    help="export a classifier model as an optimized torchscript module",
    name="export-classifier",
)
@click.argument("model_in_path", type=Path)
@click.argument("model_out_path", type=Path)
def export_classifier(model_in_path: Path, model_out_path: Path):
    c_train.export_jitted_classifier(model_in_path, model_out_path)


@cli.group(help="run models on audio file")
def run():
    pass


@run.command(help="run classifier on audio file", name="classifier")
@click.argument("modelpath", type=Path)
@click.argument("audiofile", type=Path)
@click.option(
    "--latlng", default=None, help="comma-separated lat,lng coordinates"
)
@click.option("--date", default=None, help="date of recording as YYYY-MM-DD")
@click.option(
    "--top-n",
    type=int,
    default=5,
    help="Show top n predictions",
    show_default=True,
)
@click.option(
    "--scientific", is_flag=True, help="Display results using scientific names"
)
def run_classifier(
    modelpath: Path,
    audiofile: Path,
    latlng,
    date,
    top_n: int,
    scientific: bool,
):
    """Run classifier located at MODELPATH on AUDIOFILE"""
    if latlng is not None:
        latlng = [float(x) for x in latlng.split(",")]
    if date is not None:
        date = datetime.datetime.fromisoformat(date)
    preds = infer.run_classifier(
        modelpath, audiofile, latlng, date, scientific=scientific
    )
    for label in sorted(preds, key=preds.__getitem__, reverse=True)[:top_n]:
        print(f"{label: >30}: {preds[label]:.3f}")


if __name__ == "__main__":
    cli()
