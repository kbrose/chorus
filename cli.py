import click

from chorus import pipelines
from chorus import train as c_train
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


@cli.command(help="train the model")
@click.argument("name", type=str)
def train(name: str):
    c_train.train(name)


if __name__ == "__main__":
    cli()
