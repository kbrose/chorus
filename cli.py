import click

from chorus import data as c_data
from chorus import train as c_train


@click.group()
def cli():
    pass


@cli.group(help='subcommands to download data')
def data():
    pass


@data.command('xc-meta', help='download xeno-canto meta data')
@click.option('-v', '--verbose', is_flag=True, help='Show progress bar.')
def xeno_meta(verbose: bool):
    c_data.save_all_xeno_canto_meta(verbose)


@data.command('xc-audio', help='download xeno-canto audio data')
@click.option('-v', '--verbose', is_flag=True, help='Show progress bar.')
@click.option('--redownload', is_flag=True, help="Redownload all files.")
def xeno_audio(verbose: bool, redownload: bool):
    c_data.save_all_xeno_canto_audio(verbose, skip_existing=not redownload)


@data.command('xc-to-npy', help='convert audio to .npy data at [SAMPLERATE]')
@click.argument('samplerate', type=int)
@click.option('-v', '--verbose', is_flag=True, help='Show progress bar.')
def xeno_to_numpy(samplerate, verbose: bool):
    c_data.convert_to_numpy(samplerate, verbose)


@cli.command(help='train the model')
def train():
    c_train.train()


if __name__ == "__main__":
    cli()