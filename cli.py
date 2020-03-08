import click

from chorus.data import save_all_xeno_canto_meta, save_all_xeno_canto_audio


@click.group()
def cli():
    pass


@cli.group(help='subcommands to download data')
def data():
    pass


@data.command('xc-meta', help='download xeno-canto meta data')
@click.option('-v', '--verbose', is_flag=True, help='Show progress bar.')
def xeno_meta(verbose: bool):
    save_all_xeno_canto_meta(verbose)


@data.command('xc-audio', help='download xeno-canto audio data')
@click.option('-v', '--verbose', is_flag=True, help='Show progress bar.')
@click.option('--redownload', is_flag=True, help="Redownload all files.")
def xeno_audio(verbose: bool, redownload: bool):
    save_all_xeno_canto_audio(verbose, skip_existing=not redownload)


# cli.add_command(data)
# data.add_command(xeno_meta)
# data.add_command(xeno_audio)


if __name__ == "__main__":
    cli()
