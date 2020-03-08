import click

from chorus.data import save_all_xeno_canto_meta


@click.group()
def cli():
    pass


@click.group()
def data():
    pass


@click.command('xeno-meta')
@click.option('-v', '--verbose', is_flag=True, help='Show progress bar.')
def xeno_meta(verbose: bool):
    save_all_xeno_canto_meta(verbose)


cli.add_command(data)
data.add_command(xeno_meta)


if __name__ == "__main__":
    cli()
