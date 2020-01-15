import click

from repro_vision.commands import classification


__version__ = '0.1.0'


@click.group(invoke_without_command=True)
@click.version_option(__version__)
@click.pass_context
def main(ctx, **kwargs):
    if ctx.obj is None:
        ctx.obj = {}


main.add_command(classification.main, 'classification')
