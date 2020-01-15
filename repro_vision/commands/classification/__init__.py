import click

from repro_vision.commands.classification import train


@click.group(invoke_without_command=True)
@click.pass_context
def main(ctx, **kwargs):
    pass


main.add_command(train.main, 'train')
