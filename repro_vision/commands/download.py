from urllib.parse import urlparse
from pathlib import Path

import click

from repro_vision import download


AVAILABLE_DATASETS = {
    'voc': {
        'urls': [
            'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar',  # noqa
            'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar',  # noqa
            'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar'  # noqa
        ],
        'dest_dir': ['VOCdevkit/voc2012', 'VOCdevkit/voc2007']
    }
}


@click.command()
@click.argument('name', type=str)
@click.option('--data_root_dir', type=click.Path(), default="data")
@click.pass_context
def main(ctx, name, data_root_dir):
    data = AVAILABLE_DATASETS[name]
    urls = data['urls']
    for url in urls:
        filename = Path(data_root_dir) / Path(urlparse(url).path).name
        if not filename.exists():
            download.urlretrieve(url, filename)
        download.extractall(filename, data_root_dir, filename.suffix)


if __name__ == "__main__":
    main()
