import tarfile
import zipfile
from urllib import request
import sys
import time
from pathlib import Path


def _reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        print('  %   Total    Recv       Speed  Time left')
        return
    duration = time.time() - start_time
    progress_size = count * block_size
    try:
        speed = progress_size / duration
    except ZeroDivisionError:
        speed = float('inf')
    percent = progress_size / total_size * 100
    eta = int((total_size - progress_size) / speed)
    sys.stdout.write(
        '\r{:3.0f} {:4.0f}MiB {:4.0f}MiB {:6.0f}KiB/s {:4d}:{:02d}:{:02d}'
        .format(
            percent, total_size / (1 << 20), progress_size / (1 << 20),
            speed / (1 << 10), eta // 60 // 60, (eta // 60) % 60, eta % 60))
    sys.stdout.flush()


def urlretrieve(url, filename):
    if isinstance(filename, str):
        filename = Path(filename)
    if not filename.parent.exists():
        filename.parent.mkdir(parents=True)
    request.urlretrieve(url, filename, _reporthook)


def extractall(file_path, destination, ext):
    """Extracts an archive file.
    This function extracts an archive file to a destination.
    Args:
        file_path (string): The path of a file to be extracted.
        destination (string): A directory path. The archive file
            will be extracted under this directory.
        ext (string): An extension suffix of the archive file.
            This function supports :obj:`'.zip'`, :obj:`'.tar'`,
            :obj:`'.gz'` and :obj:`'.tgz'`.
    """

    if ext == '.zip':
        with zipfile.ZipFile(file_path, 'r') as z:
            z.extractall(destination)
    elif ext == '.tar':
        with tarfile.TarFile(file_path, 'r') as t:
            t.extractall(destination)
    elif ext == '.gz' or ext == '.tgz':
        with tarfile.open(file_path, 'r:gz') as t:
            t.extractall(destination)
