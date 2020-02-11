from pathlib import Path

from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToPILImage

from repro_vision import transforms
from repro_vision import datasets

ROOT_DIR = Path(__file__).parents[2]


def get_dataset(config, train=True, transforms=None):
    dataset_klass = getattr(datasets, config['dataset_name'])
    dataset = dataset_klass(config['dataset_root'], train=train,
                            transforms=transforms, download=True)
    return dataset


def get_transforms(config):
    transform_list = [ToPILImage()]
    for aug_name, params in config.items():
        transform_class = getattr(transforms, aug_name)
        transform_list.append(transform_class(**params))
    transform_list.append(transforms.ToTensor())
    return Compose(transform_list)


def get_loaders(config, train_transforms, val_transforms, num_workers=0):
    """Get Loader.
    Args:
        config (dict): Parameter dictionary.
    Returns:
        tuple: train and test ``DataLoader`` class.
    """
    config['dataset_root'] = ROOT_DIR / config['dataset_root']

    train_dataset = get_dataset(config, transforms=train_transforms)
    val_dataset = get_dataset(config, train=False, transforms=val_transforms)

    train_loader = DataLoader(train_dataset, config['batchsize'],
                              num_workers=num_workers,
                              shuffle=True)
    val_loader = DataLoader(val_dataset, config['batchsize'],
                            num_workers=num_workers,
                            shuffle=False)
    return train_loader, val_loader
