from torch.utils.data import DataLoader

from repro_vision import datasets
from repro_vision.transforms import transforms, Compose


def get_dataset(dataset_name, dataset_root="data", train=True, transforms=None,
                **kwargs):
    dataset_klass = getattr(datasets, dataset_name)
    dataset = dataset_klass(dataset_root, train=train, transforms=transforms,
                            **kwargs)
    return dataset


def get_transforms(config):
    transform_list = []
    for name, params in config.items():
        if params:
            transform_list.append(getattr(transforms, name)(**params))
        else:
            transform_list.append(getattr(transforms, name)())
    return Compose(transform_list)


def get_loaders(dataset_name, dataset_root="data", train_transforms=None,
                val_transforms=None, batchsize=32, num_workers=0, **kwargs):
    """Get Loader.
    Args:
        config (dict): Parameter dictionary.
    Returns:
        tuple: train and test ``DataLoader`` class.
    """
    train_dataset = get_dataset(dataset_name, dataset_root,
                                transforms=train_transforms, **kwargs)
    val_dataset = get_dataset(dataset_name, dataset_root, train=False,
                              transforms=val_transforms, **kwargs)

    train_loader = DataLoader(train_dataset, batchsize,
                              num_workers=num_workers,
                              shuffle=True)
    val_loader = DataLoader(val_dataset, batchsize,
                            num_workers=num_workers,
                            shuffle=False)
    return train_loader, val_loader


def get_labels(loader):
    return loader.dataset.label_names
