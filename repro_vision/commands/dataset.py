from torch.utils.data import DataLoader

from repro_vision import datasets, transforms


def get_dataset(dataset_name, dataset_root="data", image_set="train",
                transform_funs=None, **kwargs):
    dataset_klass = getattr(datasets, dataset_name)
    dataset = dataset_klass(dataset_root, image_set=image_set,
                            transforms=transform_funs, **kwargs)
    return dataset


def get_transforms(config):
    if config is None:
        return None
    transform_list = []
    for name, params in config.items():
        if params:
            transform_list.append(getattr(transforms, name)(**params))
        else:
            transform_list.append(getattr(transforms, name)())
    return transforms.Compose(transform_list)


def get_loaders(dataset_name, dataset_root="data", train_transforms=None,
                val_transforms=None, batchsize=32, num_workers=0, **kwargs):
    """Get Loader.
    Args:
        config (dict): Parameter dictionary.
    Returns:
        tuple: train and test ``DataLoader`` class.
    """
    train_dataset = get_dataset(dataset_name, dataset_root,
                                transform_funs=train_transforms, **kwargs)
    val_dataset = get_dataset(dataset_name, dataset_root, image_set='val',
                              transform_funs=val_transforms, **kwargs)

    train_loader = DataLoader(train_dataset, batchsize,
                              num_workers=num_workers,
                              collate_fn=train_dataset.collate_fn,
                              shuffle=True)
    val_loader = DataLoader(val_dataset, batchsize,
                            num_workers=num_workers,
                            collate_fn=val_dataset.collate_fn,
                            shuffle=False)
    return train_loader, val_loader


def get_labels(loader):
    return loader.dataset.label_names
