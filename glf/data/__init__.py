"""create dataset and dataloader"""
import logging

import torch
import torch.utils.data


def create_dataloader(dataset, dataset_opt, opt=None, sampler=None):
    phase = dataset_opt['phase']
    if phase == 'train':
        if opt['dist']:
            world_size = torch.distributed.get_world_size()
            num_workers = dataset_opt['n_workers']
            assert dataset_opt['batch_size'] % world_size == 0
            batch_size = dataset_opt['batch_size'] // world_size
            shuffle = False
        else:
            num_workers = dataset_opt['n_workers'] * len(opt['gpu_ids'])
            batch_size = dataset_opt['batch_size']
            shuffle = True
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                           num_workers=num_workers, sampler=sampler, drop_last=True,
                                           pin_memory=False)
    else:
        return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1,
                                           pin_memory=False)


def create_dataset(dataset_opt, is_train):
    """
    :param dataset_opt: dict, part of config corresponding to dataset
    :param is_train: bool, train or test flag
    :return:
    """
    name = dataset_opt['name']
    if name == 'MNIST':
        from torchvision.datasets import MNIST as D
    elif name == 'Fashion':
        from torchvision.datasets import FashionMNIST as D
    elif name == 'CIFAR-10':
        from torchvision.datasets import CIFAR10 as D
    else:
        raise NotImplementedError('Dataset [{:s}] is not recognized.'.format(name))
    dataset = D(root=dataset_opt['dataroot'], train=is_train, transform=None, target_transform=None, download=True)

    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           dataset_opt['name']))
    return dataset
