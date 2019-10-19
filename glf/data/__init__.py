"""create dataset and dataloader"""
import logging
import os

import numpy as np
import torch
import torch.utils.data
from torch.utils.data import TensorDataset
from torchvision import transforms

from glf.data.dots import gen_image_count


def _create_dots(root, is_train, num_objects=3, num_samples=1000):
    filename = 'train' if is_train else 'test'
    path_fo_file = os.path.join(root, filename)

    if not os.path.exists(path_fo_file):
        images = []
        for i in range(num_samples):
            new_img = gen_image_count(num_object=num_objects).astype(np.float32) / 255.0
            images.append(new_img)

        images = np.stack(images, axis=0).transpose((0, 3, 1, 2))
        np.save(path_fo_file, images)
    else:
        images = np.load(path_fo_file)

    dataset = TensorDataset(torch.from_numpy(images), torch.from_numpy(np.array([num_objects] * num_samples)))
    return dataset


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
            num_workers = 1 if opt['gpu_ids'] is None else dataset_opt['n_workers'] * len(opt['gpu_ids'])
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
    elif name == 'CelebA':
        from torchvision.datasets import CelebA as D
    elif name == 'dots':
        pass
    else:
        raise NotImplementedError('Dataset [{:s}] is not recognized.'.format(name))

    if name == 'CelebA':
        # TODO add center crop to 160 × 160 and then resize to 64×64
        dataset = D(root=dataset_opt['dataroot'], split='train' if is_train else 'valid', target_type=None,
                    transform=transforms.ToTensor(), target_transform=None, download=True)
    elif name == 'dots':
        dataset = _create_dots(root=dataset_opt['dataroot'], is_train=is_train)
    else:
        dataset = D(root=dataset_opt['dataroot'], train=is_train, transform=transforms.ToTensor(),
                    target_transform=None, download=True)

    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           dataset_opt['name']))
    return dataset
