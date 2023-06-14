from __future__ import annotations

import os
from typing import Tuple, List, Dict
import torch
from torch import Tensor
import torchvision
from torch.utils.data import SubsetRandomSampler, DataLoader, Dataset
from torchvision.transforms import transforms

import utils
from datasets.morphem70k import SingleCellDataset
from datasets.cifar import CifarRandomInstance


def get_mean_std_dataset(dataset):
    """Calculate mean and std of cifar10, cifar100"""
    mean_cifar10, std_cifar10 = [0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784]
    mean_cifar100, std_cifar100 = [0.50707516, 0.48654887, 0.44091784], [0.26733429, 0.25643846, 0.27615047]

    ## mean, std on training sets
    mean_allen, std_allen = [0.17299604, 0.21203263, 0.06717164], [0.1803914, 0.1947802, 0.08771174]
    mean_hpa, std_hpa = ([0.08290475, 0.04112732, 0.064044476, 0.08445487],
                         [0.08106554, 0.052796874, 0.0885671, 0.0815554])
    mean_cp, std_cp = ([0.09957533, 0.19229738, 0.16250879, 0.18240248, 0.14978176],
                       [0.077283904, 0.074369825, 0.06784963, 0.066472545, 0.068180084])

    if dataset == "cifar10":
        return mean_cifar10, std_cifar10
    elif dataset == "cifar100":
        return mean_cifar100, std_cifar100
    elif dataset == "Allen":
        return mean_allen, std_allen
    elif dataset == "CP":
        return mean_cp, std_cp
    elif dataset == "HPA":
        return mean_hpa, std_hpa
    elif dataset == "morphem70k":
        return {"CP": (mean_cp, std_cp),
                "Allen": (mean_allen, std_allen),
                "HPA": (mean_hpa, std_hpa)}
    else:
        raise ValueError()


def get_data_transform(dataset: str, img_size: int):
    if dataset != "morphem70k":
        mean_data, std_data = get_mean_std_dataset(dataset)

    if dataset in ["cifar10", "cifar100"]:
        transform_train = transforms.Compose([
            transforms.RandomCrop(img_size, padding=4),
            transforms.Resize(img_size),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean_data, std_data),
        ])

        transform_eval = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean_data, std_data),
        ])
    elif dataset in ["Allen", "CP", "HPA"]:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1), antialias=True),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(15),
            # transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            ## transforms.ToTensor(), input is already a Tensor
            transforms.Normalize(mean_data, std_data),
        ])

        transform_eval = transforms.Compose([
            transforms.Resize(img_size, antialias=True),
            transforms.CenterCrop(img_size),
            transforms.Normalize(mean_data, std_data),
        ])
    elif dataset == "morphem70k":
        mean_stds = get_mean_std_dataset(dataset)
        transform_train = {}
        transform_eval = {}
        for data in ["CP", "Allen", "HPA"]:
            mean_data, std_data = mean_stds[data]
            transform_train_ = transforms.Compose([
                transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1), antialias=True),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize(mean_data, std_data),
            ])

            transform_eval_ = transforms.Compose([
                transforms.Resize(img_size, antialias=True),
                transforms.CenterCrop(img_size),
                transforms.Normalize(mean_data, std_data),
            ])
            transform_train[data] = transform_train_
            transform_eval[data] = transform_eval_

    else:
        raise ValueError()

    return transform_train, transform_eval


def get_in_dim(chunks: List[Dict]) -> List[int]:
    print("chunks", chunks)
    channels = [len(list(c.values())[0]) for c in chunks]
    print("channels", channels)
    return channels


def get_channel(dataset: str, data_channels: List[str], x: Tensor, device) -> Tensor:
    if dataset in ["cifar10", "cifar100"]:
        return _get_channel_cifar(data_channels, x, device)
    elif dataset in ["Allen", "CP", "HPA", "morphem70k"]:
        return x
    else:
        raise NotImplementedError()


def _get_channel_cifar(data_channels: List[str], x: Tensor, device) -> Tensor:
    """ x: batch of images, shape b, c, h, w, order of colors are RGB"""
    mapper = {"red": 0,
              "green": 1,
              "blue": 2}
    NUM_CHANNELS = 3
    ALL_CHANNELS = sorted(["red", "green", "blue"])

    assert len(data_channels) <= NUM_CHANNELS
    if sorted(data_channels) == ALL_CHANNELS:
        return x

    out = []

    # example for `data_channels`: data_channels = ["red", "red_green", "ZERO"]

    b, c, h, w = x.shape
    for channel in data_channels:
        if channel in mapper:  ## either red, green, or blue
            c_idx = mapper[channel]
            out.append(x[:, c_idx:c_idx + 1, ...])
        else:  # avg of some channels, or fill by zero
            splits = channel.split("_")
            if len(splits) > 1:
                reduce, channel_list = channel.split("_")[0].lower(), channel.split("_")[1:]
            else:
                reduce, channel_list = channel.split("_")[0].lower(), []

            if reduce == "avg":
                c_idx_list = [mapper[c] for c in channel_list]
                out.append(x[:, c_idx_list, ...].mean(dim=1, keepdim=True))
            elif reduce == "zero":
                out.append(torch.zeros(b, 1, h, w, device=device))
            else:
                raise ValueError()

    res = torch.concat(out, dim=1)
    return res


def get_samplers(dataset: str, img_size: int, chunk_name: str, train_split: bool) -> Tuple:
    def get_sampler_helper() -> SubsetRandomSampler:
        """
        For CIFAR, we split the dataset into 3 smaller dataset: only red, red_green, green_blue with equal size
        return indices for the sub-datasets
        :return:
        """
        ## we split dataset into 3 smaller ones by using datasets.split_datasets
        ## Read the indices for each dataset back
        if dataset in ["cifar10", "cifar100"]:
            split = "train" if train_split else "test"
            indices = utils.read_json(f"data/split/{dataset}_{split}.json")
            data_channel_idx = f"{chunk_name}_idx"
            sampler = SubsetRandomSampler(indices[data_channel_idx])
            return sampler
        else:
            raise ValueError()

    transform_train, transform_eval = get_data_transform(dataset, img_size)

    if dataset in ["cifar10", "cifar100"]:
        torch_dataset = getattr(torchvision.datasets, dataset.upper())
        data_set = torch_dataset(root='./data', train=train_split, download=True, transform=transform_train)
        data_sampler = get_sampler_helper()
        return data_set, data_sampler
    else:
        raise ValueError()


def get_train_val_test_loaders(dataset: str, img_size: int, chunk_name: str, seed: int, batch_size: int,
                               eval_batch_size: int, num_workers: int, root_dir: str, file_name: str) -> Tuple[
    DataLoader, DataLoader | None, DataLoader]:
    train_loader, val_loader, test_loader = None, None, None

    if dataset in ["cifar10", "cifar100"]:
        train_set, train_sampler = get_samplers(dataset, img_size=img_size, chunk_name=chunk_name, train_split=True)
        eval_set, eval_sampler = get_samplers(dataset, img_size=img_size, chunk_name=chunk_name, train_split=False)

        utils.set_seeds(seed + 24122022)
        train_loader = DataLoader(train_set, batch_size=batch_size, sampler=train_sampler,
                                  num_workers=num_workers,
                                  worker_init_fn=utils.worker_init_fn, pin_memory=True, drop_last=True)

        utils.set_seeds(seed + 25122022)
        test_loader = DataLoader(eval_set, batch_size=eval_batch_size, sampler=eval_sampler,
                                 num_workers=num_workers,
                                 worker_init_fn=utils.worker_init_fn, pin_memory=True)
    elif dataset in ["Allen", "CP", "HPA", "morphem70k"]:
        ## TODO: add dataset path in config file

        csv_path = os.path.join(root_dir, file_name)

        transform_train, transform_eval = get_data_transform(chunk_name, img_size)
        train_set = SingleCellDataset(csv_path, chunk_name, root_dir, is_train=True,
                                      transform=transform_train)

        test_set = SingleCellDataset(csv_path, chunk_name, root_dir, is_train=False,
                                     transform=transform_eval)

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers,
                                  worker_init_fn=utils.worker_init_fn, pin_memory=True, drop_last=True)

        ## IMPORTANT: set shuffle to False for test set. Otherwise, the order of the test set will be different when evaluating
        test_loader = DataLoader(test_set, batch_size=eval_batch_size, shuffle=False,
                                 num_workers=num_workers,
                                 worker_init_fn=utils.worker_init_fn, pin_memory=True)

    return train_loader, val_loader, test_loader


def get_classes(dataset: str, file_name: str) -> Tuple:
    if dataset in ["cifar10", "cifar100"]:
        torch_dataset = getattr(torchvision.datasets, dataset.upper())
        train_set = torch_dataset(root='./data', train=True, download=True, transform=None)
        train_classes = test_classes = train_set.classes

    else:
        if dataset == "Allen":
            if "morphem70k_v2" in file_name:
                train_classes = ['M0', 'M1M2', 'M3', 'M4M5',
                                 'M6M7_complete', 'M6M7_single']
            else:
                train_classes = ['Interphase', 'Mitotic']
        elif dataset == "CP":
            train_classes = ['BRD-A29260609', 'BRD-K04185004', 'BRD-K21680192', 'DMSO']
        elif dataset == "HPA":
            train_classes = ['golgi apparatus', 'microtubules', 'mitochondria', 'nuclear speckles']
        elif dataset == "morphem70k":
            if "morphem70k_v2" in file_name:
                train_classes = ['BRD-A29260609',
                                 'BRD-K04185004',
                                 'BRD-K21680192',
                                 'DMSO',
                                 'M0',
                                 'M1M2',
                                 'M3',
                                 'M4M5',
                                 'M6M7_complete',
                                 'M6M7_single',
                                 'golgi apparatus',
                                 'microtubules',
                                 'mitochondria',
                                 'nuclear speckles']
            else:
                train_classes = ['BRD-A29260609', 'BRD-K04185004', 'BRD-K21680192', 'DMSO',
                                 'Interphase', 'Mitotic', 'golgi apparatus', 'microtubules',
                                 'mitochondria', 'nuclear speckles']
        else:
            raise NotImplementedError(f"dataset {dataset} not valid!")

        test_classes = None

    return train_classes, test_classes


def make_cifar_random_instance_train_loader(dataset: str, img_size: int, batch_size: int, seed: int,
                                            num_workers: int) -> DataLoader:
    transform_train, _ = get_data_transform(dataset, img_size)

    train_set = CifarRandomInstance(dataset, transform_train)

    utils.set_seeds(seed + 2052023)
    cifar_collate = get_collate(CifarRandomInstance)
    train_loader = DataLoader(train_set, batch_size=batch_size, collate_fn=cifar_collate, num_workers=num_workers,
                              shuffle=True, drop_last=True)
    return train_loader


def make_morphem70k_random_instance_train_loader(img_size: int, batch_size: int, seed: int,
                                                 num_workers: int, root_dir: str, file_name: str) -> DataLoader:
    csv_path = os.path.join(root_dir, file_name)
    dataset = "morphem70k"
    transform_train, _ = get_data_transform(dataset, img_size)
    train_set = SingleCellDataset(csv_path, chunk=dataset, root_dir=root_dir, is_train=True, transform=transform_train)

    utils.set_seeds(seed + 20230322)
    morphem_collate = get_collate(SingleCellDataset)
    train_loader = DataLoader(train_set, batch_size=batch_size, collate_fn=morphem_collate, shuffle=True,
                              num_workers=num_workers, worker_init_fn=utils.worker_init_fn,
                              pin_memory=True, drop_last=True)
    return train_loader


def make_random_instance_train_loader(dataset: str, img_size: int, batch_size: int, seed: int,
                                      num_workers: int, root_dir: str, file_name: str) -> DataLoader:
    if dataset in ["cifar10", "cifar100"]:
        return make_cifar_random_instance_train_loader(dataset, img_size, batch_size, seed, num_workers)
    elif dataset in ["morphem70k"]:
        return make_morphem70k_random_instance_train_loader(img_size, batch_size, seed, num_workers, root_dir,
                                                            file_name)
    else:
        return None


def get_collate(class_name: Dataset):
    """
    class_name: one of  [CifarRandomInstance, SingleCellDataset]
    """

    def collate(data):
        out = {chunk: {"image": [], "label": []} for chunk in class_name.chunk_names}

        for d in data:
            out[d["chunk"]]["image"].append(d["image"])
            out[d["chunk"]]["label"].append(d["label"])
        for chunk in out:
            out[chunk]["image"] = torch.stack(out[chunk]["image"], dim=0)
            out[chunk]["label"] = torch.tensor(out[chunk]["label"])
        return out

    return collate
