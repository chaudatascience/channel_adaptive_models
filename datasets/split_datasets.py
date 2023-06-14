import os.path
from collections import Counter

import numpy as np
import torchvision
from sklearn.model_selection import train_test_split

import utils
from utils import write_json

### SPLIT CIFAR dataset into 3 sub-datasets: red, red-green, green-blue
def split_datasets(path="data/split", random_seed=2022):
    def split_dataset(dataset, train_split: bool):
        train_set = dataset(root='./data', train=train_split, download=True)
        targets = np.array(train_set.targets)

        the_rest_vs_red_ratio = 2 / 3
        rg_vs_gb_ratio = 1 / 2

        red_idx, the_rest_idx = train_test_split(
            np.arange(len(targets)), test_size=the_rest_vs_red_ratio, random_state=random_seed, shuffle=True,
            stratify=targets)

        red_green_idx, green_blue_idx = train_test_split(
            the_rest_idx, test_size=rg_vs_gb_ratio, random_state=random_seed, shuffle=True,
            stratify=targets[the_rest_idx])

        ## Sanity check:
        # compare the number of each class in each sub-dataset, the difference should be less or equal  1
        r_vs_rg_diff = Counter(targets[red_idx]) - Counter(targets[red_green_idx])
        assert max(r_vs_rg_diff.values()) <= 1

        rg_vs_gb_diff = Counter(targets[red_green_idx]) - Counter(targets[green_blue_idx])
        assert max(rg_vs_gb_diff.values()) <= 1

        data = {}
        data["red_idx"] = red_idx
        data["red_green_idx"] = red_green_idx
        data["green_blue_idx"] = green_blue_idx
        return data

    cifar10_dataset = torchvision.datasets.CIFAR10
    cifar100_dataset = torchvision.datasets.CIFAR100

    cifar10 = split_dataset(cifar10_dataset, train_split=True)
    write_json(os.path.join(path, "cifar10_train.json"), cifar10, cls=utils.NumpyEncoder)
    cifar10 = split_dataset(cifar10_dataset, train_split=False)
    write_json(os.path.join(path, "cifar10_test.json"), cifar10, cls=utils.NumpyEncoder)

    cifar100 = split_dataset(cifar100_dataset, train_split=True)
    write_json(os.path.join(path, "data/cifar100_train.json"), cifar100, cls=utils.NumpyEncoder)
    cifar100 = split_dataset(cifar100_dataset, train_split=False)
    write_json(os.path.join(path, "data/cifar100_test.json"), cifar100, cls=utils.NumpyEncoder)


if __name__ == '__main__':
    # split_datasets()
    pass
