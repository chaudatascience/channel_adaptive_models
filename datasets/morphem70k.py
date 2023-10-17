import os
from typing import Callable, Dict

import numpy as np
import pandas as pd
import torch
import torchvision
from torch import Tensor
from torch.utils.data import Dataset
import skimage.io


class SingleCellDataset(Dataset):
    """Single cell chunk."""

    ## define all available datasets, used for collate_fn later on
    chunk_names = ["Allen", "CP", "HPA"]

    def __init__(
        self,
        csv_path: str,
        chunk: str,
        root_dir: str,
        is_train: bool,
        ssl_flag: bool,
        target_labels: str = "label",
        transform: Callable | Dict[str, Callable] | None = None,
    ):
        """
        @param csv_path: Path to the csv file with metadata.
             e.g., "metadata/morphem70k_v2.csv".
        You should copy this file to the dataset folder to avoid modifying other config.

        Note: Allen was renamed to WTC-11 in the paper.
        @param chunk: "Allen", "HPA", "CP", or "morphem70k"to use all 3 chunks
        @param root_dir: root_dir: Directory with all the images.
        @param is_train: True for training set, False for using all data
        @param target_labels: label column in the csv file
        @param transform: data transform to be applied on a sample.
        """
        assert chunk in [
            "Allen",
            "HPA",
            "CP",
            "morphem70k",
        ], "chunk must be one of 'Allen', 'HPA', 'CP', 'morphem70k'"
        self.chunk = chunk
        self.is_train = is_train

        ## read csv file for the chunk
        self.metadata = pd.read_csv(csv_path)
        if chunk in ["Allen", "HPA", "CP"]:
            self.metadata = self.metadata[self.metadata["chunk"] == chunk]

        if is_train:
            self.metadata = self.metadata[self.metadata["train_test_split"] == "Train"]

        self.metadata = self.metadata.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform
        self.target_labels = target_labels
        self.ssl_flag = ssl_flag

        ## classes on training set:

        if chunk == "Allen":
            self.train_classes_dict = {
                "M0": 0,
                "M1M2": 1,
                "M3": 2,
                "M4M5": 3,
                "M6M7_complete": 4,
                "M6M7_single": 5,
            }

        elif chunk == "HPA":
            self.train_classes_dict = {
                "golgi apparatus": 0,
                "microtubules": 1,
                "mitochondria": 2,
                "nuclear speckles": 3,
            }

        elif chunk == "CP":
            self.train_classes_dict = {
                "BRD-A29260609": 0,
                "BRD-K04185004": 1,
                "BRD-K21680192": 2,
                "DMSO": 3,
            }

        else:  # all 3 chunks (i.e., "morphem70k")
            self.train_classes_dict = {
                "BRD-A29260609": 0,
                "BRD-K04185004": 1,
                "BRD-K21680192": 2,
                "DMSO": 3,
                "M0": 4,
                "M1M2": 5,
                "M3": 6,
                "M4M5": 7,
                "M6M7_complete": 8,
                "M6M7_single": 9,
                "golgi apparatus": 10,
                "microtubules": 11,
                "mitochondria": 12,
                "nuclear speckles": 13,
            }

        self.test_classes_dict = None  ## Not defined yet

    def __len__(self):
        return len(self.metadata)

    @staticmethod
    def _fold_channels(image: np.ndarray, channel_width: int, mode="ignore") -> Tensor:
        """
        Re-arrange channels from tape format to stack tensor
        @param image: (h, w * c)
        @param channel_width:
        @param mode:
        @return: Tensor, shape of  (c, h, w)  in the range [0.0, 1.0]
        """
        # convert to shape of (h, w, c),  (in the range [0, 255])
        output = np.reshape(image, (image.shape[0], channel_width, -1), order="F")

        if mode == "ignore":
            # Keep all channels
            pass
        elif mode == "drop":
            # Drop mask channel (last)
            output = output[:, :, 0:-1]
        elif mode == "apply":
            # Use last channel as a binary mask
            mask = output["image"][:, :, -1:]
            output = output[:, :, 0:-1] * mask
        output = torchvision.transforms.ToTensor()(output)
        return output

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.root_dir, self.metadata.loc[idx, "file_path"])
        channel_width = self.metadata.loc[idx, "channel_width"]
        image = skimage.io.imread(img_path)
        image = self._fold_channels(image, channel_width)

        if self.is_train:
            label = self.metadata.loc[idx, self.target_labels]
            label = self.train_classes_dict[label]
            label = torch.tensor(label)
        else:
            label = None  ## for now, we don't need labels for evaluation. It will be provided later in evaluation code.

        if self.chunk == "morphem70k":  ## using all 3 datasets
            chunk = self.metadata.loc[idx, "chunk"]
            if self.transform:
                image = self.transform[chunk](image)
            if self.is_train:
                data = {"chunk": chunk, "image": image, "label": label}
            else:
                data = {"chunk": chunk, "image": image}
        else:  ## using single chunk
            if self.transform:
                image = self.transform(image)
            if self.is_train:
                data = image, label
            else:
                data = image
        return data
