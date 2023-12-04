import os
from collections import defaultdict

import torch
from datasets import morphem70k
from torch.utils.data import DataLoader
import time


def compute_mean_std_morphem70k(chunk, root_dir):
    csv_path = os.path.join(root_dir, "morphem70k.csv")
    dataset = morphem70k.SingleCellDataset(
        csv_path, chunk=chunk, root_dir=root_dir, is_train=True, target_labels="label"
    )

    loader = DataLoader(dataset, batch_size=10, shuffle=False, drop_last=False)
    mean = 0.0
    for images, _ in loader:
        #     print(images.shape) : 10, 3, 238, 374
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        #     print(images.shape): 10, 3, 89012
        mean += images.mean(2).sum(0)
    mean = mean / len(loader.dataset)

    var = 0.0
    pixel_count = 0
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        var += ((images - mean.unsqueeze(1)) ** 2).sum([0, 2])
        pixel_count += images.nelement()/ images.size(1)
    std = torch.sqrt(var / pixel_count)

    return list(mean.numpy()), list(std.numpy())


if __name__ == "__main__":
    mean_std = {}
    root_dir = "/projectnb/morphem/data_70k/ver2/morphem_70k_version2/"
    out_path = os.path.join(
        "/projectnb/morphem/data_70k/ver2", "mean_std_morphem70k_ver2.txt"
    )

    for chunk in ["Allen", "CP", "HPA"]:
        start_time = time.time()

        mean, std = compute_mean_std_morphem70k(chunk, root_dir)
        mean_std[chunk] = [mean, std]

        msg = f"data={chunk}, mean={mean}, std={std}"
        with open(out_path, "a") as out:
            out.write(msg + "\n\n")
            out.write("--- %s seconds ---\n\n" % (time.time() - start_time))
