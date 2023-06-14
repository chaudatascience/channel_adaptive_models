import torchvision
from torch.utils.data import Dataset
from torchvision.transforms import transforms

import utils


class CifarRandomInstance(Dataset):
    chunk_names = ["red", "red_green", "green_blue"]

    def __init__(self, dataset: str, transform_train):

        torch_dataset = getattr(torchvision.datasets, dataset.upper())
        train_set = torch_dataset(root='../data', train=True, download=True)
        self.X = train_set.data
        self.y = train_set.targets
        self.transform_train = transforms.Compose([transforms.ToPILImage(), transform_train])

        train_idxs = utils.read_json(f"../data/split/{dataset}_train.json")
        self.idx_dict = {}
        for chunk_name in self.chunk_names:
            self.idx_dict[chunk_name] = train_idxs[f"{chunk_name}_idx"]

    def get_chunk_name(self, idx):
        for chunk_name in self.chunk_names:
            if idx in self.idx_dict[chunk_name]:
                return chunk_name
        raise ValueError(f"idx={idx} not found!")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        image = self.X[idx]
        label = self.y[idx]
        image = self.transform_train(image)
        return {"chunk": self.get_chunk_name(idx), "image": image, "label": label}

#
# def cifar_collate(data):
#     out = {chunk: {"image": [], "label": []} for chunk in CifarRandomInstance.chunk_names}
#
#     for d in data:
#         out[d["chunk"]]["image"].append(d["image"])
#         out[d["chunk"]]["label"].append(d["label"])
#     for chunk in out:
#         out[chunk]["image"] = torch.stack(out[chunk]["image"], dim=0)
#         out[chunk]["label"] = torch.tensor(out[chunk]["label"])
#     return out
