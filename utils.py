import copy
import functools
import json
import os
import random
from collections import Counter
from enum import Enum
from functools import wraps, partial
import time
from typing import Callable, Union, List, Tuple, Any, Dict
import pathlib
import dill
import h5py
import numpy as np
import torch
import torchvision
import yaml
from sklearn.model_selection import train_test_split
from torch import nn
import pprint
import torch.nn.functional as F
from torchvision import transforms as T


############ Device, GPU, Mem, Cuda ############


def get_machine_name():
    import socket

    machine_name = socket.gethostname()
    return machine_name


def running_on_server(verbose=True):
    MY_MACBOOK = "chaumac.local"

    machine_name = get_machine_name()
    if verbose:
        print(f"running on {machine_name}")
    return machine_name != MY_MACBOOK


def get_device(cuda_device=None, verbose=True):
    cuda = default(cuda_device, "cuda")
    device = torch.device(f"{cuda_device}" if torch.cuda.is_available() else "cpu")
    if verbose:
        print("device: ", device)
    return device


def gpu_mem_report_details():
    import humanize, psutil, GPUtil

    print("CPU RAM Free: " + humanize.naturalsize(psutil.virtual_memory().available))
    gpu_list = GPUtil.getGPUs()
    for i, gpu in enumerate(gpu_list):
        print(
            "GPU {:d} ... Mem Used: {:.0f}MB\t Free: {:.0f}MB / {:.0f}MB | Utilization {:3.0f}%".format(
                i,
                gpu.memoryTotal - gpu.memoryFree,
                gpu.memoryFree,
                gpu.memoryTotal,
                gpu.memoryUtil * 100,
            )
        )


def gpu_mem_report(device: Union[int, List, torch.device, None] = None, msg=None):
    def get_mem_msg(cuda):
        free, total = torch.cuda.mem_get_info(device=cuda)
        free_gb, total_gb = free / 1024**3, total / 1024**3
        used_gb = total_gb - free_gb
        msg_1 = f"Device {cuda} - {torch.cuda.get_device_name(cuda)}"
        msg_2 = (
            f"Mem used: {used_gb:.2f} GB; free/total: {free_gb:.2f}/{total_gb:.2f} GB\n"
        )
        return msg_1, msg_2

    def ensure_list(x: Union[int, List]):
        if isinstance(x, List):
            return x
        else:
            return [x]

    if not torch.cuda.is_available():  # skip if gpu is not available
        return None
    if device is None:
        device = range(torch.cuda.device_count())
    else:
        device = ensure_list(device)

    if msg is not None:
        print(msg)

    for cuda in device:
        msg_1, msg_2 = get_mem_msg(cuda)
        print(msg_1, "\n", msg_2, "------")


def move_to_cuda(sample, device):
    def _move_to_cuda(tensor):
        return tensor.to(device)

    return apply_to_sample(_move_to_cuda, sample)


def apply_to_sample(f, sample):
    if len(sample) == 0:
        return {}

    def _apply(x):
        if torch.is_tensor(x):
            return f(x)
        elif isinstance(x, dict):
            return {key: _apply(value) for key, value in x.items()}
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        else:
            return x

    return _apply(sample)


############## Model, gradient ##############


def save_pytorch_model(path, model, epoch, optimizer=None):
    model_dict = {"epoch": epoch, "model_state": model.state_dict()}
    if optimizer is not None:
        model_dict["optimizer_state"] = optimizer.state_dict()

    torch.save(model_dict, path)


def set_requires_grad(model: nn.Module, val: bool):
    for p in model.parameters():
        p.requires_grad = val


def analyze_model(model, print_trainable=False):
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(list(model.state_dict().keys()))

    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)

    for name, param in model.named_parameters():
        print(name, param.shape, param.numel(), param.requires_grad)

    if print_trainable:
        trainable_layers = [
            name for name, param in model.named_parameters() if param.requires_grad
        ]
        print("trainable_layers:")
        pp.pprint(trainable_layers)

    print(f"\nTotal parameters: {total_num:,};\tTrainable: {trainable_num:,}")
    return total_num, trainable_num


#################### Write, Read files ###############################


def write_hdf5(output_path, data, data_id, mode, dtype="uint8"):
    try:
        with h5py.File(output_path, mode) as hf:
            hf.create_dataset(data_id, data=data, dtype=dtype, compression="gzip")
    except ValueError:
        print("file exists, skipped.")


def read_hdf5(data_path, data_id=None):
    with h5py.File(data_path, "r") as hf:
        data = hf.get(data_id)
        if data is not None:
            return data[:]
        else:
            return None


def write_json(file_path, my_dict, cls=None):
    with open(file_path, "w") as fp:
        json.dump(my_dict, fp, cls=cls)
    return None


def read_json(filename):
    with open(filename, encoding="utf8") as fr:
        return json.load(fr)


def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def read_dill(path):
    with open(path, "rb") as f:
        generator = dill.load(f)
    print(f"Done reading {path}!")

    return generator


def write_dill(output_path, obj):
    ## Write to file
    if output_path is not None:
        folder_path = os.path.dirname(output_path)  # _output folder
        pathlib.Path(folder_path).mkdir(
            parents=True, exist_ok=True
        )  # create the folder(s) recursively if does not exist
        with open(output_path, "wb") as f:
            dill.dump(obj, f)
    print(f"Done writing {output_path}!")
    return True


def write_numpy(x: np.ndarray, output_path: str):
    np.save(output_path, x)
    print(f"Done writing {output_path}!")
    return True


######################## Timers ########################
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        :param val:
        :param n:
        :return:
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class TimeMeter(object):
    """Computes the average occurrence of some event per second"""

    def __init__(self, init=0):
        self.reset(init)

    def reset(self, init=0):
        self.init = init
        self.start = time.time()
        self.n = 0

    def update(self, val=1):
        self.n += val

    @property
    def avg(self):
        return self.n / self.elapsed_time

    @property
    def elapsed_time(self):
        return self.init + (time.time() - self.start)


class Time1Event(object):
    """Computes the time to finish one event in seconds"""

    def __init__(self, init=0):
        self.reset(init)

    def reset(self, init=0):
        self.init = init
        self.start = time.time()
        self.n = 0

    def update(self, val=1):
        self.n += val

    @property
    def avg(self):
        return self.elapsed_secs / self.n

    @property
    def elapsed_secs(self):
        return self.init + (time.time() - self.start)


class StopwatchMeter(object):
    """Computes the sum/avg duration of some event in seconds"""

    def __init__(self):
        self.reset()

    def start(self):
        self.start_time = time.time()

    def stop(self, n=1):
        if self.start_time is not None:
            delta = time.time() - self.start_time
            self.sum += delta
            self.n += n
            self.start_time = None

    def reset(self):
        self.sum = 0
        self.n = 0
        self.start_time = None

    @property
    def avg(self):
        return self.sum / self.n


def timer_func(func):
    @functools.wraps(func)
    def wrap_func(*args, **kwargs):
        t_start = time.time()
        result = func(*args, **kwargs)
        t_stop = time.time()
        minutes = (t_stop - t_start) / 60
        print(f'Function "{func.__name__}" executed in {minutes:.4f} minutes')
        return result

    return wrap_func


def convert_secs2time(
    epoch_time, return_string=True
) -> Union[str, Tuple[int, int, int]]:
    """return hour_min_second in string message format, or (hour, min, second)"""
    now = datetime_now(time_format="%Y-%b-%d %H:%M:%S")

    need_hour = int(epoch_time / 3600)
    need_mins = int((epoch_time - 3600 * need_hour) / 60)
    need_secs = int(epoch_time - 3600 * need_hour - 60 * need_mins)
    need_time = "[{}]  Need {:02d}:{:02d}:{:02d}".format(
        now, need_hour, need_mins, need_secs
    )

    if return_string:
        return need_time
    else:
        return need_hour, need_mins, need_secs


################ Others #############


def exists(val):
    return val is not None


def default(val, default):
    return val if exists(val) else default


def repeat(_func=None, *, num_times=2):
    def decorator_repeat(func):
        @functools.wraps(func)
        def wrapper_repeat(*args, **kwargs):
            for _ in range(num_times):
                value = func(*args, **kwargs)
            return value

        return wrapper_repeat

    if _func is None:
        return decorator_repeat
    else:
        return decorator_repeat(_func)


def acc_score(pred, y):
    return (torch.sum(pred == y) / len(y)).item()


def set_seeds(seed=2022):
    random.seed(seed)
    np.random.seed(seed + 1)
    torch.manual_seed(seed + 2)
    torch.cuda.manual_seed(seed + 4)
    torch.cuda.manual_seed_all(seed + 4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mkdir(path, mode=0o700):
    pathlib.Path(path).mkdir(mode=mode, parents=True, exist_ok=True)


def get_url_img(url="http://images.cocodataset.org/val2017/000000039769.jpg"):
    from PIL import Image
    import requests

    image = Image.open(requests.get(url, stream=True).raw)
    return image


def get_gpu_mem(cuda="cuda:0"):
    free, total = torch.cuda.mem_get_info(device=cuda)
    free_gb, total_gb = free / 1024**3, total / 1024**3
    return total_gb


def datetime_now(time_format: str = None) -> str:
    from datetime import datetime

    # time_format = default(time_format, "%Y-%b-%d %H:%M:%S.%f")
    time_format = default(time_format, "%Y-%b-%d %H:%M:%S")
    return datetime.now().strftime(time_format)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def worker_init_fn(worker_id):
    seed = worker_id + 2022
    set_seeds(seed)


def ensure_list(x: Any) -> List:
    if isinstance(x, list):
        return x
    elif isinstance(x, tuple):
        return list(x)
    return [x]


class ExtendedEnum(Enum):
    @classmethod
    def list(cls):
        """
        get all values from Enum
        """
        return list(map(lambda c: c.value, cls))


def dict2obj(my_dict: Dict):
    class Obj:
        def __init__(self, **entries):
            self.__dict__.update(entries)

    return Obj(**my_dict)


def pairwise_distance_v2(proxies, x, squared=False):
    if squared:
        return (torch.cdist(x, proxies, p=2)) ** 2
    else:
        return torch.cdist(x, proxies, p=2)


def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        try:
            p.grad.data = p.grad.data.float()
        except:
            pass
