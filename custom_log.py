from typing import Union, Dict

import torch
import time
import logging
from datetime import datetime
import os
import pathlib

from omegaconf import OmegaConf

import wandb
from dotenv import load_dotenv

from config import MyConfig, DataChunk
from utils import exists, default, get_machine_name


def get_py_logger(dataset_name: str, job_id: str = None):
    def _get_logger(logger_name, log_path, level=logging.INFO):
        logger = logging.getLogger(logger_name)  # global variance?
        formatter = logging.Formatter("%(asctime)s : %(message)s")

        fileHandler = logging.FileHandler(log_path, mode="w")
        fileHandler.setFormatter(formatter)  # `formatter` must be a logging.Formatter

        streamHandler = logging.StreamHandler()
        streamHandler.setFormatter(formatter)

        logger.setLevel(level)
        logger.addHandler(fileHandler)
        logger.addHandler(streamHandler)
        return logger

    logger_path = "logs/{dataset_name}"
    logger_name_format = "%Y-%m-%d--%H-%M-%S.%f"

    logger_name = f"{job_id} - {datetime.now().strftime(logger_name_format)}.log"

    logger_folder = logger_path.format(dataset_name=dataset_name)
    pathlib.Path(logger_folder).mkdir(parents=True, exist_ok=True)

    logger_path = os.path.join(logger_folder, logger_name)
    logger = _get_logger(logger_name, logger_path)
    return logger  # , logger_name, logger_path


def init_wandb(args: MyConfig, job_id, project_name, log_freq: int, model=None):
    # wandb.run.dir
    # https://docs.wandb.ai/guides/track/advanced/save-restore

    try:
        load_dotenv()
        os.environ["WANDB__SERVICE_WAIT"] = "300"
        wandb.login(key=os.getenv("WANDB_API_KEY"))
    except Exception as e:
        print(f"--- was trying to log in Weights and Biases... e={e}")

    ## run_name for wandb's run
    machine_name = get_machine_name()
       

    cuda = args.hardware.device.replace(":", "")  ## get cuda info instead
    if cuda == "cuda":
        try:
            cuda = torch.cuda.get_device_name(0)
        except Exception as e:
            print("error when trying to get_device_name", e)
            pass

    watermark = "{}_{}_{}_{}_{}".format(
        args.dataset.name, machine_name, cuda, job_id, time.strftime("%I-%M%p-%B-%d-%Y")
    )

    wandb.init(
        project=project_name,
           entity="adaptive_interface",
        name=watermark,
        settings=wandb.Settings(start_method="fork"),
    )

    # if exists(model):
    ## TODO: fix later
    #     wandb.watch(model, log_freq=log_freq, log_graph=True, log="all")  # log="all" to log gradients and parameters
    return watermark


class MyLogging:
    def __init__(self, args: MyConfig, model, job_id, project_name):
        self.args = args
        log_freq = self.args.logging.wandb.log_freq
        dataset = args.dataset.name
        self.use_py_logger = args.logging.use_py_log
        self.use_wandb = args.logging.wandb.use_wandb

        self.py_logger = get_py_logger(dataset, job_id) if self.use_py_logger else None
        if self.use_wandb:
            init_wandb(
                args,
                project_name=project_name,
                model=model,
                job_id=job_id,
                log_freq=log_freq,
            )

    def info(
        self,
        msg: Union[Dict, str],
        use_wandb=None,
        sep=", ",
        padding_space=False,
        pref_msg: str = "",
    ):
        use_wandb = default(use_wandb, self.use_wandb)

        if isinstance(msg, Dict):
            msg_str = (
                pref_msg
                + " "
                + sep.join(
                    f"{k} {round(v, 4) if isinstance(v, int) else v}"
                    for k, v in msg.items()
                )
            )
            if padding_space:
                msg_str = sep + msg_str + " " + sep

            if use_wandb:
                wandb.log(msg)

            if self.use_py_logger:
                self.py_logger.info(msg_str)
            else:
                print(msg_str)
        else:
            if self.use_py_logger:
                self.py_logger.info(msg)
            else:
                print(msg)

    def log_imgs(self, x, y, y_hat, classes, max_scores, name: str):
        columns = ["image", "pred", "label", "score", "correct"]
        data = []
        for j, image in enumerate(x, 0):
            # pil_image = Image.fromarray(image, mode="RGB")
            data.append(
                [
                    wandb.Image(image[:3]),
                    classes[y_hat[j].item()],
                    classes[y[j].item()],
                    max_scores[j].item(),
                    y_hat[j].item() == y[j].item(),
                ]
            )

        table = wandb.Table(data=data, columns=columns)
        wandb.log({name: table})

    def log_config(self, config: MyConfig):
        wandb.config.update(OmegaConf.to_container(config))  # , allow_val_change=True)

    def update_best_result(self, msg: str, metric, val, use_wandb=None):
        use_wandb = default(use_wandb, self.use_wandb)

        if self.use_py_logger:
            self.py_logger.info(msg)
        else:
            print(msg)
        if use_wandb:
            wandb.run.summary[metric] = val

    def finish(
        self,
        use_wandb=None,
        msg_str: str = None,
        model=None,
        model_best_name: str = "",
        dummy_batch_x=None,
    ):
        use_wandb = default(use_wandb, self.use_wandb)

        if exists(msg_str):
            if self.use_py_logger:
                self.py_logger.info(msg_str)
            else:
                print(msg_str)
        if use_wandb:
            if model_best_name:
                wandb.save(model_best_name)
                print(f"saved pytorch model {model_best_name}!")

            if exists(model):
                try:
                    # https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Simple_PyTorch_Integration.ipynb#scrollTo=j64Lu7pZcubd
                    if self.args.hardware.multi_gpus == "DataParallel":
                        model = model.module
                    torch.onnx.export(model, dummy_batch_x, "model.onnx")
                    wandb.save("model.onnx")
                    print("saved to model.onnx!")
                except Exception as e:
                    print(e)
            wandb.finish()
