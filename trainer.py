from __future__ import annotations

import collections
import os
import time
from copy import deepcopy
from os.path import join as os_join
from typing import Dict, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf, ListConfig
from torch import nn, Tensor

import models
import utils
from morphem.benchmark import run_benchmark
from helper_classes.best_result import BestResult
from config import MyConfig, DataChunk
from datasets.dataset_utils import (
    get_channel,
    get_in_dim,
    get_train_val_test_loaders,
    get_classes,
    make_random_instance_train_loader,
)
from helper_classes.channel_pooling_type import ChannelPoolingType
from helper_classes.datasplit import DataSplit
from models import model_utils
from models.depthwise_convnext import DepthwiseConvNeXt
from models.hypernet_convnext import HyperConvNeXt
from models.loss_fn import proxy_loss
from models.shared_convnext import SharedConvNeXt
from lr_schedulers import create_my_scheduler
from models.slice_param_convnext import SliceParamConvNeXt
from models.depthwise_convnext_miro import DepthwiseConvNeXtMIRO
from models.template_mixing_convnext import TemplateMixingConvNeXt
from models.template_convnextv2 import TemplateMixingConvNeXtV2
from models.convnext_shared_miro import ConvNeXtSharedMIRO
from models.hypernet_convnext_miro import HyperConvNeXtMIRO
from models.slice_param_convnext_miro import SliceParamConvNeXtMIRO
from models.template_convnextv2_miro import TemplateMixingConvNeXtV2MIRO

from optimizers import make_my_optimizer
from utils import AverageMeter, exists
from custom_log import MyLogging
from models.model_utils import get_shapes, MeanEncoder, VarianceEncoder
from torch.optim.swa_utils import AveragedModel, SWALR


class Trainer:
    def __init__(self, cfg: MyConfig) -> None:
        self.cfg = cfg

        debug = self.cfg.train.debug
        if debug:
            import debugpy

            debugpy.listen(5678)
            print("Waiting for debugger attach")
            debugpy.wait_for_client()

        self.device = utils.get_device(self.cfg.hardware.device)
        self.num_channels_list = get_in_dim(self.cfg.data_chunk.chunks)
        max_num_channels = max(self.num_channels_list)

        self.shuffle_all = "SHUFFLE_ALL"

        self.in_dim = self.cfg.model.in_dim = utils.default(self.cfg.model.in_dim, max_num_channels)
        self.data_channels = {}
        self.data_classes_train = (
            None  ## classes of dataset, e.g., ['airplane', 'bird', ...] for CIFAR10
        )
        self.seed = utils.default(self.cfg.train.seed, np.random.randint(1000, 1000000))
        job_id = utils.default(self.cfg.logging.scc_jobid, None)

        self.jobid_seed = (
            f"jobid{job_id}_seed{self.seed}" if job_id is not None else f"seed{self.seed}"
        )
        _project_name = self.cfg.logging.wandb.project_name
        self.project_name = utils.default(_project_name, f"morphemv2_v8")

        self.all_chunks = [list(chunk.keys())[0] for chunk in self.cfg.data_chunk.chunks]
        self.cfg.eval.meta_csv_file = "enriched_meta.csv"

        ## auto set eval batch size to maximize GPU memory usage
        if not self.cfg.eval.batch_size:
            if "depthwise" not in self.cfg.model.name:
                ## bs=512, takes 12 GB memory
                gpu_mem = utils.get_gpu_mem()
                eval_batch_size = int(512 * gpu_mem / 14)
                # round to the nearest power of 2
                eval_batch_size = 2 ** int(np.log2(eval_batch_size))
            else:
                eval_batch_size = 128  ## too large will cause error

            self.cfg.eval.batch_size = eval_batch_size
            print(f"self.cfg.eval.batch_size: {self.cfg.eval.batch_size}")

        #### model = adaptive_interface + shared_part
        ## optional: train adaptive_interface only for the first few epochs
        fine_tune_lr = self.cfg.optimizer.params["lr"]
        adaptive_interface_lr = self.cfg.train.adaptive_interface_lr
        ## Default: set small lr for adaptive interface
        self.cfg.train.adaptive_interface_lr = utils.default(
            adaptive_interface_lr, fine_tune_lr * 100
        )

        #### add some info to the cfg for tracking purpose
        self.cfg.tag = utils.default(self.cfg.tag, "-".join(self.all_chunks))
        self.cfg.train.seed = self.seed
        self.cfg.logging.wandb.project_name = self.project_name

        self.checkpoints = os_join(
            self.cfg.train.checkpoints,
            self.cfg.dataset.name,
            str(DataChunk(cfg.data_chunk.chunks)),
            self.jobid_seed,
        )

        self.start_epoch = 1
        self.best_model_path = os_join(self.checkpoints, "model_best.pt")

        #### Define metrics for training and evaluation
        self.eval_metric = "{split}_{chunk_name}/{metric}"
        self.eval_metric_all_chunks_obs = "{split}_ALL_CHUNKS/{metric}_obs"
        self.eval_metric_all_chunks_avg_chunk = "{split}_ALL_CHUNKS/{metric}_avg_chunk"
        self.eval_metric_all_chunks_best = "{split}_BEST_ALL_CHUNKS/{metric}_obs"
        self.eval_metric_names = ["acc", "f1"]

        self.train_loss_fn = "ce" if self.cfg.dataset.name in ["cifar10", "cifar100"] else "proxy"
        self.train_metric = "{split}_{chunk_name}/loss"
        self.train_metric_all_chunks = "{split}_ALL_CHUNKS/loss"

        self.best_res_all_chunks = collections.defaultdict(lambda: BestResult())

        self.train_loaders = {}
        self.val_loaders = {}
        self.test_loaders = {}
        self.num_loaders = None

        #### Build datasets, model, optimizer, logger
        self._build_dataset()

        if self.cfg.train.miro:  ## build MIRO, should be done before building model
            self.cfg.model.num_classes = len(self.data_classes_train)  ## duplicate, but ok for now
            if (
                hasattr(self.cfg.model, "pooling_channel_type")
                and self.cfg.model.pooling_channel_type == ChannelPoolingType.ATTENTION
            ):
                self.pre_featurizer = getattr(models, self.cfg.model.name)(
                    self.cfg.model,
                    freeze="all",
                    attn_pooling_params=self.cfg.attn_pooling,
                ).to(self.device)
                self.featurizer = getattr(models, self.cfg.model.name)(
                    self.cfg.model, attn_pooling_params=self.cfg.attn_pooling
                ).to(self.device)
            else:
                self.pre_featurizer = getattr(models, self.cfg.model.name)(
                    self.cfg.model, freeze="all"
                ).to(self.device)

                self.featurizer = getattr(models, self.cfg.model.name)(self.cfg.model).to(
                    self.device
                )

            chunk_name = self.cfg.dataset.name
            dims = {
                "Allen": 3,
                "HPA": 4,
                "CP": 5,
                "morphem70k": 3,
            }  ## "morphem70k": 3 is placeholder, it doesn't matter for shared_MIRO
            # build mean/var encoders
            shapes = get_shapes(
                self.pre_featurizer,
                (
                    dims[chunk_name],
                    self.cfg.dataset.img_size,
                    self.cfg.dataset.img_size,
                ),
            )
            self.mean_encoders = nn.ModuleList([MeanEncoder(shape) for shape in shapes]).to(
                self.device
            )
            self.var_encoders = nn.ModuleList([VarianceEncoder(shape) for shape in shapes]).to(
                self.device
            )
        self._build_model()
        self._build_log()

        ## TODO: may make an toggle here: we don't need these for evaluation mode
        self.updates_per_epoch = len(self.train_loaders[self.shuffle_all])
        self.total_epochs_all_chunks = self.cfg.train.num_epochs
        self.optimizer = self._build_optimizer()
        # self.scheduler = self._build_scheduler()
        self.scheduler = None  ## build scheduler later in training loop
        if self.cfg.train.resume_train:
            resume_path = os_join(self.checkpoints, self.cfg.train.resume_model)
            last_epoch = self._load_model(resume_path)
            self.start_epoch = last_epoch + 1

        self.use_amp = self.cfg.train.use_amp
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)  # type: ignore

        self._log_config_and_model_info()  ## log model info to wandb

        if self.cfg.train.swa or self.cfg.train.swad:
            self.swa_model = AveragedModel(self.model)
            self.swa_scheduler = SWALR(self.optimizer, swa_lr=self.cfg.train.swa_lr)

    @property
    def current_lr(self) -> float:
        lr = self.optimizer.param_groups[0]["lr"]  # or self.scheduler.get_last_lr()[-1]
        ## get all lrs:
        lrs = [param_group["lr"] for param_group in self.optimizer.param_groups]  # type: ignore
        if len(set(lrs)) == 1:
            return lrs[0]
        else:
            return lrs[0]
            ## TODO
            # return lrs
            # raise NotImplementedError("multiple lrs not supported yet")

    def _get_forward_mode(self) -> str:
        """
        Forward function for different models are different.
        Some requires chunk_name as input, some don't.
        This is a helper function for _forward_model().
        """
        if isinstance(
            self.model,
            (
                SharedConvNeXt,
                SliceParamConvNeXt,
                TemplateMixingConvNeXt,
                TemplateMixingConvNeXtV2,
                HyperConvNeXt,
                DepthwiseConvNeXt,
                ConvNeXtSharedMIRO,
                DepthwiseConvNeXtMIRO,
                HyperConvNeXtMIRO,
                SliceParamConvNeXtMIRO,
                TemplateMixingConvNeXtV2MIRO,
            ),
        ):
            forward_mode = "need_chunk_name"
        else:
            forward_mode = "normal_forward"
        return forward_mode

    def _forward_model(self, x, chunk_name: str):
        """
        forward step, depending on the type of model
        @param x:
        @param chunk_name:
        @return:
        """
        if self.forward_mode == "need_chunk_name":
            output = self.model(x, chunk_name)
        else:
            output = self.model(x)
        return output

    ## training loop
    def train(self):
        epoch_timer = utils.Time1Event()
        if not self.cfg.train.debug:
            self.logger.info("Before training, evaluate:")
            if self.cfg.dataset.name in ["cifar10", "cifar100"]:
                eval_loggers = {
                    DataSplit.TRAIN: {},
                    DataSplit.VAL: {},
                    DataSplit.TEST: {},
                }
                for chunk_name in self.all_chunks:
                    for split in DataSplit.get_all_splits():
                        eval_res = self.eval_cifar(split, chunk_name, epoch=0)
                        if eval_res:  ## avoid the case when VAL is not available
                            eval_loggers[split].update(eval_res)
            elif self.cfg.dataset.name in ["Allen", "HPA", "CP", "morphem70k"]:
                self.eval_morphem70k(epoch=0)  ## evaluate off the shelf model
            else:
                raise NotImplementedError(f"dataset {self.cfg.dataset.name} not supported yet")

        num_epochs = self.cfg.train.num_epochs + self.start_epoch - 1
        for epoch in range(self.start_epoch, num_epochs + 1):
            ### only train the adaptive interface for the first few epochs
            if epoch == self.start_epoch and self.cfg.train.adaptive_interface_epochs > 0:
                model_utils.toggle_grad(self.model.feature_extractor, requires_grad=False)
                self.logger.info(
                    f"freeze the feature extractor for the first {self.cfg.train.adaptive_interface_epochs} epochs"
                )

            ## finetune all the whole model
            if epoch == self.start_epoch + self.cfg.train.adaptive_interface_epochs:
                last_n_layers = self.cfg.model.unfreeze_last_n_layers
                model_utils.unfreeze_last_layers(
                    self.model.feature_extractor, num_last_layers=last_n_layers
                )
                self.logger.info(
                    f"unfreeze the last {last_n_layers} layers of the feature extractor"
                )

                ## build scheduler
                ## set lr of adaptive interface to lr of the whole model
                self.optimizer.param_groups[0]["lr"] = self.optimizer.param_groups[1]["lr"]

                self.scheduler = self._build_scheduler()
                self.logger.info(
                    f"build scheduler after unfreezing the last {last_n_layers} layers"
                )

                utils.analyze_model(self.model, False)

            ## Log
            self.logger.info(
                f"\n[{utils.datetime_now()}] Start Epoch {epoch}/{self.total_epochs_all_chunks}"
            )

            ## Scheduler per epoch
            if self.scheduler and not (
                (self.cfg.train.swa or self.cfg.train.swad) and epoch > self.cfg.train.swa_start
            ):
                self.scheduler.step(epoch)

            ## train
            self.train_one_epoch(epoch, self.shuffle_all)

            ## Evaluate on ALL chunks
            if self.cfg.dataset.name in ["cifar10", "cifar100"]:
                for chunk_name in self.all_chunks:
                    for split in DataSplit.get_all_splits():
                        eval_res = self.eval_cifar(split, chunk_name, epoch=epoch)
                        if eval_res:  ## avoid the case when VAL is not available
                            eval_loggers[split].update(eval_res)
                ## update best results for CIFAR
                self._update_best_res_all_chunks_cifar(eval_loggers, epoch)
            else:
                if not self.cfg.train.debug:  ## skip expensive evaluation on debug mode
                    self.eval_morphem70k(epoch=epoch)

            ## save cur model
            if self.cfg.train.save_model:
                utils.mkdir(self.checkpoints)
                cur_model_path = os_join(self.checkpoints, f"model_{epoch}.pt")
                self._save_model(path=cur_model_path, epoch=epoch, val_acc=None)

            ## Logging stuff
            epoch_timer.update()
            self.logger.info({"minute/epoch": round(epoch_timer.avg / 60, 2)})
            need_time = utils.convert_secs2time(
                epoch_timer.avg * (num_epochs - epoch), return_string=True
            )

            self.logger.info(need_time)  # type: ignore
            self.logger.info("=" * 40)

        self._finish_training()

    @torch.no_grad()
    def eval_cifar(self, split: DataSplit, chunk_name: str, epoch: int):
        """
        :param split: either TRAIN, VAL, TEST. used to determine which part of the dataset to evaluate
        :param chunk_name: e.g., "red", "red_green" for CIFAR
        :param epoch: current epoch
        :return:
        """
        if split == DataSplit.TRAIN:
            loader = self.train_loaders[chunk_name]
        elif split == DataSplit.VAL:
            loader = self.val_loaders[chunk_name]
        elif split == DataSplit.TEST:
            loader = self.test_loaders[chunk_name]
        else:
            raise ValueError(f"{split} is not valid!")

        if loader is None:
            return None

        self.model.eval()
        metrics_logger = collections.defaultdict(lambda: AverageMeter())

        for batch in loader:
            x, y = utils.move_to_cuda(batch, self.device)
            x = get_channel(self.cfg.dataset.name, self.data_channels[chunk_name], x, self.device)
            output = self._forward_model(x, chunk_name)
            if self.cfg.dataset.name in ["cifar10", "cifar100"]:
                loss = torch.nn.CrossEntropyLoss()(output, y)
                accuracy = self.eval_accuracy(output, y)

                loss_key = self.train_metric.format(split=split, chunk_name=chunk_name)
                acc_key = self.eval_metric.format(split=split, chunk_name=chunk_name, metric="acc")
                loss_dict = {loss_key: loss.item(), acc_key: accuracy}
                for k, v in loss_dict.items():
                    metrics_logger[k].update(v, len(y))
            else:
                raise NotImplementedError()

        self.logger.info(
            {k: v.avg for k, v in metrics_logger.items()}, sep="| ", padding_space=True
        )
        if self.cfg.train.debug:
            self.logger.info("----------- DEBUG MODE!!! -----------")
        return metrics_logger

    @torch.no_grad()
    def eval_morphem70k(self, epoch: int):
        def log_res(eval_cfg, knn_metric):
            call_umap = eval_cfg["umap"] and (epoch == 0 or epoch == self.cfg.train.num_epochs)
            if knn_metric in ["l2", "cosine"]:
                full_res = run_benchmark(
                    eval_cfg["root_dir"],
                    eval_cfg["dest_dir"],
                    eval_cfg["feature_dir"],
                    eval_cfg["feature_file"],
                    eval_cfg["classifier"],
                    call_umap,
                    eval_cfg["use_gpu"],
                    knn_metric,
                    # self.cfg.dataset.name,  # quick hack to run benchmark on only 1 dataset
                )
                ## log results
                full_res["key"] = full_res.iloc[:, 0:3].apply(
                    lambda x: "/".join(x.astype(str)), axis=1
                )

                acc = dict(
                    zip(
                        full_res["key"] + f"/{knn_metric}/acc",
                        full_res["accuracy"] * 100,
                    )
                )
                f1 = dict(
                    zip(
                        full_res["key"] + f"/{knn_metric}/f1",
                        full_res["f1_score_macro"],
                    )
                )
                metrics_logger = {
                    **acc,
                    **f1,
                    f"{classifier}/{knn_metric}/score_acc/": np.mean(list(acc.values())[1:]),
                    f"{classifier}/{knn_metric}/score_f1/": np.mean(list(f1.values())[1:]),
                }
            else:
                knn_metric = "l2"
                full_res = run_benchmark(
                    eval_cfg["root_dir"],
                    eval_cfg["dest_dir"],
                    eval_cfg["feature_dir"],
                    eval_cfg["feature_file"],
                    eval_cfg["classifier"],
                    call_umap,
                    eval_cfg["use_gpu"],
                    knn_metric,
                    # self.cfg.dataset.name,  # quick hack to run benchmark on only 1 dataset
                )
                ## log results
                full_res["key"] = full_res.iloc[:, 0:3].apply(
                    lambda x: "/".join(x.astype(str)), axis=1
                )
                acc = dict(zip(full_res["key"] + f"/acc", full_res["accuracy"] * 100))
                f1 = dict(zip(full_res["key"] + f"/f1", full_res["f1_score_macro"]))
                metrics_logger = {
                    **acc,
                    **f1,
                    f"{classifier}/score_acc/": np.mean(list(acc.values())[1:]),
                    f"{classifier}/score_f1/": np.mean(list(f1.values())[1:]),
                }
            self.logger.info(metrics_logger, sep="| ", padding_space=True)

        if self.cfg.eval.only_eval_first_and_last:
            if epoch != 0 and epoch != self.cfg.train.num_epochs:
                return None  ## bail out, skip this expensive evaluation

        self.model.eval()

        eval_cfg = deepcopy(self.cfg.eval)
        ## make a new folder for each epoch
        scc_jobid = utils.default(self.cfg.logging.scc_jobid, "")
        FOLDER_NAME = (
            f'{utils.datetime_now("%Y-%b-%d")}_seed{self.cfg.train.seed}_sccjobid{scc_jobid}'
        )
        eval_cfg.dest_dir = os_join(
            eval_cfg.dest_dir.format(FOLDER_NAME=FOLDER_NAME), f"epoch_{epoch}"
        )
        if epoch == 0:  ### store the first epoch features in a separate folder
            eval_cfg.feature_dir = eval_cfg.feature_dir.format(FOLDER_NAME=FOLDER_NAME) + "_epoch0"
        else:
            eval_cfg.feature_dir = eval_cfg.feature_dir.format(FOLDER_NAME=FOLDER_NAME)

        start_time = time.time()

        out_path_list = []
        for chunk_name in self.all_chunks:
            feat_outputs = []  # store feature vectors
            eval_loader = self.test_loaders[chunk_name]
            for batch in eval_loader:
                x = utils.move_to_cuda(batch, self.device)
                output = self._forward_model(x, chunk_name)
                if self.cfg.train.miro:
                    output = output[0]
                feat_outputs.append(output)

            print(
                f"done forward passes {chunk_name} in {(time.time() - start_time) / 60:.2f} minutes"
            )
            feat_outputs = torch.cat(feat_outputs, dim=0).cpu().numpy()
            print(
                f"Have all features for {chunk_name} in {(time.time() - start_time) / 60:.2f} minutes"
            )

            folder_path = os_join(eval_cfg.feature_dir, chunk_name)
            utils.mkdir(folder_path)

            out_path = os_join(folder_path, eval_cfg.feature_file)
            out_path_list.append(out_path)
            utils.write_numpy(feat_outputs, out_path)
            print(
                f"-- Done writing features for {chunk_name} in total {(time.time() - start_time) / 60:.2f} minutes"
            )

        ## after we have all features for 3 chunks (i.e., Allen, HPA, CP), we run the benchmark
        torch.cuda.empty_cache()
        for classifier in eval_cfg.classifiers:
            eval_cfg.classifier = classifier

            if classifier == "knn":
                for knn_metric in eval_cfg.knn_metrics:
                    log_res(eval_cfg=eval_cfg, knn_metric=knn_metric)
            else:
                log_res(eval_cfg=eval_cfg, knn_metric=None)

        if self.cfg.train.debug:
            self.logger.info("----------- DEBUG MODE!!! -----------")

        if self.cfg.eval.clean_up:
            for out_path in out_path_list:
                os.remove(out_path)
            self.logger.info(f"cleaned up {len(out_path_list)} files after evaluation")

    @torch.no_grad()
    def eval_accuracy(self, output, y):
        pred = torch.argmax(output, dim=-1)
        correct = 100 * torch.sum(pred == y) / len(y)
        return correct.item()

    def train_one_epoch(self, epoch: int, chunk_name: str):
        """
        train one epoch for `chunk_name`, chunk_name can be one of ["red", "red_green", `self.shuffle_all`, ...]
        :param epoch:
        :param chunk_name
        :return:
        """

        self.model.train()
        verbose, bid = self.cfg.train.verbose_batches, 0

        loss_meter = collections.defaultdict(lambda: AverageMeter())

        for bid, batch in enumerate(self.train_loaders[chunk_name], 1):
            num_updates = (epoch - 1) * self.updates_per_epoch + bid

            ## a batch consists of images from all chunks
            loss_dict = self.train_one_batch(batch, num_updates=num_updates, epoch=epoch)

            for k, v in loss_dict.items():
                loss_meter[k].update(v)

            if bid % verbose == 0:
                self._update_batch_log(
                    epoch=epoch, bid=bid, lr=self.current_lr, loss_meter=loss_meter
                )

            if self.cfg.train.debug and bid > 10:
                print("Debug mode, only run 10 batches")
                break

        if bid % verbose != 0:
            self._update_batch_log(epoch=epoch, bid=bid, lr=self.current_lr, loss_meter=loss_meter)
        if self.cfg.train.swa and not self.cfg.train.swad and epoch > self.cfg.train.swa_start:
            self.swa_model.update_parameters(self.model)
            self.swa_scheduler.step()
        # utils.gpu_mem_report()
        return None

    def train_one_batch(
        self, batch: Tuple[Dict[str, Tensor], Tensor], num_updates: int, epoch: int
    ) -> Dict:
        batch = utils.move_to_cuda(batch, self.device)

        ## Zero out grads
        self.optimizer.zero_grad()
        for chunk_name in self.all_chunks:
            ## if more than 1 chunk/dataset, and chunk_name/dataset not in this batch, skip
            if len(self.all_chunks) == 1:
                x, y = batch
            else:
                if chunk_name in batch:
                    x, y = batch[chunk_name]["image"], batch[chunk_name]["label"]
                else:
                    continue
            x = get_channel(
                self.cfg.dataset.name,
                data_channels=self.data_channels[chunk_name],
                x=x,
                device=self.device,
            )
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                output = self._forward_model(x, chunk_name)
                if self.cfg.dataset.name in ["cifar10", "cifar100"]:
                    loss = torch.nn.CrossEntropyLoss()(output, y)
                elif self.cfg.dataset.name in ["Allen", "HPA", "CP", "morphem70k"]:
                    if self.cfg.model.learnable_temp:
                        scale = self.model.logit_scale.exp()
                    else:
                        scale = self.model.scale
                    if self.cfg.train.miro:
                        y_pred, inter_feats = output
                        loss = proxy_loss(self.model.proxies, y_pred, y, scale)
                        with torch.no_grad():
                            if "base" in self.cfg.model.name:
                                _, pre_feats = self.pre_featurizer(x)
                            else:
                                _, pre_feats = self.pre_featurizer(x, chunk=chunk_name)

                        reg_loss = 0.0
                        for f, pre_f, mean_enc, var_enc in model_utils.zip_strict(
                            inter_feats,
                            pre_feats,
                            self.mean_encoders,
                            self.var_encoders,
                        ):
                            # mutual information regularization
                            mean = mean_enc(f)
                            var = var_enc(f)
                            vlb = (mean - pre_f).pow(2).div(var) + var.log()
                            reg_loss += vlb.mean() / 2.0

                        loss += reg_loss * self.cfg.train.miro_ld
                    else:
                        loss = proxy_loss(self.model.proxies, output, y, scale)
                else:
                    raise NotImplementedError(f"dataset {self.cfg.dataset.name} not implemented")

            ## scale loss then call backward to have scaled grads.
            self.scaler.scale(loss).backward()
            # loss.backward()

        ## after looping over all chunks, we call optimizer.step() once
        if exists(self.cfg.train.clip_grad_norm):
            self.scaler.unscale_(self.optimizer)  # unscale grads of optimizer
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.clip_grad_norm)

        ## unscale grads of `optimizer` if it hasn't, then call optimizer.step() if grads
        # don't contain NA(s), inf(s) (o.w. skip calling)
        self.scaler.step(self.optimizer)
        # self.optimizer.step()

        ## update scaler
        self.scaler.update()

        ## Scheduler per batch
        if self.scheduler and not (self.cfg.train.swad and epoch > self.cfg.train.swa_start):
            self.scheduler.step_update(num_updates=num_updates)

        loss_dict = {
            self.train_metric.format(
                split="TRAINING_LOSS", chunk_name=self.shuffle_all
            ): loss.item()
        }  ## loss on training

        if self.cfg.train.swad and epoch > self.cfg.train.swa_start:
            self.swa_model.update_parameters(self.model)
            self.swa_scheduler.step()

        return loss_dict

    def _update_batch_log(
        self,
        epoch: int,
        bid: int,
        lr: float,
        loss_meter: Dict,
        only_print: bool = False,
    ) -> None:
        msg_dict = {"epoch": epoch, "bid": bid, "lr": lr}
        if self.cfg.model.learnable_temp:
            scale = self.model.logit_scale.exp()
            msg_dict["temperature"] = 1 / scale.data.item()

        for metric, value in loss_meter.items():
            msg_dict[metric] = value.avg
            value.reset()
        if only_print:
            print(msg_dict)
        else:
            self.logger.info(msg_dict)
        return None

    def _get_avg_metric_all_chunks_by_obs(
        self, key_base: str, metric: str, logger_dict: Dict, split: DataSplit
    ):
        """
        get avg on ALL CHUNKS, not only on training_chunks,
        as training_chunks can be a subset of self.chunk_names
        :param key_base:
        :return:
        """
        total_obs, total_corrects = 0, 0
        for chunk_name in self.all_chunks:
            key = key_base.format(split=split, chunk_name=chunk_name, metric=metric)
            logger = logger_dict[key]
            num_obs, num_corrects = logger.count, logger.sum
            total_obs += num_obs
            total_corrects += num_corrects
        return total_corrects / total_obs

    def _get_avg_metric_all_chunks_by_avg_chunk(
        self, key_base: str, metric: str, logger_dict: Dict, split: DataSplit
    ):
        res_list = []
        for chunk_name in self.all_chunks:
            key = key_base.format(split=split, chunk_name=chunk_name, metric=metric)
            logger = logger_dict[key]
            res_list.append(logger.avg)
        return np.mean(np.array(res_list))

    def _update_best_res_all_chunks_cifar(self, eval_loggers: Dict[DataSplit, Dict], epoch: int):
        """
        evaluate on all chunks, i.e., self.all_chunks (NOT only training_chunks)
        Used during training to update the best results so far
        :return:
        """
        for split in DataSplit.get_all_splits():
            if not eval_loggers[split]:
                print("skipped", split)
                continue
            logger_dict = eval_loggers[split]
            cur_acc_obs = self._get_avg_metric_all_chunks_by_obs(
                key_base=self.eval_metric,
                metric="acc",
                logger_dict=logger_dict,
                split=split,
            )
            cur_acc_chunk = self._get_avg_metric_all_chunks_by_avg_chunk(
                key_base=self.eval_metric,
                metric="acc",
                logger_dict=logger_dict,
                split=split,
            )

            acc_obs_key = self.eval_metric_all_chunks_obs.format(split=split, metric="acc")
            acc_chunk_key = self.eval_metric_all_chunks_avg_chunk.format(split=split, metric="acc")

            cur = {
                acc_obs_key: cur_acc_obs,
                acc_chunk_key: cur_acc_chunk,
            }
            loss_key, f1_chunk_key = None, None
            cur_loss = self._get_avg_metric_all_chunks_by_obs(
                key_base=self.train_metric, logger_dict=logger_dict, split=split
            )
            loss_key = self.train_metric_all_chunks.format(split=split)
            cur[loss_key] = cur_loss

            self.logger.info(
                {k: v for k, v in cur.items()},
                sep="| ",
                padding_space=True,
                pref_msg=f"epoch = {epoch}; ALL CHUNKS: ",
            )

            ## Update best result
            best_acc_obs = self.best_res_all_chunks[split].avg_acc_obs
            if cur[acc_obs_key] > best_acc_obs:
                self.best_res_all_chunks[split].avg_acc_obs = cur[acc_obs_key]
                self.best_res_all_chunks[split].avg_acc_chunk = cur[acc_chunk_key]
                self.best_res_all_chunks[split].avg_loss = cur[loss_key] if loss_key else None
                self.best_res_all_chunks[split].avg_f1_chunk = (
                    cur[f1_chunk_key] if f1_chunk_key else None
                )
                self.best_res_all_chunks[split].epoch = epoch

                msg = f"updated best results, best avg acc_obs={cur[acc_obs_key]}; best epoch {epoch}.\n"
                self.logger.update_best_result(
                    msg=msg,
                    metric=f'{self.eval_metric_all_chunks_best.format(split=split, metric="acc")}',
                    val=cur[acc_obs_key],
                )

    def _build_dataset(self):
        data_cfg = self.cfg.dataset
        dataset = data_cfg.name
        batch_size = self.cfg.train.batch_size
        eval_batch_size = self.cfg.eval.batch_size
        img_size = self.cfg.dataset.img_size

        num_workers = self.cfg.hardware.num_workers
        data_chunks = self.cfg.data_chunk.chunks

        root_dir = data_cfg.root_dir
        file_name = data_cfg.file_name
        tps_prob = self.cfg.train.tps_prob
        ssl_flag = self.cfg.train.ssl

        for chunk in data_chunks:
            chunk_name = list(chunk.keys())[0]
            train_loader, val_loader, test_loader = get_train_val_test_loaders(
                dataset=dataset,
                img_size=img_size,
                chunk_name=chunk_name,
                seed=self.seed,
                batch_size=batch_size,
                eval_batch_size=eval_batch_size,
                num_workers=num_workers,
                root_dir=root_dir,
                file_name=file_name,
                tps_prob=tps_prob,
                ssl_flag=ssl_flag,
            )

            self.train_loaders[chunk_name] = train_loader
            self.val_loaders[chunk_name] = val_loader
            self.test_loaders[chunk_name] = test_loader
            self.data_channels[chunk_name] = chunk[chunk_name]

        train_loader_all = make_random_instance_train_loader(
            dataset,
            img_size,
            batch_size=batch_size,
            seed=self.seed,
            num_workers=num_workers,
            root_dir=root_dir,
            file_name=file_name,
            tps_prob=tps_prob,
            ssl_flag=ssl_flag,
        )
        self.train_loaders[self.shuffle_all] = utils.default(train_loader_all, train_loader)

        self.num_loaders = len(data_chunks)
        self.data_classes_train, self.data_classes_test = get_classes(
            dataset, file_name
        )  ## list of class names

    def _build_model(self):
        assert self.data_classes_train is not None, "self.data_classes_train is None!"
        self.cfg.model.num_classes = len(self.data_classes_train)

        if self.cfg.train.miro:
            self.model = self.featurizer

        else:
            if (
                hasattr(self.cfg.model, "pooling_channel_type")
                and self.cfg.model.pooling_channel_type == ChannelPoolingType.ATTENTION
            ):
                if "miro" in self.cfg.model.name:
                    self.model = getattr(models, self.cfg.model.name)(
                        self.cfg.model,
                        freeze=None,
                        attn_pooling_params=self.cfg.attn_pooling,
                    )
                else:
                    self.model = getattr(models, self.cfg.model.name)(
                        self.cfg.model,
                        attn_pooling_params=self.cfg.attn_pooling,
                    )
            else:
                self.model = getattr(models, self.cfg.model.name)(self.cfg.model)

        self.model = self.model.to(self.device)
        # https://pytorch.org/get-started/pytorch-2.0/
        ## check torch version >= 2
        self.forward_mode = (
            self._get_forward_mode()
        )  # determine the type of self.model before compiling
        if torch.__version__ >= "2.0.0" and self.cfg.train.get("compile_pytorch", False):
            self.model = torch.compile(self.model, mode="reduce-overhead")

        if self.cfg.hardware.multi_gpus == "DataParallel":
            print("os.environ['CUDA_VISIBLE_DEVICES']", os.getenv("CUDA_VISIBLE_DEVICES"))
            print(f"using {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)
        elif self.cfg.hardware.multi_gpus == "DistributedDataParallel":
            raise NotImplementedError()
        elif self.cfg.hardware.multi_gpus is None:
            pass
        else:
            raise ValueError(f"{self.cfg.hardware.multi_gpus} is not valid!")

    def _build_log(self):
        self.logger = MyLogging(
            self.cfg,
            model=self.model,
            job_id=self.jobid_seed,
            project_name=self.project_name,
        )

    def _log_config_and_model_info(self):
        if self.cfg.logging.wandb.use_wandb:
            self.logger.log_config(self.cfg)
        self.logger.info(OmegaConf.to_yaml(self.cfg))
        self.logger.info(str(self.model))
        total_num, trainable_num = utils.analyze_model(self.model, print_trainable=True)
        self.logger.info({"total_params": total_num, "trainable_params": trainable_num})
        self.logger.info("Pytorch version: {}".format(torch.__version__))
        self.logger.info("Cuda version: {}".format(torch.version.cuda))  # type: ignore
        self.logger.info("Cudnn version: {}".format(torch.backends.cudnn.version()))  # type: ignore

    def _build_optimizer(self):
        name = self.cfg.optimizer.name
        optimizer_cfg = dict(**self.cfg.optimizer.params)
        adaptive_interface_lr = self.cfg.train.adaptive_interface_lr
        if self.model.adaptive_interface is not None:
            adaptive_params = [p for p in self.model.adaptive_interface.parameters()]  # type: ignore
        else:
            adaptive_params = []
        feature_extractor_params = [p for p in self.model.feature_extractor.parameters()]  # type: ignore
        lr = optimizer_cfg["lr"]
        miro_lr_mult = self.cfg.train.miro_lr_mult
        param_list = [
            {"params": adaptive_params, "lr": adaptive_interface_lr},
            {"params": feature_extractor_params, "lr": lr},
        ]

        if self.cfg.train.miro:
            miro_params = [
                {"params": self.mean_encoders.parameters(), "lr": lr * miro_lr_mult},
                {
                    "params": self.var_encoders.parameters(),
                    "lr": lr * miro_lr_mult,
                },
            ]
            param_list += miro_params

        optimizer = make_my_optimizer(name, param_list, optimizer_cfg)
        return optimizer

    def _build_scheduler(self):
        # https://github.com/rwightman/pytorch-image-models/blob/9f5bba9ef9db8a32a5a04325c8eb181c9f13a9b2/timm/scheduler/scheduler_factory.py
        sched_name = self.cfg.scheduler.name.lower()
        if sched_name == "none":
            ## bail out if no scheduler
            return None
        sched_cfg = dict(**self.cfg.scheduler.params)
        sched_cfg["t_initial"] = (
            self.cfg.train.num_epochs - self.cfg.train.adaptive_interface_epochs
        )
        t_in_epochs = sched_cfg.get("t_in_epochs", True)
        convert_to_batch = self.cfg.scheduler.convert_to_batch
        if convert_to_batch and not t_in_epochs:
            for k in sched_cfg:
                if k in ["t_initial", "warmup_t", "decay_t"]:
                    if isinstance(sched_cfg[k], ListConfig):
                        sched_cfg[k] = (np.array(sched_cfg[k]) * self.updates_per_epoch).tolist()
                    else:
                        sched_cfg[k] = sched_cfg[k] * self.updates_per_epoch
        scheduler = create_my_scheduler(self.optimizer, sched_name, sched_cfg)

        self.logger.info(
            {
                "updates_per_epoch": self.updates_per_epoch,
                "total_epochs_all_chunks": self.total_epochs_all_chunks,
            }
        )

        self.logger.info(scheduler.state_dict())
        return scheduler

    def _save_model(self, path: str, epoch: int, val_acc: float | None):
        state_dict = {
            "epoch": epoch,
            "accuracy": val_acc,
            "config": self.cfg,
            "optimizer_params": self.optimizer.state_dict(),
            "model_params": self.model.state_dict(),
            "scheduler_params": self.scheduler.state_dict() if exists(self.scheduler) else None,
            "scaler_params": self.scaler.state_dict(),
            "datetime": utils.datetime_now(),
        }

        torch.save(state_dict, path)
        self.logger.info("saved model to {}".format(path))

    def _load_model(self, path):
        state_dict = torch.load(path)

        self.model.load_state_dict(state_dict["model_params"])
        self.optimizer.load_state_dict(state_dict["optimizer_params"])
        if "scheduler_params" in state_dict:
            self.scheduler.load_state_dict(state_dict["scheduler_params"])
        self.scaler.load_state_dict(state_dict["scaler_params"])

        epoch = int(state_dict.get("epoch", 0))
        self.logger.info("loaded model from {path}, epoch {epoch}".format(path=path, epoch=epoch))

        return epoch

    def _finish_training(self):
        best_res = self.best_res_all_chunks[DataSplit.TEST]

        # ## Update best results
        # best_epoch_path = os_join(self.checkpoints, f"model_{best_res.epoch}.pt")
        # os.system(f"cp {best_epoch_path} {self.best_model_path}")
        # print(f"copied the best model to {self.best_model_path}...")

        ## Log the best model
        if self.cfg.train.swa or self.cfg.train.swad:
            torch.optim.swa_utils.update_bn(self.train_loaders[self.shuffle_all], self.swa_model)
            self.model = self.swa_model

        self.logger.info(best_res.to_dict(), use_wandb=False, sep="| ", padding_space=True)
        h = w = int(self.cfg.dataset.img_size)
        best_model_path = self.best_model_path if self.cfg.train.save_model else ""
        self.logger.finish(
            msg_str="--------------- DONE TRAINING! ---------------",
            model=self.model,
            model_best_name=best_model_path,
            dummy_batch_x=torch.randn((1, self.in_dim, h, w)).to(self.device),
        )
        return None
