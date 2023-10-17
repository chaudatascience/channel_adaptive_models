from __future__ import annotations

import collections
import os
import time
from copy import deepcopy
from os.path import join as os_join
from typing import Optional, Dict, Tuple

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from einops import repeat
from omegaconf import OmegaConf, ListConfig
from torch import nn, Tensor
import torch.nn.functional as F
from sklearn.metrics import f1_score

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
from models.hypernetwork_resnet import HyperNetworkResNet
from models.loss_fn import proxy_loss
from models.shared_convnext import SharedConvNeXt
from models.shared_resnet import SharedResNet
from lr_schedulers import create_my_scheduler
from models.slice_param_convnext import SliceParamConvNeXt
from models.slice_param_resnet import SliceParamResNet
from models.depthwise_resnet import DepthWiseResNet
from models.depthwise_convnext_miro import DepthwiseConvNeXtMIRO
from models.template_mixing_convnext import TemplateMixingConvNeXt
from models.template_convnextv2 import TemplateMixingConvNeXtV2
from models.convnext_base_miro import convnext_base_miro
from models.convnext_shared_miro import ConvNeXtSharedMIRO
from models.hypernet_convnext_miro import HyperConvNeXtMIRO
from models.template_mixing_first_layer_resnet import TemplateMixingFirstLayerResNet
from models.slice_param_convnext_miro import SliceParamConvNeXtMIRO
from models.template_convnextv2_miro import TemplateMixingConvNeXtV2MIRO

from optimizers import make_my_optimizer
from utils import AverageMeter, exists
from custom_log import MyLogging
from models.model_utils import get_shapes, MeanEncoder, VarianceEncoder
from torch.optim.swa_utils import AveragedModel, SWALR
from trainer import Trainer

#SimCLR from https://github.com/sthalles/SimCLR/blob/master

class SimCLR(object):

    def __init__(self, cfg, device, **kwargs):
        self.cfg = cfg
        self.device = device


    def info_nce_loss(self, features):

        labels = torch.cat([torch.arange(features.size()[0]/2) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape
                # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        logits = logits / 0.07 #set temperature as 0.07
        return logits, labels

    # def train(self):

    #     scaler = GradScaler(enabled=True)

    #     self.logger.info(f"Start SimCLR training.")
    #     self.logger..info(f"Training with gpu: {self.device}.")

    #     for epoch_counter in range(self.cfg.train.num_epochs):
    #         for images, _ in tqdm(train_loader):
    #             images = torch.cat(images, dim=0)

    #             images = images.to(self.device)

    #             with autocast(enabled=True):
    #                 features = self.model(images)
    #                 logits, labels = self.info_nce_loss(features)
    #                 loss = self.criterion(logits, labels)

    #             self.optimizer.zero_grad()

    #             scaler.scale(loss).backward()

    #             scaler.step(self.optimizer)
    #             scaler.update()


class SSLTrainer(Trainer):
    def __init__(self, cfg: MyConfig) -> None:
        super().__init__(cfg)
        
        self.simclr = SimCLR(cfg, self.device)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
    
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

    def train_one_epoch(self, epoch, chunk_name):
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
            print("SWA AVERAGE MODEL!!!!!!!!!!!!!!!!!!!!")
            self.swa_model.update_parameters(self.model)
            self.swa_scheduler.step()
        # utils.gpu_mem_report()
        return None


    def train_one_batch(self, batch, num_updates, epoch):
        batch = utils.move_to_cuda(batch, self.device)

        ## Zero out grads
        self.optimizer.zero_grad()
        for chunk_name in self.all_chunks:
            ## if more than 1 chunk/dataset, and chunk_name/dataset not in this batch, skip
            if len(self.all_chunks) == 1:
                x, y = batch
                x = torch.flatten(x, end_dim=1)
                y = y.repeat_interleave(2)
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
                # print("OUTPUT SHAPE!!!!!!!!!!!", output.size())
                if self.cfg.dataset.name in ["cifar10", "cifar100"]:
                    loss = torch.nn.CrossEntropyLoss()(output, y)
                elif self.cfg.dataset.name in ["Allen", "HPA", "CP", "morphem70k"]:
                    if self.cfg.model.learnable_temp:
                        scale = self.model.logit_scale.exp()
                    else:
                        scale = self.model.scale
                    logits, labels = self.simclr.info_nce_loss(output)
                    # loss = self.criterion(logits, labels)
                    loss = self.cfg.train.ssl_lambda* self.criterion(logits, labels) + (1 - self.cfg.train.ssl_lambda)*torch.nn.CrossEntropyLoss()(output, y)
                else:
                    raise NotImplementedError(f"dataset {self.cfg.dataset.name} not implemented")

            ## scale loss then call backward to have scaled grads.
            self.scaler.scale(loss).backward()
            # loss.backward()

            if self.cfg.model.use_auto_rgn:
                rgn = utils.compute_autorgn(self.model)
                for x in rgn:
                    self.logger.info({"RGN": x})
                self.cfg.model.use_auto_rgn = False
                self.logger.info("logged RGN for 1 batch. Turn off auto rgn")

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
        if self.scheduler:
            self.scheduler.step_update(num_updates=num_updates)

        loss_dict = {
            self.train_metric.format(
                split="TRAINING_LOSS", chunk_name=self.shuffle_all
            ): loss.item()
        }  ## loss on training

        return loss_dict


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
        param_list = [
            {"params": adaptive_params, "lr": adaptive_interface_lr},
            {"params": feature_extractor_params, "lr": lr},
        ]
        optimizer = make_my_optimizer(name, param_list, optimizer_cfg)
        return optimizer



    def _finish_training(self):
        best_res = self.best_res_all_chunks[DataSplit.TEST]

        # ## Update best results
        # best_epoch_path = os_join(self.checkpoints, f"model_{best_res.epoch}.pt")
        # os.system(f"cp {best_epoch_path} {self.best_model_path}")
        # print(f"copied the best model to {self.best_model_path}...")

        ## Log the best model

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
