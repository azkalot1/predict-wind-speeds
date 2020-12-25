import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import (
    CyclicLR, CosineAnnealingLR, CosineAnnealingWarmRestarts, StepLR)
from warmup_scheduler import GradualWarmupScheduler
from torch.utils.data import DataLoader
import torch_optimizer as optim
from sklearn.metrics import mean_squared_error
import numpy as np
from .models import SimpleClassificationModel
from .dataset import WindDataset
from .transforms import get_training_trasnforms, mixup_data, mixup_criterion
import pandas as pd
from os.path import join as join_path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')


class WindModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.net = self.get_net()
        self.criterion = self.get_criterion()
        self.learning_rate = hparams.learning_rate
        self.batch_size = hparams.batch_size
        self.use_mixup = hparams.use_mixup
        if self.use_mixup:
            self.alpha = hparams.mixup_alpha

    def forward(self, x: torch.tensor):
        return self.net(x)

    def training_step(self, batch, batch_idx: int) -> dict:
        if self.use_mixup:
            x, targets_a, targets_b, lam = mixup_data(batch['features'], batch['target'], self.alpha)
            y_hat = self.forward(x)
            loss = mixup_criterion(self.criterion, y_hat, targets_a, targets_b, lam)
        else:
            y_hat = self.forward(batch['features'])
            loss = self.criterion(y_hat, batch['target'])

        y_hat = y_hat.detach().cpu().numpy()
        y_true = batch['target'].detach().cpu().numpy()
        batch_rmse = mean_squared_error(y_true, y_hat, squared=False)

        train_step = {
            "loss": loss,
            "predictions": y_hat,
            "targets": y_true
        }
        self.logger.experiment.log_metric('train/batch_loss', loss.item())
        self.logger.experiment.log_metric('train/batch_rmse', batch_rmse)
        return train_step

    def validation_step(self, batch, batch_idx: int) -> dict:
        y_hat = self.forward(batch['features'])
        loss = self.criterion(y_hat, batch['target'])

        y_hat = y_hat.detach().cpu().numpy()
        y_true = batch['target'].detach().cpu().numpy()
        batch_rmse = mean_squared_error(y_true, y_hat, squared=False)

        # log each most wrong image in a batch
        # save images
        img = batch['features'].cpu().detach().numpy()
        img = (img * 255).astype(int)
        val_step = {
            "loss": loss,
            "predictions": y_hat,
            "targets": y_true,
            "img": img
        }
        self.logger.experiment.log_metric('val/batch_loss', loss.item())
        self.logger.experiment.log_metric('val/batch_rmse', batch_rmse)
        return val_step

    def training_epoch_end(self, outputs: torch.tensor) -> dict:
        avg_loss = torch.stack(
            [x["loss"] for x in outputs]).detach().cpu().numpy().mean()
        predictions = np.concatenate(
            [x["predictions"] for x in outputs])
        targets = np.concatenate(
            [x["targets"] for x in outputs])
        rmse = mean_squared_error(targets, predictions, squared=False)

        # self.logger.experiment.log_metric('train_loss', avg_loss)
        # self.logger.experiment.log_metric('train_rmse', rmse)
        self.log('train_loss', avg_loss)
        self.log('train_rmse', rmse)

    def validation_epoch_end(self, outputs: torch.tensor) -> dict:
        avg_loss = torch.stack(
            [x["loss"] for x in outputs]).detach().cpu().numpy().mean()
        predictions = np.concatenate(
            [x["predictions"] for x in outputs])
        targets = np.concatenate(
            [x["targets"] for x in outputs])
        img = np.concatenate(
            [x["img"] for x in outputs])
        predictions = np.squeeze(predictions)
        targets = np.squeeze(targets)
        rmse = mean_squared_error(targets, predictions, squared=False)
        # find most wrong images
        prediction_diff = np.abs(predictions - targets)
        most_wrong = prediction_diff.argsort()[::-1][:self.batch_size]
        fig, ax = plt.subplots(4, 4, figsize=(16, 16))
        ax = ax.flatten()
        for ax_idx in range(min(len(ax), self.batch_size)):
            ax[ax_idx].imshow(img[most_wrong[ax_idx], -1], cmap='Greys_r')
            ax[ax_idx].axis('off')
            ax[ax_idx].set_title(f'predicted: {predictions[most_wrong[ax_idx]]}, gt: {targets[most_wrong[ax_idx]]}')
        plt.tight_layout()

        self.logger.experiment.log_image(
                'valid_misclassified_images',
                fig,
                description='Most incorrect images'
        )
        plt.close()
        # self.logger.experiment.log_metric('val_loss', avg_loss)
        # self.logger.experiment.log_metric('val_rmse', rmse)
        self.log('val_loss', avg_loss)
        self.log('val_rmse', rmse)

    def configure_optimizers(self):
        optimizer = self.get_optimizer()
        scheduler = self.get_scheduler(optimizer)

        return [optimizer], [scheduler]

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        # train_df = pd.read_csv(self.hparams.training_data_path)
        train_df = pd.read_csv(join_path(self.hparams.data_folder, f'fold{self.hparams.fold}_train.csv'))
        images = train_df['image_id'].values
        images = [x+'.jpg' for x in images]
        wind_speed = train_df['wind_speed'].values
        train_dataset = WindDataset(
                    images=images,
                    folder=self.hparams.image_folder,
                    wind_speed=wind_speed,
                    load_n=self.hparams.load_n,
                    type_agg=self.hparams.type_agg,
                    transform=get_training_trasnforms(self.hparams.training_transforms, self.hparams.resize),

            )
        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            drop_last=True
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        val_df = pd.read_csv(join_path(self.hparams.data_folder, f'fold{self.hparams.fold}_val.csv'))
        images = val_df['image_id'].values
        images = [x+'.jpg' for x in images]
        wind_speed = val_df['wind_speed'].values
        val_dataset = WindDataset(
                    images=images,
                    folder=self.hparams.image_folder,
                    wind_speed=wind_speed,
                    load_n=self.hparams.load_n,
                    type_agg=self.hparams.type_agg,
                    transform=get_training_trasnforms('valid', self.hparams.resize),

            )
        return DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True
        )

    @staticmethod
    def net_mapping(model_name: str, n_input_channels: int, pretrained: bool = True) -> torch.nn.Module:
        return SimpleClassificationModel(model_name, n_input_channels, pretrained=pretrained)

    def get_optimizer(self) -> object:
        if "adam" == self.hparams.optimizer:
            return torch.optim.Adam(
                self.net.parameters(),
                lr=self.learning_rate,
                weight_decay=self.hparams.weight_decay)
        elif "adamw" == self.hparams.optimizer:
            return torch.optim.AdamW(
                self.net.parameters(),
                lr=self.learning_rate,
                weight_decay=self.hparams.weight_decay)
        elif "sgd" == self.hparams.optimizer:
            return torch.optim.SGD(
                self.net.parameters(),
                lr=self.learning_rate,
                momentum=self.hparams.sgd_momentum,
                weight_decay=self.hparams.sgd_wd,
            )
        elif "adabelief" == self.hparams.optimizer:
            return optim.AdaBelief(
                self.net.parameters(),
                lr=self.learning_rate,
                weight_decay=self.hparams.weight_decay,
            )
        elif "adamp" == self.hparams.optimizer:
            return optim.AdamP(
                self.net.parameters(),
                lr=self.learning_rate,
                weight_decay=self.hparams.weight_decay,
            )
        elif "qhadam" == self.hparams.optimizer:
            return optim.QHAdam(
                self.net.parameters(),
                lr=self.learning_rate,
                weight_decay=self.hparams.weight_decay,
            )
        elif "radam" == self.hparams.optimizer:
            base_otim = optim.RAdam(
                self.net.parameters(),
                lr=self.learning_rate,
                weight_decay=self.hparams.weight_decay)
            return optim.Lookahead(
                base_otim,
                k=5
            )
        elif "ranger" == self.hparams.optimizer:
            return optim.Ranger(
                self.net.parameters(),
                lr=self.learning_rate,
                alpha=0.5,
                k=6,
                weight_decay=self.hparams.weight_decay,
            )                
        else:
            raise NotImplementedError("Not a valid optimizer configuration.")

    def get_scheduler(self, optimizer) -> object:
        if 'step' == self.hparams.scheduler:
            return StepLR(optimizer, step_size=self.hparams.step_size, gamma=self.hparams.step_gamma)
        elif 'step+warmup' == self.hparams.scheduler:
            scheduler_steplr = StepLR(optimizer, step_size=self.hparams.step_size, gamma=self.hparams.step_gamma)
            scheduler_warmup = GradualWarmupScheduler(
                optimizer,
                multiplier=self.hparams.warmup_factor,
                total_epoch=self.hparams.warmup_epochs,
                after_scheduler=scheduler_steplr)
            return scheduler_warmup
        elif "cosine" == self.hparams.scheduler:
            return CosineAnnealingLR(optimizer, T_max=self.hparams.tmax)
        elif "cosine+warmup" == self.hparams.scheduler:
            cosine = CosineAnnealingLR(
                optimizer, T_max=self.hparams.tmax)
            return GradualWarmupScheduler(
                optimizer,
                multiplier=self.hparams.warmup_factor,
                total_epoch=self.hparams.warmup_epochs,
                after_scheduler=cosine
            )
        elif "cosine+restarts":
            return CosineAnnealingWarmRestarts(
                optimizer,
                T_0=self.hparams.tzero, T_mult=self.hparams.tmult
                )
        elif "cyclic":
            scheduler = CyclicLR(
                optimizer, 
                base_lr=self.learning_rate, 
                mode='exp_range',
                max_lr=self.learning_rate*self.hparams.max_lr_factor)
            return scheduler
        else:
            raise NotImplementedError("Not a valid scheduler configuration.")

    def get_criterion(self):
        if "mse" == self.hparams.criterion:
            return nn.MSELoss()

    def get_net(self):
        print('Using imagenet init: {}'.format(self.hparams.use_imagenet_init))
        if self.hparams.type_agg == 'minmaxmean':
            in_ch = 3
        elif self.hparams.type_agg == 'simple':
            in_ch = self.hparams.load_n
        return WindModel.net_mapping(
            self.hparams.model_name,
            in_ch,
            self.hparams.use_imagenet_init
            )

    def load_weights_from_checkpoint(self, checkpoint: str) -> None:
        """ Function that loads the weights from a given checkpoint file.
        Note:
            If the checkpoint model architecture is different then `self`, only
            the common parts will be loaded.
        :param checkpoint: Path to the checkpoint containing the weights to be loaded.
        """
        if checkpoint.endswith('ckpt'):
            checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage,)
            pretrained_dict = checkpoint["state_dict"]
            model_dict = self.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 3. load the new state dict
            self.load_state_dict(pretrained_dict)
        elif checkpoint.endswith('pth'):
            pretrained_dict = torch.load(checkpoint)
            model_dict = self.state_dict()
            pretrained_dict = {'net.'+k: v for k, v in pretrained_dict.items() if 'net.'+k in model_dict}
            model_dict.update(pretrained_dict)
            self.load_state_dict(pretrained_dict)
