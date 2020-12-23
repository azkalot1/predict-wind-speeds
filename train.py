import datetime
from argparse import ArgumentParser, Namespace, ArgumentTypeError

import torch
from pytorch_lightning import Trainer, loggers, seed_everything
from pytorch_lightning.callbacks import (
    ModelCheckpoint, EarlyStopping)
from pytorch_lightning.callbacks import LearningRateMonitor
from src.pl_model import WindModel
import os

SEED = 12
seed_everything(12)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')


def main(hparams: Namespace):
    now = datetime.datetime.now().strftime("%d.%H")
    if hparams.experiment_name is None:
        experiment_name = f"{now}_{hparams.model_name}_{hparams.optimizer}_{hparams.training_transforms}_{hparams.load_n}_fold_{hparams.fold}"
    else:
        experiment_name = f"{now}_{hparams.experiment_name}"
    model = WindModel(hparams=hparams)

    if hparams.load_weights is not None:
        print(f'Restoring checkpoint {hparams.load_weights}')
        model.load_weights_from_checkpoint(hparams.load_weights)

    pl_logger = loggers.neptune.NeptuneLogger(
            api_key=os.getenv('NEPTUNE_API_TOKEN'),
            experiment_name=experiment_name,
            params=vars(hparams),
            project_name='azkalot1/wind-speed'
            )

    callbacks = [LearningRateMonitor(logging_interval='epoch')]
    checkpoint_callback = ModelCheckpoint(
        filepath=f"logs/{experiment_name}/{experiment_name}_" + "best_{val_loss:.3f}",
        monitor='val_loss', save_top_k=5, mode='min', save_last=True)
    early_stop_callback = EarlyStopping(
        monitor='val_loss', patience=10, mode='min', verbose=True)

    # a weird way to add arguments to Trainer constructor, but we'll take it
    hparams.__dict__['logger'] = pl_logger
    hparams.__dict__['callbacks'] = callbacks
    hparams.__dict__['checkpoint_callback'] = checkpoint_callback
    hparams.__dict__['early_stop_callback'] = early_stop_callback

    trainer = Trainer.from_argparse_args(hparams)

    trainer.fit(model)

    # to make submission without lightning
    torch.save(model.net.state_dict(), f"logs/{experiment_name}.pth")


if __name__ == "__main__":
    # TODO: move configuration to *.yaml with Hydra
    parser = ArgumentParser(add_help=False)

    parser.add_argument("--image_folder", default='./data/all_images/')
    parser.add_argument("--data_folder", default='./data/')
    parser.add_argument("--load_n", default=1, type=int)
    parser.add_argument("--fold", default=0, type=int)
    parser.add_argument("--experiment_name", default=None)
    parser.add_argument("--load_weights", default=None, type=str)
    parser.add_argument("--training_transforms", default="light")
    parser.add_argument("--resize", default=256, type=int)
    parser.add_argument("--use_mixup", default=False, type=str2bool)
    parser.add_argument("--mixup_alpha", default=1.0, type=float)
    parser.add_argument("--profiler", default=False, type=str2bool)
    parser.add_argument("--fast_dev_run", default=False, type=str2bool)
    parser.add_argument("--auto_lr_find", default=False, type=str2bool)
    parser.add_argument("--use_imagenet_init", default=True, type=str2bool)
    parser.add_argument("--precision", default=16, type=int)
    parser.add_argument("--val_check_interval", default=1.0, type=float)
    parser.add_argument("--limit_train_batches", default=1.0, type=float)
    parser.add_argument("--limit_val_batches", default=1.0, type=float)
    # parser.add_argument("--distributed_backend", default="horovod", type=str)
    parser.add_argument("--gpus", default=1, type=int)
    parser.add_argument("--batch_size", default=6, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--deterministic", default=True, type=str2bool)
    parser.add_argument("--benchmark", default=True, type=str2bool)

    parser.add_argument("--model_name", default="resnet34", type=str)
    parser.add_argument("--criterion", default="mse", type=str)

    parser.add_argument("--optimizer", default="adamw", type=str)
    parser.add_argument("--sgd_momentum", default=0.9, type=float)
    parser.add_argument("--sgd_wd", default=1e-4, type=float)
    parser.add_argument("--learning_rate", default=3e-4, type=float)
    parser.add_argument("--weight_decay", default=1e-3, type=float)
    parser.add_argument("--gradient_clip_val", default=5, type=float)

    parser.add_argument("--scheduler", default="cosine+restarts", type=str)
    parser.add_argument("--warmup_epochs", default=10, type=int)
    parser.add_argument("--warmup_factor", default=1., type=int)
    parser.add_argument("--max_epochs", default=100, type=int)
    parser.add_argument("--factor", default=0.5, type=float)
    parser.add_argument("--tzero", default=10, type=int)
    parser.add_argument("--tmult", default=1, type=int)
    parser.add_argument("--scheduler_patience", default=5, type=int)
    parser.add_argument("--step_size", default=10, type=int)
    parser.add_argument("--step_gamma", default=0.1, type=float)
    parser.add_argument("--max_lr_factor", default=10, type=float)
    args = parser.parse_args()
    main(args)
