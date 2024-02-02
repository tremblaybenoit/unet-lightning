import os
from argparse import ArgumentParser

import numpy as np
import torch

from Unet import Unet
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


def main(hparams):
    model = Unet(hparams)

    os.makedirs(hparams.log_dir, exist_ok=True)
    print(hparams.log_dir)
    try:
        log_dir = sorted(os.listdir(hparams.log_dir))[-1]
    except IndexError:
        log_dir = os.path.join(hparams.log_dir, 'version_0')
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss', 
        mode='min',
        dirpath=os.path.join(log_dir, 'weighted_entropy'),
        save_top_k=1, 
        verbose=True,
    )
    stop_callback = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=10,
        verbose=True,
    )
    trainer = Trainer(
        accelerator='gpu', 
        log_every_n_steps=40, 
        callbacks=[stop_callback, checkpoint_callback]
    )

    trainer.fit(model)


if __name__ == '__main__':
    parent_parser = ArgumentParser(add_help=False)
    parent_parser.add_argument('--dataset', required=True)
    parent_parser.add_argument('--log_dir', default='lightning_logs')

    parser = Unet.add_model_specific_args(parent_parser)
    hparams = parser.parse_args()

    main(hparams)
