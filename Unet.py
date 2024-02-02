import os
import logging
from argparse import ArgumentParser
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch import optim
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from loss import Weighted_Cross_Entropy_Loss, dice_loss

import pytorch_lightning as pl

from dataset_loops import DirDataset


class Unet(pl.LightningModule):
    def __init__(self, hparams):
        super(Unet, self).__init__()
        # self.hparams = hparams
        #for key in hparams.keys():
        #    self.hparams[key] = hparams[key]
        self.dataset = hparams.dataset
        self.save_hyperparameters()

        self.n_channels = hparams.n_channels
        self.n_classes = hparams.n_classes
        self.n_filters = hparams.n_filters
        self.n_augment = hparams.augmentation
        self.bilinear = True
        self.loss = Weighted_Cross_Entropy_Loss()

        def double_conv(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        def down(in_channels, out_channels):
            return nn.Sequential(
                nn.MaxPool2d(2),
                double_conv(in_channels, out_channels)
            )

        class up(nn.Module):
            def __init__(self, in_channels, out_channels, bilinear=False):
                super().__init__()

                if bilinear:
                    self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                else:
                    #self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2,
                    #                            kernel_size=2, stride=2)
                    self.up = nn.ConvTranspose2d(in_channels, in_channels // 2,
                                                kernel_size=2, stride=2)

                self.conv = double_conv(in_channels, out_channels)

            def forward(self, x1, x2):
                x1 = self.up(x1)
                # [?, C, H, W]
                diffY = x2.size()[2] - x1.size()[2]
                diffX = x2.size()[3] - x1.size()[3]

                x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                                diffY // 2, diffY - diffY // 2])
                x = torch.cat([x2, x1], dim=1) ## why 1?
                return self.conv(x)

        self.inc = double_conv(self.n_channels, self.n_filters)
        self.down1 = down(self.n_filters, 2*self.n_filters)
        self.down2 = down(2*self.n_filters, 4*self.n_filters)
        self.down3 = down(4*self.n_filters, 8*self.n_filters)
        # self.down4 = down(8*self.n_filters, 16*self.n_filters)
        # self.up1 = up(16*self.n_filters, 8*self.n_filters)
        self.up2 = up(8*self.n_filters, 4*self.n_filters)
        self.up3 = up(4*self.n_filters, 2*self.n_filters)
        self.up4 = up(2*self.n_filters, self.n_filters)
        self.out = nn.Conv2d(self.n_filters, self.n_classes, kernel_size=1)
        # self.out = nn.Sequential(nn.Conv2d(self.n_filters, self.n_classes, kernel_size=1), nn.Sigmoid())

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        # x5 = self.down4(x4)
        # x = self.up1(x5, x4)
        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.out(x)

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y) if self.n_classes > 1 else \
            F.binary_cross_entropy_with_logits(y_hat, y)
        #loss = self.loss(y_hat, y)
        #loss = dice_loss(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y) if self.n_classes > 1 else \
            F.binary_cross_entropy_with_logits(y_hat, y)
        # loss = self.loss(y_hat, y)
        #loss = dice_loss(y_hat, y)
        # self.log("val_loss", torch.tensor([loss]))
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        self.log('val_loss', avg_loss)
        return {'val_loss': avg_loss, 'log': tensorboard_logs}
    
    def transform_img(self):
        # transforms = []
        #transforms.append(transforms.PILToTensor())
        #transforms.append(transforms.ConvertImageDtype(torch.float))
        #
        # transforms.append(transforms.RandomHorizontalFlip(0.5))
        # transforms.append(transforms.RandomVerticalFlip(0.5))
        # transforms.ColorJitter(), 
        return transforms.Compose([transforms.RandomAdjustSharpness(sharpness_factor=1)])


    def transform_stack(self):
        # transforms = []
        #transforms.append(transforms.PILToTensor())
        #transforms.append(transforms.ConvertImageDtype(torch.float))
        #
        # transforms.append(transforms.RandomHorizontalFlip(0.5))
        # transforms.append(transforms.RandomVerticalFlip(0.5))
        return transforms.Compose([transforms.RandomHorizontalFlip(0.5), transforms.RandomVerticalFlip(0.5)])

    def configure_optimizers(self):
        return torch.optim.RMSprop(self.parameters(), lr=0.1, weight_decay=1e-8)

    def __dataloader(self):
        dataset = self.dataset
        dataset = DirDataset(f'{dataset}/', f'{dataset}/', transforms_stack=None,
                             transforms_img=self.transform_img())
        n_val = int(len(dataset) * 0.2)
        n_train = len(dataset) - n_val
        print(n_train, n_val)
        train_ds, val_ds = random_split(dataset, [n_train, n_val])
        train_loader = DataLoader(train_ds, batch_size=2, pin_memory=True, shuffle=True, num_workers=8)
        val_loader = DataLoader(val_ds, batch_size=2, pin_memory=True, shuffle=False, num_workers=8)

        return {
            'train': train_loader,
            'val': val_loader,
        }

    # @pl.data_loader
    def train_dataloader(self):
        return self.__dataloader()['train']

    # @pl.data_loader
    def val_dataloader(self):
        return self.__dataloader()['val']

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser])

        parser.add_argument('--n_channels', type=int, default=1)
        parser.add_argument('--n_classes', type=int, default=1)
        parser.add_argument('--n_filters', type=int, default=64)
        parser.add_argument('--augmentation', type=int, default=1)
        return parser
