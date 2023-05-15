import os
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
import time
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
from torchvision import transforms
from autoencoder import load_autoencoder
from dataset import LensingDataset
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.trainer import Trainer
from lightning.pytorch import seed_everything

NUM_WORKERS = int(os.environ["SLURM_CPUS_PER_TASK"])
NUM_NODES = int(os.environ["SLURM_NNODES"])
ALLOCATED_GPUS_PER_NODE = int(os.environ["SLURM_GPUS_ON_NODE"])

def main(config):
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomRotation(180),
        transforms.CenterCrop(180),
        transforms.Resize(256, antialias=True),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    timestr = time.strftime("%Y%m%d-%H%M%S")
    save_dir = config['save_dir'] + 'autoencoder_' + timestr
    
    print("Loading train dataset!")
    start = time.time()
    train_dataset = LensingDataset(config['dataset'], transform)
    end = time.time()
    print(f"dataset loaded in {end - start:.3f} s")
    
    val_len = int(config['parameters']['val_size'] * len(train_dataset))
    train_len = len(train_dataset) - val_len
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_len, val_len])

    train_dataloader = DataLoader(train_dataset, batch_size=config['parameters']['batch_size'], num_workers = int(os.environ['SLURM_CPUS_PER_TASK']), pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config['parameters']['batch_size'], num_workers = int(os.environ['SLURM_CPUS_PER_TASK']), pin_memory=True)

    
    model = load_autoencoder(config=config)

    wand_logger = WandbLogger(project='DE-2-Lensing')

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(save_dir, 'checkpoints'),
        verbose=True,
        save_top_k=2,
        monitor="val_loss",
        mode="min",
        save_on_train_epoch_end=True,
        auto_insert_metric_name=True,
        filename='autoencoder-{epoch:02d}-{val_loss:.2f}'
    )

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=config['parameters']['early_stopping'],
        verbose=True
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = Trainer(devices=2,
    		num_nodes=2,
		strategy='ddp',
		accelerator='auto',
		max_epochs=config['parameters']['epochs'],
		callbacks=[checkpoint_callback, early_stop_callback, lr_monitor], 
		logger=wand_logger, 
		deterministic='warn')

    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description = 'Train the models')
    parser.add_argument('--config', metavar = 'config', required=True,
                    help='Location of the config file')

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    seed_everything(42, workers=True)
    main(config)
    
