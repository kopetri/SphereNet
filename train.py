import torch
import sys
import numpy as np
import json
import random
from pathlib import Path
from argparse import ArgumentParser
from module import ShpereNetModule
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataset import Structured3DDataset

def parse_ckpt(path):    
    return [p for p in Path(path).glob("**/*") if p.suffix == ".ckpt"][0].as_posix()

if __name__ == '__main__':
    parser = ArgumentParser('Train scatterplot model')
    parser.add_argument('--seed', default=None, type=int, help='Random Seed')
    parser.add_argument('--precision', default=16,   type=int, help='16 to use Mixed precision (AMP O2), 32 for standard 32 bit float training')
    parser.add_argument('--gpus', type=int, default=-1, help='Number of GPUs')
    parser.add_argument('--dev', action='store_true', help='Activate Lightning Fast Dev Run for debugging')
    parser.add_argument('--overfit', action='store_true', help='If this flag is set the network is overfit to 1 batch')
    parser.add_argument('--min_epochs', default=10, type=int, help='Minimum number of epochs.')
    parser.add_argument('--max_epochs', default=50, type=int, help='Maximum number ob epochs to train')
    parser.add_argument('--worker', default=6, type=int, help='Number of workers for data loader')
    parser.add_argument('--detect_anomaly', action='store_true', help='Enables pytorch anomaly detection')

    parser.add_argument('--learning_rate', default=1e-04, type=float, help='Learning rate')
    parser.add_argument('--dataset_path', required=True, type=str, help='Path to data set.')
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size')
    parser.add_argument('--dropout', default=0.5, type=float, help='Dropout')
    parser.add_argument('--learning_rate_decay', default=0.99999, type=float, help='Add learning rate decay.')
    parser.add_argument('--early_stop_patience', default=0, type=int, help='Stop training after n epochs with ne val_loss improvement.')
    parser.add_argument('--name', default="SphereNet", type=str, help='Name of output folder.')
    parser.add_argument('--shuffle', action='store_true', help="Enable shuffling points")
    

    args = parser.parse_args()

    if args.detect_anomaly:
        print("Enabling anomaly detection")
        torch.autograd.set_detect_anomaly(True)
    
    # windows safe
    if sys.platform in ["win32"]:
        args.worker = 0

    # Manage Random Seed
    if args.seed is None: # Generate random seed if none is given
        args.seed = random.randrange(4294967295) # Make sure it's logged
    pl.seed_everything(args.seed)

    callbacks = []

    if args.learning_rate_decay:
        callbacks += [pl.callbacks.lr_monitor.LearningRateMonitor()]

    callbacks += [pl.callbacks.ModelCheckpoint(
        verbose=True,
        save_top_k=1,
        filename='{epoch}-{valid_loss}',
        monitor='valid_loss',
        mode='min'
    )]

    if args.early_stop_patience > 0:
        callbacks += [pl.callbacks.EarlyStopping(
            monitor='valid_loss',
            min_delta=0.00,
            patience=args.early_stop_patience,
            verbose=True,
            mode='min'
        )]

    use_gpu = not args.gpus == 0

    trainer = pl.Trainer(
        log_gpu_memory=False,
        fast_dev_run=args.dev,
        profiler=False,
        gpus=args.gpus,
        log_every_n_steps=1,
        overfit_batches=1 if args.overfit else 0,
        precision=args.precision if use_gpu else 32,
        amp_level='O2' if use_gpu else None,
        min_epochs=args.min_epochs,
        max_epochs=args.max_epochs,
        logger=pl.loggers.WandbLogger(project="SphereNet", name=args.name),
        callbacks=callbacks
    )

    yaml = args.__dict__
    yaml.update({
            'random_seed': args.seed,
            'gpu_name': torch.cuda.get_device_name(0) if use_gpu else None,
            'gpu_capability': torch.cuda.get_device_capability(0) if use_gpu else None
            })

    train_dataset = Structured3DDataset(path=args.dataset_path, split="train", shuffle=args.shuffle)
    val_dataset   = Structured3DDataset(path=args.dataset_path, split="valid")
    

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.worker
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.worker
    )
    
    model = ShpereNetModule(args)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
