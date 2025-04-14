#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import gc
import shutil
from collections import OrderedDict

from main import DigitDataset, get_transforms, collate_fn
from models import DigitDetector

def train_one_epoch_sgd(model, data_loader, optimizer, device, epoch, print_freq=100):
    metric_logger = MetricLogger()
    model.train()
    
    for i, (images, targets) in enumerate(data_loader):
        # Move inputs to the device
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        # Backward pass and optimize
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        # Log progress
        metric_logger.update(loss=losses.item(), **{k: v.item() for k, v in loss_dict.items()})
        
        if i % print_freq == 0:
            print(f'Epoch: [{epoch}][{i}/{len(data_loader)}] {metric_logger}')
            
        # Clear memory
        del images, targets, loss_dict, losses
        if i % 10 == 0:  # Run garbage collection every 10 batches
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return metric_logger.meters['loss'].avg

def train_one_epoch(model, data_loader, optimizer, lr_scheduler, device, epoch, print_freq=10):
    """Train the model for one epoch."""
    model.train()
    metric_logger = MetricLogger()
    
    for i, (images, targets) in enumerate(data_loader):
        # Move inputs to the device
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        # Backward pass and optimize
        optimizer.zero_grad()
        losses.backward()
        
        # Gradient clipping is often helpful with AdamW
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Step the scheduler (for OneCycleLR)
        lr_scheduler.step()
        
        # Log progress
        metric_logger.update(loss=losses.item(), **{k: v.item() for k, v in loss_dict.items()})
        
        if i % print_freq == 0:
            print(f'Epoch: [{epoch}][{i}/{len(data_loader)}] {metric_logger} LR: {optimizer.param_groups[0]["lr"]:.6f}')
    
    return metric_logger.meters['loss'].avg

def train(cfg, checkpoint_path=None):
    """
    Train the Faster R-CNN model.
    
    Args:
        cfg: Configuration object
        checkpoint_path: Optional path to a checkpoint to resume training
    """
    # Force garbage collection before starting
    gc.collect()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create transforms
    train_transform = get_transforms(train=True)
    val_transform = get_transforms(train=False)
    
    # Create datasets
    train_dataset = DigitDataset(
        img_dir=cfg.train_dir,
        annotations_file=cfg.train_annotations,
        transform=train_transform
    )
    
    val_dataset = DigitDataset(
        img_dir=cfg.val_dir,
        annotations_file=cfg.val_annotations,
        transform=val_transform
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=False  # Set to False to reduce memory usage
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=False  # Set to False to reduce memory usage
    )
    
    # Build model
    model = DigitDetector(cfg)
    model.to(device)
    
    # Create optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    if cfg.optimizer == "SGD":
        optimizer = optim.SGD(
            params,
            lr=cfg.learning_rate,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay
        )
    elif cfg.optimizer == "Adam":
        optimizer = torch.optim.AdamW(
            params,
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay  # AdamW applies proper weight decay
        )
        
    # Learning rate scheduler
    if cfg.lr == "step":
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=8,
            gamma=0.1
        )
    else:
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg.learning_rate,
            total_steps=cfg.num_epochs * len(train_loader),
            pct_start=0.1,  # Spend 10% of training time warming up
            anneal_strategy='cos',
            div_factor=25.0   # Initial LR is max_lr/25
        )
    
    # Load checkpoint if provided
    start_epoch = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Loaded checkpoint from {checkpoint_path}, resuming from epoch {start_epoch}")
    
    # Training loop
    num_epochs = cfg.num_epochs
    best_loss = float('inf')
    
    print(f"Starting training for {num_epochs} epochs...")
    for epoch in range(start_epoch, num_epochs):
        # Train for one epoch
        if cfg.optimizer == "SGD":
            train_loss = train_one_epoch_sgd(model, train_loader, optimizer, device, epoch)
        else:
            train_loss = train_one_epoch(model, train_loader, optimizer, lr_scheduler, device, epoch)
        
        # Update learning rate
        lr_scheduler.step()
        
        # Evaluate on validation set
        val_loss = evaluate_loss(model, val_loader, device)
        
        print(f"Epoch {epoch} complete: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        # Save checkpoint
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'val_loss': val_loss
        }
        
        # Save the best model
        if val_loss < best_loss:
            best_loss = val_loss
            best_path = os.path.join(cfg.output_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"Saved best model with val_loss: {val_loss:.4f}")
        
        # Save epoch checkpoint
        epoch_path = os.path.join(cfg.output_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, epoch_path)
        
        
        # Always run garbage collection between epochs
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Clear references to possible memory leaks
        del checkpoint
        
        # Break if out of memory issue is detected
        if torch.cuda.is_available() and torch.cuda.memory_allocated() > 0.95 * torch.cuda.get_device_properties(0).total_memory:
            print("Warning: GPU memory nearly exhausted, stopping training.")
            break
    
    print("Training complete!")


def evaluate_loss(model, data_loader, device):
    """Evaluate model loss on the validation set."""
    # Save the current model mode
    was_training = model.training
    
    # Set model to training mode temporarily to get losses
    model.train()
    metric_logger = MetricLogger()
    
    with torch.no_grad():
        for i, (images, targets) in enumerate(data_loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Forward pass (in training mode to get losses)
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            # Log metrics
            metric_logger.update(loss=losses.item(), **{k: v.item() for k, v in loss_dict.items()})
            
            # Clear memory
            del images, targets, loss_dict, losses
            if i % 1000 == 0:  # Run garbage collection periodically
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Restore original model mode
    if not was_training:
        model.eval()
    
    return metric_logger.meters['loss'].avg


class MetricLogger:
    """Simple class to log and compute average metrics."""
    
    def __init__(self):
        self.meters = {}
        self.delimiter = '  '
    
    def update(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self.meters:
                self.meters[k] = AverageMeter()
            self.meters[k].update(v)
    
    def __str__(self):
        return self.delimiter.join(
            f"{name}: {meter.avg:.4f}" for name, meter in self.meters.items()
        )


class AverageMeter:
    """Compute and store the average and current value."""
    
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == "__main__":
    from main import Config, set_seed
    set_seed()
    train(Config)