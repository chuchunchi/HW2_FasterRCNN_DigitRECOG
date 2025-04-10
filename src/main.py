#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import torch
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
from PIL import Image
import torchvision.transforms as T
import gc

# Configuration
class Config:
    # Data settings
    data_root_dir = 'data'
    
    train_dir = os.path.join(data_root_dir, 'train')
    val_dir = os.path.join(data_root_dir, 'valid')
    test_dir = os.path.join(data_root_dir, 'test')
    train_annotations = os.path.join(data_root_dir, 'train.json')
    val_annotations = os.path.join(data_root_dir, 'valid.json')
    img_size = (512, 512)
    num_classes = 10  # (0-9)
    
    # Faster R-CNN settings
    rpn_anchor_sizes = (32, 64, 128, 256, 512)
    rpn_aspect_ratios = (0.5, 1.0, 2.0)
    rpn_nms_thresh = 0.7
    rpn_fg_iou_thresh = 0.7
    rpn_bg_iou_thresh = 0.3
    box_score_thresh = 0.05
    box_nms_thresh = 0.5
    box_detections_per_img = 100
    
    # Training settings
    batch_size = 8
    num_workers = 2
    learning_rate = 0.001
    weight_decay = 0.0001
    momentum = 0.9
    num_epochs = 24
    
    # Output settings
    output_dir = 'outputs'
# Dataset class
class DigitDataset(Dataset):
    """Dataset for digit detection and recognition."""
    
    def __init__(self, img_dir, annotations_file=None, transform=None, is_test=False):
        self.transform = transform
        self.is_test = is_test
        
        # Get all image files
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith('.png')]
        self.img_dir = img_dir
        self.img_files.sort()
        
        # Initialize COCO API for annotations
        self.coco = None
        self.annotations = {}
        
        if annotations_file is not None and os.path.exists(annotations_file):
            self.coco = COCO(annotations_file)
            self.img_ids = list(sorted(self.coco.imgs.keys()))
            
            # Map file name to image id
            self.file_to_id = {self.coco.imgs[img_id]['file_name']: img_id 
                               for img_id in self.img_ids}
            
            # Preprocess annotations
            self._preprocess_annotations()
    
    def _preprocess_annotations(self):
        """Preprocess COCO annotations for faster access."""
        for img_id in self.img_ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            if len(ann_ids) > 0:
                anns = self.coco.loadAnns(ann_ids)
                
                # Extract boxes, labels, and areas
                boxes = []
                labels = []
                areas = []
                
                for ann in anns:
                    # COCO bbox format: [x_min, y_min, width, height]
                    x, y, w, h = ann['bbox']
                    
                    # Convert to [x_min, y_min, x_max, y_max] format
                    boxes.append([x, y, x + w, y + h])
                    
                    # Convert category_id to 0-9 (COCO starts from 1)
                    labels.append(ann['category_id'] - 1)  # Convert to 0-indexed
                    areas.append(w * h)
                
                self.annotations[img_id] = {
                    'boxes': torch.as_tensor(boxes, dtype=torch.float32),
                    'labels': torch.as_tensor(labels, dtype=torch.int64),
                    'areas': torch.as_tensor(areas, dtype=torch.float32),
                    'iscrowd': torch.zeros(len(boxes), dtype=torch.int64)
                }
            else:
                # Empty annotations
                self.annotations[img_id] = {
                    'boxes': torch.zeros((0, 4), dtype=torch.float32),
                    'labels': torch.zeros(0, dtype=torch.int64),
                    'areas': torch.zeros(0, dtype=torch.float32),
                    'iscrowd': torch.zeros(0, dtype=torch.int64)
                }
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        img_file = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_file)
        
        # Open image and convert to RGB, making sure to close it after processing
        with Image.open(img_path) as img:
            img = img.convert('RGB')
            # Apply transforms to a copy of the image
            if self.transform is not None:
                img_tensor = self.transform(img)
        
        # Get image id and annotations
        if self.coco is not None and img_file in self.file_to_id:
            img_id = self.file_to_id[img_file]
            target = self.annotations[img_id].copy()
            target['image_id'] = torch.tensor([img_id])
        else:
            # For test set where annotations are not available
            img_id = int(os.path.splitext(img_file)[0])  # image_id = filename
            target = {
                'image_id': torch.tensor([img_id]),
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.zeros(0, dtype=torch.int64),
                'areas': torch.zeros(0, dtype=torch.float32),
                'iscrowd': torch.zeros(0, dtype=torch.int64)
            }
        
        return img_tensor, target


# Custom collate function for the DataLoader
def collate_fn(batch):
    return tuple(zip(*batch))


# Get transforms for the datasets
def get_transforms(train=False):
    transforms = []
    # Add transforms for both train and test
    transforms.append(T.ToTensor())
    transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    
    # Add data augmentation for training
    if train:
        transforms = [
            T.RandomRotation(10),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ] + transforms
        
    return T.Compose(transforms)


# Set random seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Command-line argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description='Digit Recognition with Faster R-CNN')
    parser.add_argument('--mode', type=str, choices=['train', 'eval', 'inference'], default='train',
                        help='Running mode')
    parser.add_argument('--checkpoint', type=str, default=None, help='Model checkpoint path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser.parse_args()


if __name__ == '__main__':
    # Parse command-line arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(Config.output_dir, exist_ok=True)
    
    # Import the appropriate module based on the mode
    if args.mode == 'train':
        from train import train
        train(Config, args.checkpoint)
    elif args.mode == 'eval':
        from evaluate import evaluate
        evaluate(Config, args.checkpoint, mode='val')
    elif args.mode == 'inference':
        from evaluate import inference
        inference(Config, args.checkpoint)
    
    # Force garbage collection at the end
    gc.collect()