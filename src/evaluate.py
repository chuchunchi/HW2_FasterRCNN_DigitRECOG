#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from main import DigitDataset, get_transforms, collate_fn
from models import DigitDetector


def evaluate(cfg, checkpoint_path=None, mode='val'):
    """
    Evaluate the model on validation set.
    
    Args:
        cfg: Configuration object
        checkpoint_path: Path to model checkpoint
        mode: 'val' or 'test'
    """
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create dataset and dataloader
    if mode == 'val':
        dataset = DigitDataset(
            img_dir=cfg.val_dir,
            annotations_file=cfg.val_annotations,
            transform=get_transforms(train=False)
        )
        has_annotations = True
    else:  # test mode
        dataset = DigitDataset(
            img_dir=cfg.test_dir,
            transform=get_transforms(train=False),
            is_test=True
        )
        has_annotations = False
    
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn
    )
    
    # Build model
    model = DigitDetector(cfg)
    model.to(device)
    
    # Load checkpoint
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        # Try to load best model by default
        best_model_path = os.path.join(cfg.output_dir, 'best_model.pth')
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=device)
            model.load_state_dict(checkpoint['model'])
            print(f"Loaded best model from {best_model_path}")
        else:
            print("No checkpoint provided. Using randomly initialized model.")
    
    # Switch to evaluation mode
    model.eval()
    
    # Evaluate
    if has_annotations and mode == 'val':
        # Calculate mAP using COCO API
        coco_results = []
        coco_gt = dataset.coco
        
        with torch.no_grad():
            for images, targets in tqdm(dataloader, desc="Evaluating"):
                images = [img.to(device) for img in images]
                outputs = model(images)
                
                # Convert outputs to COCO format
                for output, target in zip(outputs, targets):
                    image_id = target['image_id'].item()
                    boxes = output['boxes'].cpu().numpy()
                    scores = output['scores'].cpu().numpy()
                    labels = output['labels'].cpu().numpy()
                    
                    # Skip if no detections
                    if len(boxes) == 0:
                        continue
                    
                    # Convert boxes from [x1, y1, x2, y2] to [x, y, width, height]
                    coco_boxes = boxes.copy()
                    coco_boxes[:, 2] = coco_boxes[:, 2] - coco_boxes[:, 0]
                    coco_boxes[:, 3] = coco_boxes[:, 3] - coco_boxes[:, 1]
                    
                    for box, score, label in zip(coco_boxes, scores, labels):
                        coco_results.append({
                            'image_id': image_id,
                            'category_id': int(label) + 1,  # Convert back to 1-indexed for COCO
                            'bbox': box.tolist(),
                            'score': float(score)
                        })
        
        # Save detection results
        with open(os.path.join(cfg.output_dir, 'val_detections.json'), 'w') as f:
            json.dump(coco_results, f)
        
        # COCO evaluation
        coco_dt = coco_gt.loadRes(coco_results)
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        # Also evaluate Task 2 (number recognition) on validation set
        task2_eval(model, dataloader, device, dataset, cfg.output_dir)
    
    print("Evaluation complete!")


def task2_eval(model, dataloader, device, dataset, output_dir):
    """
    Evaluate Task 2 (number recognition) on validation set.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader for validation set
        device: Device to use
        dataset: Dataset with ground truth
        output_dir: Directory to save results
    """
    # Get ground truth numbers for validation set
    gt_numbers = {}
    
    # This would need to be implemented based on your dataset structure
    # For this example, we'll assume we can extract full numbers from annotations
    for img_id in dataset.coco.imgs:
        ann_ids = dataset.coco.getAnnIds(imgIds=img_id)
        anns = dataset.coco.loadAnns(ann_ids)
        
        if not anns:
            gt_numbers[img_id] = -1
            continue
        
        # Sort annotations from left to right
        anns.sort(key=lambda x: x['bbox'][0])
        
        # Combine digits to form the number
        digits = [str(ann['category_id'] - 1) for ann in anns]
        number = int(''.join(digits))
        
        gt_numbers[img_id] = number
    
    # Predict numbers using the model
    pred_numbers = {}
    
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Evaluating Task 2"):
            images = [img.to(device) for img in images]
            outputs = model(images)
            
            for output, target in zip(outputs, targets):
                image_id = target['image_id'].item()
                pred_number = model.predict_number(output)
                pred_numbers[image_id] = pred_number
    
    # Calculate accuracy
    correct = 0
    total = 0
    
    for img_id in gt_numbers:
        if img_id in pred_numbers:
            total += 1
            if gt_numbers[img_id] == pred_numbers[img_id]:
                correct += 1
    
    accuracy = correct / total if total > 0 else 0
    print(f"Task 2 Accuracy: {accuracy:.4f} ({correct}/{total})")
    
    # Save results
    results = {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'predictions': pred_numbers
    }
    
    with open(os.path.join(output_dir, 'task2_results.json'), 'w') as f:
        json.dump(results, f)


def inference(cfg, checkpoint_path=None):
    """
    Run inference on test set and create submission files.
    
    Args:
        cfg: Configuration object
        checkpoint_path: Path to model checkpoint
    """
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create dataset and dataloader for test set
    test_dataset = DigitDataset(
        img_dir=cfg.test_dir,
        transform=get_transforms(train=False),
        is_test=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn
    )
    
    # Build model
    model = DigitDetector(cfg)
    model.to(device)
    
    # Load checkpoint
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        # Try to load best model by default
        best_model_path = os.path.join(cfg.output_dir, 'best_model.pth')
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=device)
            model.load_state_dict(checkpoint['model'])
            print(f"Loaded best model from {best_model_path}")
        else:
            print("No checkpoint provided. Using randomly initialized model.")
    
    # Switch to evaluation mode
    model.eval()
    
    # Run inference for Task 1 (detection)
    coco_results = []
    image_ids = []
    
    # For Task 2 (number recognition)
    task2_results = []
    
    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="Running inference"):
            images = [img.to(device) for img in images]
            outputs = model(images)
            
            # Process each image
            for output, target in zip(outputs, targets):
                image_id = target['image_id'].item()
                image_ids.append(image_id)
                
                # Task 1: Object detection results
                boxes = output['boxes'].cpu().numpy()
                scores = output['scores'].cpu().numpy()
                labels = output['labels'].cpu().numpy()
                
                # TODO: test bbox threshold
                # socre_threshold = 0.3
                # high_score_indices = scores > socre_threshold
                # boxes = boxes[high_score_indices]
                # scores = scores[high_score_indices]
                # labels = labels[high_score_indices]
                
                # Convert boxes from [x1, y1, x2, y2] to [x, y, width, height]
                if len(boxes) > 0:
                    coco_boxes = boxes.copy()
                    coco_boxes[:, 2] = coco_boxes[:, 2] - coco_boxes[:, 0]
                    coco_boxes[:, 3] = coco_boxes[:, 3] - coco_boxes[:, 1]
                    
                    for box, score, label in zip(coco_boxes, scores, labels):
                        coco_results.append({
                            'image_id': image_id,
                            'category_id': int(label) + 1,  # Convert back to 1-indexed for COCO
                            'bbox': box.tolist(),
                            'score': float(score)
                        })
                
                # Task 2: Number recognition
                pred_number = model.predict_number(output)
                task2_results.append({
                    'image_id': image_id,
                    'pred_label': pred_number
                })
    
    # Save Task 1 results (pred.json)
    with open(os.path.join(cfg.output_dir, 'pred.json'), 'w') as f:
        json.dump(coco_results, f)
    
    # Save Task 2 results (pred.csv)
    task2_df = pd.DataFrame(task2_results)
    task2_df.to_csv(os.path.join(cfg.output_dir, 'pred.csv'), index=False)
    
    print(f"Inference complete! Results saved to {cfg.output_dir}")


if __name__ == "__main__":
    from main import Config, set_seed
    set_seed()
    
    # Default to evaluation mode
    evaluate(Config, mode='val')
