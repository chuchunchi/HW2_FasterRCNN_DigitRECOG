#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np


def build_faster_rcnn_model(cfg):
    """
    Build a Faster R-CNN model with the given configuration.
    
    Args:
        cfg: Configuration object
        
    Returns:
        model: A Faster R-CNN model
    """
    # Load pre-trained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=True,
        pretrained_backbone=True
    )
    
    # Replace the classifier with a new one for our number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features, 
        cfg.num_classes + 1  # +1 for background class
    )
    
    # Set RPN parameters
    model.rpn.nms_thresh = cfg.rpn_nms_thresh
    model.rpn.fg_iou_thresh = cfg.rpn_fg_iou_thresh
    model.rpn.bg_iou_thresh = cfg.rpn_bg_iou_thresh
    
    # Set RoI parameters
    model.roi_heads.score_thresh = cfg.box_score_thresh
    model.roi_heads.nms_thresh = cfg.box_nms_thresh
    model.roi_heads.detections_per_img = cfg.box_detections_per_img
    
    return model


class DigitDetector(nn.Module):
    """
    Digit detector model based on Faster R-CNN.
    Handles both Task 1 (digit detection) and Task 2 (whole number recognition).
    """
    
    def __init__(self, cfg):
        super().__init__()
        
        # Build the base Faster R-CNN model
        self.model = build_faster_rcnn_model(cfg)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def forward(self, images, targets=None):
        """
        Forward pass for training and inference.
        
        Args:
            images: List[Tensor] or Tensor, input images
            targets: Optional[List[Dict]], ground-truth boxes and labels
        
        Returns:
            losses: Dict, during training
            detections: List[Dict], during inference
        """
        # Ensure images are in a list format
        if isinstance(images, torch.Tensor):
            images = [images]
        
        # Move data to the correct device
        images = [img.to(self.device) for img in images]
        if targets is not None:
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
        
        # Forward pass through the model
        output = self.model(images, targets)
        
        return output
    
    def predict_number(self, detections):
        """
        Predict the entire number from individual digit detections (Task 2).
        
        Args:
            detections: Dict with 'boxes', 'labels', and 'scores' for a single image
        
        Returns:
            predicted_number: int, the predicted number or -1 if no digits detected
        """
        if len(detections['boxes']) == 0:
            return -1
        
        # Extract boxes and labels
        boxes = detections['boxes'].cpu().numpy()
        labels = detections['labels'].cpu().numpy()
        scores = detections['scores'].cpu().numpy()

        confident_indices = scores > 0.6
        if not any(confident_indices):
            return -1
        else:
            boxes = boxes[confident_indices]
            labels = labels[confident_indices]
            scores = scores[confident_indices]

        # Calculate the vertical center of each box
        y_centers = (boxes[:, 1] + boxes[:, 3]) / 2
        
        # Calculate average y-center
        mean_y = np.mean(y_centers)
        
        # Calculate height of each box
        heights = boxes[:, 3] - boxes[:, 1]
        avg_height = np.mean(heights)
        
        # threshold for vertical alignment
        y_threshold = avg_height * 0.3


            
        # Filter digits that are within a reasonable vertical range
        aligned_indices = np.abs(y_centers - mean_y) <= y_threshold
        
        # If we've filtered out too many digits, it might indicate multiple numbers in the image
        # In that case, we might need more sophisticated grouping
        if sum(aligned_indices) < len(boxes) * 0.5:
            
            # Sort boxes by x position
            x_centers = (boxes[:, 0] + boxes[:, 2]) / 2
            sorted_indices = np.argsort(x_centers)
            
            # currently: take the largest cluster of horizontally adjacent boxes
            # TODO: cluster?
        
        # Only keep digits that are vertically aligned
        boxes = boxes[aligned_indices]
        labels = labels[aligned_indices]
        
        # If we have no digits left after filtering, return -1
        if len(boxes) == 0:
            return -1
        
        # Sort remaining boxes from left to right
        x_lefts = boxes[:, 0]  # Left edge
        sorted_indices = np.argsort(x_lefts)

        sorted_labels = labels[sorted_indices]
    
        # Combine digits to form the number
        digits = [str(int(label)) for label in sorted_labels]
        number_str = ''.join(digits)
        
        try:
            return int(number_str)
        except ValueError:
            return -1

    
    def detect_and_recognize(self, images):
        """
        Perform both detection (Task 1) and recognition (Task 2).
        
        Args:
            images: List[Tensor] or Tensor, input images
        
        Returns:
            detections: List[Dict], detection results for each image
            numbers: List[int], recognized numbers for each image
        """
        # Switch to evaluation mode
        self.eval()
        
        with torch.no_grad():
            # Get detection results
            detections = self.forward(images)
            
            # Recognize numbers from detections
            numbers = []
            for detection in detections:
                number = self.predict_number(detection)
                numbers.append(number)
        
        return detections, numbers