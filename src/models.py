#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torchvision
# from torchvision.models.detection import FasterRCNN
# from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np
import gc


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
    Handles both Task 1 (digit detection) and Task 2 (number recognition).
    """

    def __init__(self, cfg):
        super().__init__()

        # Build the base Faster R-CNN model
        self.model = build_faster_rcnn_model(cfg)
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

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
            targets = [{k: v.to(self.device)
                        for k, v in t.items()} for t in targets]

        # Forward pass through the model
        output = self.model(images, targets)

        return output

    def predict_number(self, detections):
        """
        Predict the entire number from individual digit detections (Task 2).

        Args:
            detections: Dict with 'boxes', 'labels', and 'scores' for a image

        Returns:
            predicted_number: int, the number or -1 if no digits detected
        """
        if len(detections['boxes']) == 0:
            return -1

        # Extract boxes and labels
        boxes = detections['boxes'].cpu().numpy()
        labels = detections['labels'].cpu().numpy()
        scores = detections['scores'].cpu().numpy()

        # Filter by confidence threshold
        confidence_threshold = 0.7
        confident_indices = scores > confidence_threshold
        if not any(confident_indices):
            return -1

        boxes = boxes[confident_indices]
        labels = labels[confident_indices]
        scores = scores[confident_indices]

        # If we only have one digit, return it directly
        if len(boxes) == 1:
            return int(labels[0])

        # Calculate centers of each box
        centers = np.zeros((len(boxes), 2))
        for i, box in enumerate(boxes):
            centers[i] = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]

        # Calculate box dimensions
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        avg_width = np.mean(widths)
        avg_height = np.mean(heights)

        # Determine if digits are arranged horizontally or vertically
        x_span = np.max(centers[:, 0]) - np.min(centers[:, 0])
        y_span = np.max(centers[:, 1]) - np.min(centers[:, 1])

        # Analyze orientation: horizontal, vertical, or diagonal
        is_horizontal = x_span > y_span

        # Calculate spacing threshold based on average size
        spacing_threshold = max(avg_width, avg_height) * 1.5

        # Cluster digits that belong to the same number
        clusters = []
        visited = set()

        for i in range(len(boxes)):
            if i in visited:
                continue

            cluster = [i]
            visited.add(i)

            # Find all connected digits
            queue = [i]
            while queue:
                current = queue.pop(0)

                for j in range(len(boxes)):
                    if j in visited:
                        continue

                    # Calculate distance between centers
                    dist = np.linalg.norm(centers[current] - centers[j])

                    if dist < spacing_threshold:
                        cluster.append(j)
                        visited.add(j)
                        queue.append(j)

            clusters.append(cluster)

        # Process each cluster to form a number
        result = -1
        max_cluster_size = 0

        for cluster in clusters:
            if len(cluster) > max_cluster_size:
                max_cluster_size = len(cluster)

                # Get boxes and labels for this cluster
                # cluster_boxes = boxes[cluster]
                cluster_labels = labels[cluster]
                cluster_centers = centers[cluster]

                # Sort indices based on orientation
                if is_horizontal:
                    # Sort from left to right
                    sorted_indices = np.argsort(cluster_centers[:, 0])
                else:
                    # Sort from top to bottom
                    sorted_indices = np.argsort(cluster_centers[:, 1])

                # Sort labels
                sorted_labels = cluster_labels[sorted_indices]

                # Combine digits to form the number
                digits = [str(int(label)) for label in sorted_labels]
                number_str = ''.join(digits)

                try:
                    result = int(number_str)
                except ValueError:
                    result = -1

        return result

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

            # Force garbage collection after processing a batch
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return detections, numbers
